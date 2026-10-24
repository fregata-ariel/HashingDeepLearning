"""Pinned Idiap negative controls and maintained local CUDA/SBLocal composition.

TRACE_TEST_ID: SCATTERBRAIN-LOCAL-PRODUCT-CUDA
TRACE_TEST_ID: SCATTERBRAIN-LOCAL-HYBRID-CUDA
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[2]


def check(upstream: Path, build: Path, results: dict[str, object], reuse: bool = False) -> dict[str, object]:
    git = ["git", "-c", f"safe.directory={upstream}"]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    assert revision == "f22c13716fc748bb21a7b226ada7f7b5f87f867f"
    assert not subprocess.check_output(git + ["status", "--porcelain"], cwd=upstream, text=True).strip()
    build.mkdir(parents=True, exist_ok=reuse)
    manifest_path = build / "native-build.json"
    previous = json.loads(manifest_path.read_text()) if reuse else None
    fixture_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if previous is not None:
        assert previous["torch"] == torch.__version__ and previous["commit"] == revision
        assert previous["fixture_sha256"] == fixture_hash, "fixture changed since native build"
    sys.path[:0] = [str(upstream), str(ROOT / "ports/scatterbrain")]
    binaries: dict[str, str] = {}
    sources: dict[str, str] = {}
    results.update(source_commit=revision, source_sha256=sources, binary_sha256=binaries)

    def load_native(name: str, source: Path) -> ModuleType:
        key = "maintained:" + str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source.relative_to(upstream))
        sources[key] = hashlib.sha256(source.read_bytes()).hexdigest()
        target = build / name
        if previous is None:
            target.mkdir()
            module = load(name, [str(source)], build_directory=str(target), extra_cflags=["-O1", "-std=c++17"],
                          extra_cuda_cflags=["-O1", "-lineinfo", "-std=c++17"], verbose=True)
        else:
            assert previous["source_sha256"][key] == sources[key], "native source changed"
            binary = target / (name + ".so")
            assert hashlib.sha256(binary.read_bytes()).hexdigest() == previous["binary_sha256"][name], "native binary changed"
            spec = importlib.util.spec_from_file_location(name, binary)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("VERIFIED_NATIVE_REUSE", name, flush=True)
        assert isinstance(module, ModuleType) and module.__file__ is not None
        binaries[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        return module
    # Import the pinned autograd wrappers with real CPU and CUDA callees.
    # CPU modules are needed by the upstream wrapper import, but no CPU input
    # is used in this GPU fixture.
    for group in ("local_product", "causal_product"):
        for backend, suffix in (("cpu", "cpp"), ("cuda", "cu")):
            name = f"{group}_{backend}"
            source = upstream / "fast_transformers" / group / f"{name}.{suffix}"
            if group == "local_product" and backend == "cuda":
                original = load_native("local_product_cuda_original", source)
                source = ROOT / "ports/scatterbrain/cuda/local_product_cuda.cu"
            native = load_native(name, source)
            sys.modules[f"fast_transformers.{group}.{name}"] = native
    if previous is None:
        manifest_path.write_text(json.dumps({"torch":torch.__version__, "commit":revision, "fixture_sha256":fixture_hash,
            "source_sha256":sources, "binary_sha256":binaries},indent=2)+"\n")
    results["reused_verified_native_build"] = reuse
    native = importlib.import_module("fast_transformers.local_product")
    causal = importlib.import_module("fast_transformers.causal_product")
    assert native.LocalDotProduct.dot["cuda"] is sys.modules["fast_transformers.local_product.local_product_cuda"].local_dot_product
    errors: list[float] = []

    negative_controls = []
    for context, prefix in ((1, 2), (3, 3)):
        q0 = torch.tensor([1.,2.,3.], device="cuda").view(1,1,3,1)
        k0 = torch.tensor([4.,5.,6.], device="cuda").view(1,1,3,1)
        cot0 = torch.arange(1, 3*context+1, device="cuda", dtype=torch.float32).view(1,1,3,context)
        raw_q,raw_k = original.local_dot_backward(q0,k0,torch.tensor([prefix],device="cuda"),cot0,context)
        qref,kref = [t.cpu().double().requires_grad_() for t in (q0,k0)]
        loss = sum(qref[0,0,i,0]*kref[0,0,i-context//2+j,0]*cot0[0,0,i,j].cpu().double()
                   for i in range(3) for j in range(context) if 0 <= i-context//2+j < prefix)
        dq,dk = torch.autograd.grad(loss,(qref,kref))
        discrepancy = float((raw_q.cpu().double()-dq).abs().max()) if context == 1 else float((raw_k.cpu().double()-dk).abs().max())
        assert discrepancy > .1, "original negative control unexpectedly matched independent gradients"
        negative_controls.append({"contract":"padded gradient" if context==1 else "reversed key-gradient indexing",
                                  "observed_status":"FAIL", "max_abs_error":discrepancy,
                                  "actual":raw_q.cpu().flatten().tolist() if context==1 else raw_k.cpu().flatten().tolist(),
                                  "expected":dq.flatten().tolist() if context==1 else dk.flatten().tolist()})
    results["original_negative_controls"] = negative_controls

    def close(actual: torch.Tensor, expected: torch.Tensor, label: str, tolerance: float = 8e-6) -> None:
        assert actual.is_cuda and torch.isfinite(actual).all(), label
        error = float((actual.double().cpu() - expected.double().cpu()).abs().max())
        assert error < tolerance, f"{label}: {error} >= {tolerance}"
        errors.append(error)

    results["unequal_physical_lengths"] = "UNSUPPORTED: pinned CUDA reshapes keys using query sequence length (observed shape error in the initial probe)"
    for context in (1, 2, 3, 4, 9):
        n, s = 5, 5
        q = torch.randn(2, 2, n, 3, device="cuda", requires_grad=True)
        k = torch.randn(2, 2, s, 3, device="cuda", requires_grad=True)
        values = torch.randn(2, 2, s, 2, device="cuda", requires_grad=True)
        lengths = torch.tensor([s, s - 1], device="cuda")
        mask = torch.zeros(n, s, device="cuda")
        mask[2, 1] = -.75
        qr, kr, vr = [x.detach().cpu().double().requires_grad_() for x in (q, k, values)]
        reference = torch.stack([torch.stack([torch.stack([torch.stack([
            (qr[b, h, i] * kr[b, h, i-context//2+j]).sum() + mask[i, i-context//2+j].cpu().double()
            if 0 <= i-context//2+j < (s if b == 0 else s-1) else torch.tensor(-1e24, dtype=torch.float32).double()
            for j in range(context)]) for i in range(n)]) for h in range(2)]) for b in range(2)])
        actual = native.local_dot_product(q, k, mask, lengths, context)
        torch.cuda.synchronize()
        close(actual, reference, "local dot")
        cotangent = torch.randn_like(actual)
        ga = torch.autograd.grad((actual * cotangent).sum(), (q, k))
        ge = torch.autograd.grad((reference * cotangent.cpu().double()).sum(), (qr, kr))
        for a, e in zip(ga, ge):
            close(a, e, "local dot backward")
        weights = torch.randn_like(actual, requires_grad=True)
        wr = weights.detach().cpu().double().requires_grad_()
        reference_v = torch.stack([torch.stack([torch.stack([
            sum((wr[b,h,i,j] * vr[b,h,i-context//2+j] for j in range(context) if 0 <= i-context//2+j < s), torch.zeros(2, dtype=torch.float64))
            for i in range(n)]) for h in range(2)]) for b in range(2)])
        average = native.local_weighted_average(weights, values)
        torch.cuda.synchronize()
        close(average, reference_v, "weighted average")
        cotangent_v = torch.randn_like(average)
        ga = torch.autograd.grad((average * cotangent_v).sum(), (weights, values))
        ge = torch.autograd.grad((reference_v * cotangent_v.cpu().double()).sum(), (wr, vr))
        for a, e in zip(ga, ge):
            close(a, e, "weighted average backward")

    # Production Torch ops and maintained native launches share the caller's
    # nondefault stream, including source creation and asynchronous completion.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        q = torch.arange(5., device="cuda").view(1,1,5,1).contiguous().requires_grad_()
        k = torch.ones_like(q,requires_grad=True)
        out = native.local_dot_product(q,k,torch.zeros(5,5,device="cuda"),torch.tensor([5],device="cuda"),1)
        dq,dk = torch.autograd.grad(out.sum(),(q,k))
        stream.synchronize()
        close(out,q,"nondefault stream dot",1e-7)
        close(dq,torch.ones_like(q),"nondefault stream q gradient",1e-7)
        close(dk,q,"nondefault stream k gradient",1e-7)

    # Causal primitive is a separate acceptance, not inferred from local kernels.
    cq, ck, cv = [torch.randn(1, 1, 7, 4, device="cuda", requires_grad=True) for _ in range(3)]
    cqr, ckr, cvr = [x.detach().cpu().double().requires_grad_() for x in (cq, ck, cv)]
    expected_causal = torch.stack([sum(((cqr[0,0,i] * ckr[0,0,j]).sum() * cvr[0,0,j] for j in range(i+1)), torch.zeros(4, dtype=torch.float64)) for i in range(7)])[None,None]
    actual_causal = causal.causal_dot_product(cq, ck, cv)
    close(actual_causal, expected_causal, "causal dot", 2e-5)
    cot = torch.randn_like(actual_causal)
    ga = torch.autograd.grad((actual_causal * cot).sum(), (cq, ck, cv))
    ge = torch.autograd.grad((expected_causal * cot.cpu().double()).sum(), (cqr, ckr, cvr))
    for a,e in zip(ga,ge):
        close(a,e,"causal backward",2e-5)

    model = importlib.import_module("src.models.modules.attention.sblocal_attention")
    spec = importlib.util.spec_from_file_location("cpu_dense_oracle", ROOT / "tests/traceability/test_scatterbrain_initial_cpu.py")
    assert spec and spec.loader
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    module = model.SBLocalAttention(3, 4, nb_features=8, softmax_temp=.25, device="cuda")
    q = (torch.randn(1, 5, 1, 4, device="cuda") * .2).requires_grad_()
    k = (torch.randn_like(q) * .2).detach().requires_grad_()
    v = torch.randn(1, 5, 1, 2, device="cuda", requires_grad=True)
    module.train()
    output, weights = module(q,k,v,need_weights=True)
    qr,kr,vr = [x.detach().cpu().double().requires_grad_() for x in (q,k,v)]
    ref = oracle.dense_oracle(qr,kr,vr,module.feature_map.projection_matrix.detach().cpu(),3,.25,torch.tensor([5]))
    close(output,ref[0],"hybrid output")
    close(weights,ref[1],"hybrid weights")
    ga = torch.autograd.grad(output.square().sum(),(q,k,v))
    ge = torch.autograd.grad(ref[0].square().sum(),(qr,kr,vr))
    for a,e in zip(ga,ge):
        close(a,e,"hybrid gradients",2e-5)
    # Tiny connected train/save/reload: a real trainable projection feeds actual
    # attention and CUDA autograd, with persistent feature state restored.
    projection = torch.nn.Linear(4,4).cuda()
    optimizer = torch.optim.SGD(projection.parameters(),lr=.01)
    initial = projection.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        out,_ = module(projection(q.detach()),k.detach(),v.detach())
        out.square().mean().backward()
        optimizer.step()
    assert not torch.equal(initial,projection.weight)
    module.feature_map.deterministic_eval = True
    module.eval()
    saved = io.BytesIO()
    torch.save({"attention":module.state_dict(),"projection":projection.state_dict(),"optimizer":optimizer.state_dict()}, saved)
    saved.seek(0)
    checkpoint = torch.load(saved,weights_only=True)
    restored = model.SBLocalAttention(3,4,nb_features=8,softmax_temp=.25,device="cuda").eval()
    restored.feature_map.deterministic_eval=True
    restored.load_state_dict(checkpoint["attention"])
    projection2 = torch.nn.Linear(4,4).cuda()
    projection2.load_state_dict(checkpoint["projection"])
    optimizer2 = torch.optim.SGD(projection2.parameters(),lr=.01)
    optimizer2.load_state_dict(checkpoint["optimizer"])
    out,_ = module(projection(q.detach()),k.detach(),v.detach())
    restored_out,_ = restored(projection2(q.detach()),k.detach(),v.detach())
    close(restored_out,out,"checkpoint output",1e-7)
    torch.cuda.synchronize()
    results.update({"source_commit":revision,"source_sha256":sources,"binary_sha256":binaries,"local_contexts":[1,2,3,4,9],
            "max_abs_error":max(errors),"causal_primitive":True,"noncausal_hybrid":True,
            "training_updates":2,"save_reload":True,"causal_hybrid":"NOT_RUN", "nondefault_stream":"PASS: maintained local CUDA; original launches remain default-stream-only"})
    return results
