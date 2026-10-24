#!/usr/bin/env python3
"""Actual four SYCL kernels and maintained hybrid against independent FP64 oracles.

TRACE_TEST_ID: SCATTERBRAIN-LOCAL-PRODUCT-SYCL-CPU
TRACE_TEST_ID: SCATTERBRAIN-LOCAL-HYBRID-SYCL-CPU
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import FunctionType
from typing import Callable, cast
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--gpu-backend", action="store_true", help="library compiled to require NVIDIA; ABI and Torch hybrid still use host tensors")
    args = parser.parse_args()
    args.evidence.unlink(missing_ok=True)
    root = Path(__file__).resolve().parents[4]
    sycl_root = root / "ports/scatterbrain/sycl"
    sys.path[:0] = [str(sycl_root), str(root / "ports/scatterbrain")]
    torch.set_num_threads(2)
    boundary = importlib.import_module("sblocal_sycl")
    native = boundary.local_backend
    assert native.__file__ is not None
    assert Path(native.__file__).resolve().is_relative_to(sycl_root)
    native.load_library(args.library)
    identity = ctypes.CDLL(str(args.library.resolve()))
    identity.sb_is_gpu.restype = ctypes.c_int
    assert identity.sb_is_gpu() == int(args.gpu_backend), "library backend does not match requested profile"
    model = importlib.import_module("src.models.modules.attention.sblocal_attention")
    bounded = boundary.SBLocalSYCLAttention
    assert model.local_dot_product is native.local_dot_product
    assert model.local_weighted_average is native.local_weighted_average
    spec = importlib.util.spec_from_file_location("independent_cpu_oracles", root / "tests/traceability/test_scatterbrain_initial_cpu.py")
    assert spec is not None and spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    setattr(oracle, "native", native)
    setattr(oracle, "SBLocalAttention", bounded)
    setattr(oracle, "LengthMask", importlib.import_module("src.models.modules.masking").LengthMask)
    setattr(oracle, "model_module", model)
    # Existing substantive mutations inspect the actual maintained hybrid body;
    # the bounded class adds explicit mode validation around that same body.
    mutate = oracle.mutated_forward
    def actual_body_mutation(kind: str) -> FunctionType:
        setattr(oracle, "SBLocalAttention", model.SBLocalAttention)
        try:
            return cast(FunctionType, mutate(kind))
        finally:
            setattr(oracle, "SBLocalAttention", bounded)
    setattr(oracle, "mutated_forward", actual_body_mutation)
    products = oracle.products()
    composed = oracle.composition()
    lifecycle = oracle.connected_lifecycle()
    assert products["cases"] == 5 and composed["cases"] == 10
    assert composed["full_selection_cases"] == 2
    assert all(count == 10 for count in composed["mutations"].values())
    # Additional launch-size/unequal-length/repetition/zero-prefix checks use a
    # scalar dense FP64 gather independent of the production index helpers.
    extra = 0
    for nq, nk, window in ((1, 1, 2), (7, 3, 4), (3, 7, 9), (17, 16, 3)):
        for repeat in range(2):
            torch.manual_seed(nq * 317 + repeat)
            q = torch.randn(2, 2, nq, 3, requires_grad=True)
            k = torch.randn(2, 2, nk, 3, requires_grad=True)
            lengths = torch.tensor([nk, 0])
            mask = torch.randn(nq, nk) * .1
            actual = native.local_dot_product(q, k, mask, lengths, window)
            qref = q.detach().double().requires_grad_()
            kref = k.detach().double().requires_grad_()
            reference = torch.stack([torch.stack([torch.stack([torch.stack([
                (qref[b,h,i] * kref[b,h,i-window//2+slot]).sum() + mask[i,i-window//2+slot].double()
                if 0 <= i-window//2+slot < int(lengths[b]) else q.new_tensor(-1e24).double()
                for slot in range(window)]) for i in range(nq)]) for h in range(2)]) for b in range(2)])
            oracle.close(actual, reference, 3e-6, "extra scalar dot")
            cot = torch.randn_like(actual)
            ga = torch.autograd.grad((actual * cot).sum(), (q,k), retain_graph=True)
            ge = torch.autograd.grad((reference * cot.double()).sum(), (qref,kref))
            for a,b in zip(ga,ge):
                oracle.close(a,b,5e-6,"extra scalar dot gradients")
            attention = torch.randn(2, 2, nq, window, requires_grad=True)
            values = torch.randn(2, 2, nk, 2, requires_grad=True)
            aref = attention.detach().double().requires_grad_()
            vref = values.detach().double().requires_grad_()
            average = native.local_weighted_average(attention, values)
            # Independent scalar/vector FP64 dense gather, physical K support.
            average_ref = torch.stack([torch.stack([torch.stack([
                sum((aref[b,h,i,slot] * vref[b,h,i-window//2+slot]
                     for slot in range(window) if 0 <= i-window//2+slot < nk),
                    torch.zeros(2, dtype=torch.float64))
                for i in range(nq)]) for h in range(2)]) for b in range(2)])
            oracle.close(average, average_ref, 5e-6, "extra FP64 weighted average")
            cot_v = torch.randn_like(average)
            ga = torch.autograd.grad((average * cot_v).sum(), (attention, values))
            ge = torch.autograd.grad((average_ref * cot_v.double()).sum(), (aref, vref))
            for a,b in zip(ga,ge):
                oracle.close(a,b,5e-6,"extra FP64 weight/V gradients")
            extra += 1
    rejected = 0
    for kwargs in ({"causal": True}, {"attention_dropout": .1}, {"softmax_eps": 1e-6}, {"dtype": torch.float64}, {"device": "cuda"}):
        try:
            bounded(3,3,**kwargs)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("unsupported mode accepted")
    q = torch.ones(1,1,1,3)
    actions: tuple[Callable[[], object], ...] = (lambda: bounded(3,3)(q,q,q,attn_mask=object()),
                   lambda: native.local_dot_product(q,q,torch.zeros(1,1),torch.tensor([2]),3),
                   lambda: bounded(3,3)(q,q,q,key_padding_mask=oracle.LengthMask(torch.tensor([0]),max_len=1),need_weights=True),
                   lambda: native.local_dot_product(q.double(),q,torch.zeros(1,1),torch.tensor([1]),3))
    for action in actions:
        try:
            action()
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("invalid contract accepted")
    sources = list(sycl_root.rglob("*.hpp")) + list(sycl_root.rglob("*.cpp")) + list(sycl_root.rglob("*.py"))
    sources += [root / "tests/traceability/test_scatterbrain_initial_cpu.py", root / "ports/scatterbrain/src/models/modules/attention/sblocal_attention.py", args.library.resolve()]
    result = dict(status="passed", checkout=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                  ids=["SCATTERBRAIN-LOCAL-PRODUCT-SYCL-NVIDIA","SCATTERBRAIN-HOST-HYBRID-SYCL-NVIDIA"] if args.gpu_backend else ["SCATTERBRAIN-LOCAL-PRODUCT-SYCL-CPU","SCATTERBRAIN-LOCAL-HYBRID-SYCL-CPU"],
                  native_products=products, composition=composed, feature_lifecycle=lifecycle, extra_cases=extra,
                  rejected_contracts=rejected, actual_sycl_kernels=True, handwritten_cpu_fallback=False,
                  hybrid_composition="maintained Torch hybrid with actual SYCL local4 forward/backward boundary",
                  original_cuda_executed=False, gpu_executed=args.gpu_backend,
                  torch_hybrid_device="cpu", sycl_backend="NVIDIA GPU" if args.gpu_backend else "CPU",
                  sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    args.evidence.parent.mkdir(parents=True,exist_ok=True)
    args.evidence.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
