#!/usr/bin/env python3
"""Actual maintained feature/projection CPU checks; no sampling-unbiasedness claim.

The dense oracle uses scalar FP64 exp/dot products and analytic derivatives,
independent of the maintained normalization/amax/einsum helpers. Known initial
lifecycle defects are characterized separately and are not repaired here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = "src/models/modules/attention/"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Checks:
    def __init__(self) -> None:
        self.assertions = 0
        self.cases: dict[str, int] = {}
        self.max_abs_error: dict[str, float] = {}

    def require(self, condition: bool, message: str) -> None:
        self.assertions += 1
        if not condition:
            raise AssertionError(message)

    def case(self, category: str) -> None:
        self.cases[category] = self.cases.get(category, 0) + 1

    def close(self, actual: torch.Tensor, reference: torch.Tensor, label: str,
              *, atol: float = 3e-6, rtol: float = 3e-6) -> None:
        actual64 = actual.detach().to(torch.float64)
        ref64 = reference.detach().to(torch.float64)
        self.require(actual64.shape == ref64.shape, label + " shape")
        self.require(bool(torch.isfinite(actual64).all()), label + " actual nonfinite")
        self.require(bool(torch.isfinite(ref64).all()), label + " reference nonfinite")
        error = (actual64 - ref64).abs()
        self.require(bool(torch.isfinite(error).all()), label + " error nonfinite")
        maximum = float(error.max()) if error.numel() else 0.0
        self.max_abs_error[label] = max(self.max_abs_error.get(label, 0.0), maximum)
        self.require(bool((error <= atol + rtol * ref64.abs()).all()),
                     f"{label}: max abs error {maximum}")

    def raises(self, kind: type[Exception], text: str, action: Callable[[], object]) -> None:
        try:
            action()
        except kind as error:
            self.require(text in str(error), "unexpected exception message: " + str(error))
        else:
            self.require(False, "expected " + kind.__name__)


def inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([[[.2, -.4, .1], [-.3, .7, .4], [.5, .2, -.6]],
                      [[-.1, .3, .6], [.4, -.2, .3], [-.7, -.1, .2]]], dtype=dtype)
    projection = torch.tensor([[.3, -.7, .2], [-.8, .1, .5],
                               [.6, .4, -.3], [.2, -.5, -.9]], dtype=torch.float64)
    return x, projection


def dense_reference(x: torch.Tensor, projection: torch.Tensor, temperature: float,
                    cosh: bool, query: bool, eps: float = 0.0,
                    log: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalar raw Phi = exp(+-sqrt(t) p.x - t||x||²/2)/sqrt(M)."""
    # Match the documented input cast; all reference arithmetic after it is FP64.
    rows = projection.to(x.dtype).to(torch.float64).tolist()
    data = x.detach().to(torch.float64).tolist()
    signed_rows = rows + [[-v for v in row] for row in rows] if cosh else rows
    m = len(signed_rows)
    log_raw = [[[math.sqrt(temperature) * sum(a*b for a, b in zip(vector, row))
                 - .5 * temperature * sum(a*a for a in vector) - .5 * math.log(m)
                 for row in signed_rows] for vector in batch] for batch in data]
    if query:
        scales = [[[max(vector)] for vector in batch] for batch in log_raw]
    else:
        scales = [[[max(value for vector in batch for value in vector)]] for batch in log_raw]
    raw = torch.tensor(log_raw, dtype=torch.float64).exp()
    scale = torch.tensor(scales, dtype=torch.float64)
    normalized_log = torch.tensor(log_raw, dtype=torch.float64) - scale
    features = normalized_log if log else normalized_log.exp() + eps / math.sqrt(m)
    return features, scale, raw


def feature_checks(check: Checks, features: ModuleType) -> None:
    for dtype in (torch.float32, torch.float64):
        x, p = inputs(dtype)
        atol, rtol = (3e-6, 3e-6) if dtype == torch.float32 else (8e-15, 5e-14)
        for cosh in (False, True):
            for requested_temp in (None, .35, 1.3):
                temperature = 1 / math.sqrt(x.shape[-1]) if requested_temp is None else requested_temp
                for query in (False, True):
                    for eps, log in ((0.0, False), (0.0, True), (1e-4, False)):
                        check.case("dense_feature")
                        actual, scale = features.softmax_kernel(x, projection_matrix=p,
                            is_query=query, softmax_temp=requested_temp, eps=eps, cosh=cosh, return_log=log)
                        expected, expected_scale, raw = dense_reference(x, p, temperature, cosh, query, eps, log)
                        check.require(actual.dtype == dtype and scale.dtype == dtype, "feature dtype")
                        check.close(actual, expected, "features", atol=atol, rtol=rtol)
                        check.close(scale, expected_scale, "log_scale", atol=atol, rtol=rtol)
                        restored = (actual.exp() if log else actual) * scale.exp()
                        expected_restored = raw + (0 if log else eps / math.sqrt(raw.shape[-1]) * expected_scale.exp())
                        check.close(restored, expected_restored, "restored_features", atol=atol, rtol=rtol)
                    check.raises(NotImplementedError, "return_log is not compatible with nonzero eps",
                        lambda: features.softmax_kernel(x, projection_matrix=p, is_query=query,
                            cosh=cosh, eps=1e-4, return_log=True))
                # This is the finite sampled-feature kernel, not an exact exp(q.k) claim.
                q, qs = features.softmax_kernel(x, projection_matrix=p, is_query=True,
                                               softmax_temp=requested_temp, cosh=cosh)
                k, ks = features.softmax_kernel(x.flip(-2)*.7, projection_matrix=p, is_query=False,
                                               softmax_temp=requested_temp, cosh=cosh)
                _, _, rawq = dense_reference(x, p, temperature, cosh, True)
                _, _, rawk = dense_reference(x.flip(-2)*.7, p, temperature, cosh, False)
                actual_kernel = (q*qs.exp()) @ (k*ks.exp()).transpose(-1, -2)
                dense_kernel = torch.tensor([[[sum(a*b for a, b in zip(qv, kv)) for kv in kb]
                    for qv in qb] for qb, kb in zip(rawq.tolist(), rawk.tolist())], dtype=torch.float64)
                check.case("restored_kernel")
                check.close(actual_kernel, dense_kernel, "sampled_kernel", atol=atol, rtol=rtol)
                for query in (False, True):
                    # Analytic derivative of raw Phi, independent of amax or autograd reference.
                    variable = x.detach().clone().requires_grad_(True)
                    variable_projection = p.detach().clone().requires_grad_(True)
                    value, scale = features.softmax_kernel(variable, projection_matrix=variable_projection,
                        is_query=query, softmax_temp=requested_temp, cosh=cosh)
                    raw_actual = value * scale.exp()
                    weights = torch.linspace(.2, 1.1, raw_actual.numel(), dtype=dtype).reshape(raw_actual.shape)
                    (raw_actual*weights).sum().backward()
                    _, _, raw = dense_reference(x, p, temperature, cosh, query)
                    rows = p.to(dtype).to(torch.float64).tolist()
                    if cosh:
                        rows += [[-v for v in row] for row in rows]
                    gradient = [[[sum(w*phi*(math.sqrt(temperature)*row[d]-temperature*vector[d])
                        for w, phi, row in zip(wv, phiv, rows)) for d in range(3)]
                        for vector, wv, phiv in zip(batch, wb, pb)]
                        for batch, wb, pb in zip(x.to(torch.float64).tolist(), weights.to(torch.float64).tolist(), raw.tolist())]
                    check.case("analytic_gradient")
                    check.require(variable.grad is not None, "missing input gradient")
                    check.close(variable.grad, torch.tensor(gradient, dtype=torch.float64),
                                "gradient", atol=atol, rtol=rtol)
                    # dPhi/dp_j = +-sqrt(t) * x * Phi; combine the two cosh halves.
                    projection_gradient = []
                    for row_index in range(p.shape[0]):
                        row_gradient = []
                        for dimension in range(x.shape[-1]):
                            total = 0.0
                            for batch, wb, pb in zip(x.to(torch.float64).tolist(), weights.to(torch.float64).tolist(), raw.tolist()):
                                for vector, wv, phiv in zip(batch, wb, pb):
                                    coefficient = wv[row_index] * phiv[row_index]
                                    if cosh:
                                        coefficient -= wv[row_index+p.shape[0]] * phiv[row_index+p.shape[0]]
                                    total += coefficient * math.sqrt(temperature) * vector[dimension]
                            row_gradient.append(total)
                        projection_gradient.append(row_gradient)
                    check.case("analytic_projection_gradient")
                    check.require(variable_projection.grad is not None, "missing projection gradient")
                    check.close(variable_projection.grad, torch.tensor(projection_gradient, dtype=torch.float64),
                                "projection_gradient", atol=atol, rtol=rtol)


def projection_checks(check: Checks, projection: ModuleType) -> None:
    previous_default = torch.get_default_dtype()
    try:
        for dtype in (torch.float32, torch.float64):
            torch.set_default_dtype(dtype)
            tolerance = 2e-6 if dtype == torch.float32 else 2e-14
            for rows, cols in ((1, 3), (3, 3), (7, 3), (4, 1)):
                for scaling in (0, 1):
                    torch.manual_seed(812)
                    before = torch.random.get_rng_state().clone()
                    # Probe only the QR defining property; do not compute a second QR.
                    original = torch.randn((math.ceil(rows/cols), cols, cols))
                    torch.random.set_rng_state(before)
                    actual = projection.gaussian_orthogonal_random_matrix(rows, cols, scaling=scaling,
                                                                         device="cpu", dtype=dtype)
                    after = torch.random.get_rng_state().clone()
                    check.case("projection")
                    check.require(actual.shape == (rows, cols) and actual.dtype == dtype, "projection shape/dtype")
                    check.require(actual.device.type == "cpu", "projection device")
                    check.require(not torch.equal(before, after), "projection must consume RNG")
                    lengths = actual.norm(dim=1)
                    check.require(bool(torch.isfinite(actual).all()) and bool((lengths > 0).all()), "projection finite/nonzero")
                    if scaling == 1:
                        check.close(lengths, torch.full_like(lengths, math.sqrt(cols)), "projection_scale", atol=tolerance, rtol=tolerance)
                    unit = actual / lengths[:, None]
                    for block in range(math.ceil(rows/cols)):
                        part = unit[block*cols:(block+1)*cols]
                        check.close(part @ part.T, torch.eye(part.shape[0], dtype=dtype), "block_orthogonality", atol=tolerance, rtol=tolerance)
                        # Q^T A is upper triangular with nonnegative diagonal after sign fixing.
                        triangular = part @ original[block]
                        check.close(torch.tril(triangular, diagonal=-1), torch.zeros_like(triangular), "qr_triangular_property", atol=tolerance, rtol=tolerance)
                        check.require(bool((triangular.diagonal() > 0).all()), "QR sign correction")
                    torch.random.set_rng_state(before)
                    repeated = projection.gaussian_orthogonal_random_matrix(rows, cols, scaling=scaling, device="cpu", dtype=dtype)
                    check.require(torch.equal(actual, repeated), "seed reproducibility")
                    check.require(torch.equal(after, torch.random.get_rng_state()), "seed state reproducibility")
                    if scaling == 0:
                        # Replay only consumed normal draws, not the orthogonalization implementation.
                        torch.random.set_rng_state(before)
                        torch.randn(original.shape)
                        independent_lengths = torch.randn((rows, cols), dtype=dtype).square().sum(1).sqrt()
                        check.close(lengths, independent_lengths, "gaussian_row_lengths", atol=tolerance, rtol=tolerance)
            check.raises(ValueError, "Invalid scaling 2", lambda: projection.gaussian_orthogonal_random_matrix(3, 3, scaling=2, dtype=dtype))
    finally:
        torch.set_default_dtype(previous_default)


def lifecycle_checks(check: Checks, features: ModuleType) -> list[dict[str, object]]:
    known: list[dict[str, object]] = []
    torch.manual_seed(98)
    cold = features.SBPerformerFeatures(3, n_features=8, deterministic_eval=True, dtype=torch.float64)
    cold.eval()
    state = torch.random.get_rng_state().clone()
    cold.new_feature_map("cpu")
    check.require(cold._calls == -1 and cold.projection_matrix.shape == (3, 8), "cold eval leaves wrong initial shape")
    check.require(torch.equal(state, torch.random.get_rng_state()), "cold eval does not draw")
    check.raises(RuntimeError, "subscript d has size", lambda: cold.forward_queries(torch.zeros(1, 2, 3, dtype=torch.float64)))
    known.append({"issue": 106, "behavior": "cold deterministic_eval preserves transposed placeholder and forward raises RuntimeError"})
    warm = features.SBPerformerFeatures(3, n_features=8, redraw=1, deterministic_eval=True, dtype=torch.float64)
    warm.new_feature_map("cpu")
    first = warm.projection_matrix.clone()
    state = torch.random.get_rng_state().clone()
    for _ in range(8):
        warm.new_feature_map("cpu")
    check.require(warm.redraw == math.pi and warm._calls == 8, "requested redraw replaced by pi")
    check.require(torch.equal(first, warm.projection_matrix) and torch.equal(state, torch.random.get_rng_state()), "no periodic redraw in first eight calls")
    known.append({"issue": 106, "behavior": "requested redraw=1 ignored; irrational pi cadence retains first projection through calls 1..8"})
    warm.eval()
    warm.new_feature_map("cpu")
    check.require(warm._calls == 8 and torch.equal(first, warm.projection_matrix), "warm deterministic eval stable")
    x, _ = inputs(torch.float64)
    check.require(warm.forward_queries(x)[0].shape == (2, 3, 8), "warm query output")
    saved = {key: value.detach().clone() for key, value in warm.state_dict().items()}
    check.require(set(saved) == {"projection_matrix"}, "only projection is persisted")
    fresh = features.SBPerformerFeatures(3, n_features=8, dtype=torch.float64)
    check.raises(RuntimeError, "size mismatch for projection_matrix", lambda: fresh.load_state_dict(saved))
    loaded = features.SBPerformerFeatures(3, n_features=8, dtype=torch.float64)
    loaded.new_feature_map("cpu")
    loaded.load_state_dict(saved)
    check.close(loaded.forward_queries(x)[0], warm.forward_queries(x)[0], "warm_save_load", atol=0, rtol=0)
    check.require(loaded._calls == 0 and warm._calls == 8, "call counter is not persisted")
    known.append({"issue": 106, "behavior": "default warm state load into a fresh module fails; compatible initialized load succeeds; call counter is absent from state_dict"})
    odd = features.SBPerformerFeatures(3, n_features=7, cosh=True, dtype=torch.float64)
    odd.new_feature_map("cpu")
    check.require(odd.n_features == 7 and odd.forward_keys(x)[0].shape[-1] == 6, "odd cosh count truncates")
    known.append({"issue": 106, "behavior": "cosh requested 7 features produces 6"})
    check.cases["lifecycle_characterization"] = 4
    return known


def mutant_checks(check: Checks, features: ModuleType) -> None:
    source = inspect.getsource(features.softmax_kernel)
    mutations = {
        "missing_half_norm": (" / 2 * (data_normalizer ** 2)", " / 4 * (data_normalizer ** 2)"),
        "wrong_cosh_sign": ("[data_dash, -data_dash]", "[data_dash, data_dash]"),
        "wrong_feature_scale": ("math.log(m) / 2", "math.log(m)"),
    }
    x, p = inputs(torch.float64)
    _, _, expected = dense_reference(x, p, .35, True, True)
    for name, (before, after) in mutations.items():
        check.require(before in source, "mutant target absent: " + name)
        namespace = dict(vars(features))
        exec(compile(source.replace(before, after), "<"+name+">", "exec"), namespace)
        actual, scale = namespace["softmax_kernel"](x, projection_matrix=p, is_query=True, softmax_temp=.35, cosh=True)
        restored = actual * scale.exp()
        check.require(bool(torch.isfinite(restored).all()) and bool(torch.isfinite(expected).all()), "mutant nonfinite")
        difference = (restored-expected).abs()
        check.require(bool(torch.isfinite(difference).all()), "mutant error nonfinite")
        check.require(float(difference.max()) > 1e-3, "oracle failed to discriminate " + name)
        check.case("rejected_arithmetic_mutant")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-root", type=Path, default=ROOT / "ports/scatterbrain")
    parser.add_argument("--emit-dir", type=Path)
    args = parser.parse_args()
    evidence = args.emit_dir / "scatterbrain_feature_cpu_evidence.json" if args.emit_dir else None
    if evidence:
        evidence.unlink(missing_ok=True)
    global torch
    import torch

    port_root = args.port_root.resolve(strict=True)
    source_files = {name: port_root / (CORE+name) for name in ("feature_maps_sb.py", "projection_utils.py")}
    for path in source_files.values():
        if not path.is_file() or path.is_symlink():
            raise ValueError("missing maintained feature/projection source")
    sys.path.insert(0, str(port_root))
    features = importlib.import_module("src.models.modules.attention.feature_maps_sb")
    projection = importlib.import_module("src.models.modules.attention.projection_utils")
    if Path(features.__file__).resolve() != source_files["feature_maps_sb.py"] or Path(projection.__file__).resolve() != source_files["projection_utils.py"]:
        raise ValueError("import resolved outside requested maintained port")
    torch.set_num_threads(1)
    check = Checks()
    feature_checks(check, features)
    projection_checks(check, projection)
    known = lifecycle_checks(check, features)
    mutant_checks(check, features)
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "einops": importlib.import_module("einops").__version__, "device": "cpu", "threads": torch.get_num_threads(), "machine": platform.machine()}
    result = {"status": "cpu_pass_with_characterized_lifecycle_defects", "ledger_id": "SCATTERBRAIN-FEATURES-CPU",
              "relation": "variant", "upstream_commit": "eeb689c00b9f322f33f6b6f85eaaac0a1d6a5eab",
              "paper_locator": "main p6 section 4.2 feature map", "cases": check.cases,
              "passed_assertions": check.assertions, "known_failures": known,
              "source_sha256": {name: sha256(path) for name, path in source_files.items()},
              "test_sha256": sha256(Path(__file__)), "runtime": runtime,
              "runtime_sha256": hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest(),
              "max_abs_error": check.max_abs_error,
              "error_budgets": {"float32": {"atol": 3e-6, "rtol": 3e-6},
                                "float64_features": {"atol": 8e-15, "rtol": 5e-14},
                                "float64_projection": {"atol": 2e-14, "rtol": 2e-14}},
              "native_dependency_executed": False,
              "statistical_unbiasedness_validated": False, "variance_reduction_validated": False,
              "gpu_validated": False}
    if evidence:
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
