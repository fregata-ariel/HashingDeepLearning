#!/usr/bin/env python3
"""Actual maintained Scatterbrain initial CPU native composition and independent oracles.

The original pinned native wrapper/kernel are loaded from the driver's isolated
build. The maintained port is mandatory; the frozen archive is never a fallback.

Evidence: SCATTERBRAIN-LOCAL-PRODUCT-CPU, SCATTERBRAIN-LOCAL-HYBRID-CPU.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import math
import sys
import textwrap
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, atol, name):
    require(torch.isfinite(a).all() and torch.isfinite(b).all(), name + " nonfinite")
    err = (a.double() - b.double()).abs().max().item()
    require(err < atol, f"{name}: max_error={err} bound={atol}")
    return err


def products():
    metrics = {"cases": 0, "forward_error": 0.0, "gradient_error": 0.0}
    for context in (1, 2, 3, 4, 9):
        torch.manual_seed(175 + context)
        q = torch.randn(2, 2, 5, 3, dtype=torch.float32, requires_grad=True)
        k = torch.randn(2, 2, 4, 3, dtype=torch.float32, requires_grad=True)
        v = torch.randn(2, 2, 4, 2, dtype=torch.float32, requires_grad=True)
        lengths = torch.tensor([4, 3])
        mask = torch.zeros(5, 4)
        mask[2, 1] = -1e24
        actual = native.local_dot_product(q, k, mask, lengths, context)
        i = torch.arange(5)[:, None]
        offset = torch.arange(context)[None, :]
        positions = i - context // 2 + offset
        valid = (positions >= 0) & (positions < lengths[:, None, None])
        gathered = k[:, :, positions.clamp(0, 3), :]
        expected = (q.unsqueeze(-2) * gathered).sum(-1)
        gathered_mask = mask[torch.arange(5)[:, None], positions.clamp(0, 3)]
        expected = expected + gathered_mask
        expected = torch.where(
            valid[:, None, :, :], expected, expected.new_full((), -1e24)
        )
        metrics["forward_error"] = max(
            metrics["forward_error"],
            close(actual, expected, 2e-6, "native local_dot_product"),
        )
        # The pinned backward receives no additive mask, so mask only controls
        # the forward additive constant, not q/k differentiation at valid pairs.
        cotangent = torch.randn_like(actual)
        ga = torch.autograd.grad((actual * cotangent).sum(), (q, k), retain_graph=True)
        ge = torch.autograd.grad((expected * cotangent).sum(), (q, k))
        for x, y in zip(ga, ge):
            metrics["gradient_error"] = max(
                metrics["gradient_error"],
                close(x, y, 2e-6, "native local_dot backward"),
            )
        weights = torch.randn_like(actual, requires_grad=True)
        average = native.local_weighted_average(weights, v)
        # weighted_average uses physical key length, not key_padding_lengths;
        # invalid physical neighborhood entries contribute exactly zero.
        physically_valid = (positions >= 0) & (positions < 4)
        expected_average = (
            weights.unsqueeze(-1)
            * v[:, :, positions.clamp(0, 3), :]
            * physically_valid[None, None, :, :, None]
        ).sum(-2)
        metrics["forward_error"] = max(
            metrics["forward_error"],
            close(average, expected_average, 2e-6, "native local weighted average"),
        )
        cotangent_v = torch.randn_like(average)
        ga = torch.autograd.grad(
            (average * cotangent_v).sum(), (weights, v), retain_graph=True
        )
        ge = torch.autograd.grad((expected_average * cotangent_v).sum(), (weights, v))
        for x, y in zip(ga, ge):
            metrics["gradient_error"] = max(
                metrics["gradient_error"],
                close(x, y, 2e-6, "native weighted average backward"),
            )
        metrics["cases"] += 1
    return metrics


def dense_oracle(q, k, v, projection, context, temp, lengths):
    # Independent unstabilized FP64 cosh features with a fixed sampled matrix.
    # No archived feature helper, local kernel or log-normalizer is called.
    q = q.double().transpose(1, 2)
    k = k.double().transpose(1, 2)
    v = v.double().transpose(1, 2)
    projection = projection.double()
    width = projection.shape[0] * 2
    q_scaled = q * math.sqrt(temp)
    k_scaled = k * math.sqrt(temp)
    qp = q_scaled @ projection.T
    kp = k_scaled @ projection.T
    qfeatures = torch.exp(
        torch.cat((qp, -qp), -1) - q_scaled.square().sum(-1, keepdim=True) / 2
    ) / math.sqrt(width)
    kfeatures = torch.exp(
        torch.cat((kp, -kp), -1) - k_scaled.square().sum(-1, keepdim=True) / 2
    ) / math.sqrt(width)
    lowrank = qfeatures @ kfeatures.transpose(-1, -2)
    exact = torch.exp((q @ k.transpose(-1, -2)) * temp)
    i = torch.arange(q.shape[-2])[:, None]
    j = torch.arange(k.shape[-2])[None, :]
    support = (j - i >= -(context // 2)) & (j - i < (context + 1) // 2)
    valid = j[None, None, :, :] < lengths[:, None, None, None]
    approximate = torch.where(support, exact, lowrank) * valid
    normalization = approximate.sum(-1, keepdim=True)
    weights = approximate / normalization
    output = (weights @ v).transpose(1, 2)
    return (
        output,
        weights,
        approximate,
        lowrank * valid,
        support,
        approximate @ v,
        normalization,
        exact * valid,
    )


def mutated_forward(kind):
    """Execute a precise one-statement numerical mutation without source edits."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(SBLocalAttention.forward)))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if (
            kind == "omit_subtraction"
            and "dots" in names
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Sub)
        ):
            node.value = node.value.left
            changed += 1
        if kind == "mismatched_normalizer" and "log_normalization" in names:
            node.value = ast.parse(
                "torch.log(rearrange(q_prime_k_prime_1, 'b h s -> b h s 1')) + prime_log_scale",
                mode="eval",
            ).body
            changed += 1
    require(
        changed == 1, "mutation must change exactly one supported assignment: " + kind
    )
    namespace = dict(model_module.__dict__)
    exec(
        compile(
            ast.fix_missing_locations(tree),
            "scatterbrain-CPU-sensitivity-" + kind,
            "exec",
        ),
        namespace,
    )
    return namespace["forward"]


def composition():
    metrics = {
        "cases": 0,
        "full_selection_cases": 0,
        "output_error": 0.0,
        "matrix_error": 0.0,
        "gradient_error": 0.0,
        "mutations": {"omit_subtraction": 0, "mismatched_normalizer": 0},
    }
    for context in (1, 2, 3, 4, 16):
        for temp in (1 / math.sqrt(3), 0.25):
            torch.manual_seed(2357 + context)
            module = SBLocalAttention(
                context,
                3,
                nb_features=6,
                softmax_temp=temp,
                attention_dropout=0.0,
                softmax_eps=0.0,
            )
            module.feature_map.new_feature_map("cpu")
            projection = module.feature_map.projection_matrix.detach().clone()
            q = (torch.randn(2, 5, 2, 3) * 0.25).requires_grad_()
            k = (torch.randn(2, 4, 2, 3) * 0.25).requires_grad_()
            v = (torch.randn(2, 4, 2, 2) * 0.25).requires_grad_()
            lengths = torch.tensor([4, 3])
            mask = LengthMask(lengths, max_len=4)
            qref = q.detach().double().requires_grad_()
            kref = k.detach().double().requires_grad_()
            vref = v.detach().double().requires_grad_()
            ref = dense_oracle(qref, kref, vref, projection, context, temp, lengths)
            actual, (weights, unnormalized, lowrank) = module(
                q,
                k,
                v,
                key_padding_mask=mask,
                need_weights=True,
                return_attn_unnormalized=True,
            )
            # The maintained return_attn_unnormalized path exposes raw mass.
            # Check its row sum and V product independently, beyond outputs.
            actual_normalizer = unnormalized.sum(-1, keepdim=True)
            actual_numerator = torch.einsum("bhts,bshd->bhtd", unnormalized, v)
            close(actual_normalizer, ref[6], 4e-6, "actual raw normalizer")
            close(actual_numerator, ref[5], 4e-6, "actual raw numerator")
            # Cross-check the actual normalized native execution against those
            # independently reconstructed raw intermediates.
            close(
                actual.transpose(1, 2) * actual_normalizer,
                actual_numerator,
                4e-6,
                "native output times normalizer",
            )
            metrics["output_error"] = max(
                metrics["output_error"],
                close(actual, ref[0], 3e-6, "actual SBLocal output"),
            )
            metrics["matrix_error"] = max(
                metrics["matrix_error"],
                close(weights, ref[1], 3e-6, "actual SBLocal weights"),
                close(unnormalized, ref[2], 3e-6, "actual SBLocal unnormalized"),
                close(lowrank, ref[3], 3e-6, "actual SBLocal global lowrank"),
            )
            cotangent = torch.randn_like(actual)
            actual_grads = torch.autograd.grad(
                (actual * cotangent).sum(), (q, k, v), retain_graph=True
            )
            oracle_grads = torch.autograd.grad(
                (ref[0] * cotangent.double()).sum(), (qref, kref, vref)
            )
            for x, y in zip(actual_grads, oracle_grads):
                metrics["gradient_error"] = max(
                    metrics["gradient_error"],
                    close(x, y, 4e-6, "actual SBLocal QKV gradient"),
                )
            for mutation in ("omit_subtraction", "mismatched_normalizer"):
                mutant = mutated_forward(mutation)
                bad, _ = mutant(
                    module, q, k, v, key_padding_mask=mask, need_weights=False
                )
                require(
                    torch.isfinite(bad).all(), "mutation result nonfinite: " + mutation
                )
                if (bad.double() - ref[0]).abs().max().item() > 1e-3:
                    metrics["mutations"][mutation] += 1
            if context == 16:
                dense_logits = (
                    torch.einsum("bthe,bshe->bhts", q.double(), k.double()) * temp
                )
                dense_logits = dense_logits.masked_fill(
                    torch.arange(4)[None, None, None, :]
                    >= lengths[:, None, None, None],
                    -math.inf,
                )
                dense = torch.einsum(
                    "bhts,bshd->bthd", dense_logits.softmax(-1), v.double()
                )
                close(actual, dense, 3e-6, "full local coverage equals dense")
                metrics["full_selection_cases"] += 1
            metrics["cases"] += 1
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--port-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.unlink(missing_ok=True)
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.native_root.resolve()))
    sys.path.insert(0, str(args.port_root.resolve()))
    global native, SBLocalAttention, LengthMask, model_module
    native = importlib.import_module("fast_transformers.local_product")
    model_module = importlib.import_module(
        "src.models.modules.attention.sblocal_attention"
    )
    SBLocalAttention = model_module.SBLocalAttention
    LengthMask = importlib.import_module("src.models.modules.masking").LengthMask
    wrapper = Path(native.__file__).resolve()
    require(
        wrapper.is_relative_to(args.native_root.resolve()),
        "exact locally staged native wrapper required",
    )
    extension = importlib.import_module(
        "fast_transformers.local_product.local_product_cpu"
    )
    binary = Path(extension.__file__).resolve()
    require(
        binary.suffix == ".so" and binary.is_relative_to(args.native_root.resolve()),
        "actual locally built extension required",
    )
    modules = [
        "src.models.modules.attention.sblocal_attention",
        "src.models.modules.attention.feature_maps_sb",
        "src.models.modules.attention.projection_utils",
        "src.models.modules.attention.scatterbrain_utils",
        "src.models.modules.masking",
    ]
    sources = {}
    for name in modules:
        path = Path(importlib.import_module(name).__file__).resolve()
        require(
            path.is_relative_to(args.port_root.resolve()),
            "maintained port import required: " + name,
        )
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    primitive = products()
    composed = composition()
    require(
        composed["cases"] == 10 and composed["full_selection_cases"] == 2,
        "complete fixture matrix",
    )
    require(
        all(value == 10 for value in composed["mutations"].values()),
        "mutation sensitivity incomplete",
    )
    result = {
        "status": "passed",
        "native_CPU_executed": True,
        "maintained_port_executed": True,
        "substitute_local_product": False,
        "source_sha256": sources,
        "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        "extension_path": str(binary),
        "extension_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_products": primitive,
        "composition": composed,
        "gradient_reference": "detached FP64 leaves; independent unstabilized cosh features and dense support replacement",
        "scope": "float32 CPU noncausal/dropout0/eps0; positive masked global rows; no GPU/causal/arbitrary masks/extremes/redraw-state/memory-sanitizer claim",
        "gpu_validated": False,
        "memory_sanitizers_executed": False,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(result, indent=2) + "\n")
    print("SCATTERBRAIN_INITIAL_NATIVE_CPU_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
