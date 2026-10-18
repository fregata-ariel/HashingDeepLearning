#!/usr/bin/env python3
"""Run maintained complete CPU server orchestration with real native callees."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from check_magicpig_extension_cpu import ROOT, build_extensions, emit_native_notice, extension_build_flags
from magicpig_simde import source_record, verify_emulated_binary


def provenance() -> dict[str, object]:
    """Verify pinned source hashes and preserve original complete-method locators."""
    descriptor = json.loads((ROOT / "tests/traceability/magicpig_server_suite.json").read_text())
    result: dict[str, object] = {}
    for relative, expected in descriptor["original_sha256"].items():
        source = ROOT / relative
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("original source hash differs: " + relative)
        tree = ast.parse(source.read_text())
        locators: list[dict[str, object]] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in ("LSHSparseAttnServer", "AttnServer"):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        locators.append({"symbol": node.name + "." + method.name, "line": method.lineno, "end_line": method.end_lineno, "ast_sha256": hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest()})
            elif isinstance(node, ast.FunctionDef) and node.name in ("_make_causal_mask", "repeat_kv", "rotate_half", "apply_rotary_pos_emb", "layer_norm"):
                locators.append({"symbol": node.name, "line": node.lineno, "end_line": node.end_lineno, "ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
        result[relative] = {"sha256": actual, "original_locators": locators}
    return result


def main() -> None:
    """Emit actual execution separately from unsupported native capabilities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cxx", default="g++-14")
    parser.add_argument("--require-native", action="store_true", help="fail after recording evidence if the native CPU orchestrator skips")
    parser.add_argument("--execution-mode", choices=["native", "simde"], default="native")
    parser.add_argument("--require-execution", action="store_true", help="fail if the selected orchestrator execution skips")
    args = parser.parse_args()
    if args.require_native and args.execution_mode != "native":
        parser.error("--require-native cannot be combined with --execution-mode simde")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    originals = provenance()
    flags = Path("/proc/cpuinfo").read_text()
    required = ["avx512f", "avx512bw", "avx512dq", "avx512vl", "fma"]
    missing = [flag for flag in required if flag not in flags] if args.execution_mode == "native" else []
    if missing:
        evidence: dict[str, object] = {"status": "skip", "missing_native_capabilities": missing, "whole_orchestrator_executed": False, "originals": originals}
    else:
        sparse, lsh = build_extensions(args.build_dir.resolve(), False, args.cxx, args.execution_mode)
        sys.path.insert(0, str(ROOT / "tests/traceability"))
        from test_magicpig_server_cpu import exercise_server
        evidence = exercise_server(sparse, lsh)
        evidence["originals"] = originals
        if args.execution_mode == "simde":
            evidence["simde_source"] = source_record(args.build_dir.resolve() / "simde-dependency")
            evidence["adapter_sha256"] = hashlib.sha256((ROOT / "ports/magicpig/compat/simde_cpu.h").read_bytes()).hexdigest()
            evidence["binary_audits"] = [verify_emulated_binary(sparse, True), verify_emulated_binary(lsh, False)]
            evidence["callee_boundary"] = "maintained C++ bodies compiled with pinned SIMDe; real Torch and FBGEMM runtime conversion dispatch; CPU resident attention/merge; FlashInfer uncalled"
    files = list((ROOT / "ports/magicpig/magicpig_cpu").glob("*.py")) + [ROOT / "tests/traceability/test_magicpig_server_cpu.py", ROOT / "tests/traceability/magicpig_server_suite.json"]
    evidence["maintained_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    native = list((ROOT / "ports/magicpig/library").rglob("*.cc")) + list((ROOT / "ports/magicpig/library").rglob("*.h"))
    evidence["native_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in native}
    evidence["driver_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["tools/check_magicpig_server_cpu.py", "tools/check_magicpig_extension_cpu.py"]}
    evidence["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    evidence["thread_limit"] = os.environ.get("OMP_THREAD_LIMIT")
    evidence["native_required"] = args.require_native
    evidence["execution_required"] = args.require_execution
    evidence["execution_mode"] = args.execution_mode
    evidence["native_simd_verified"] = args.execution_mode == "native" and evidence["status"] == "pass"
    evidence["compiler_flags"] = extension_build_flags(False, args.execution_mode)
    (args.build_dir / "server-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    if evidence["status"] == "skip":
        details = f"SKIPPED commit={evidence['commit']} missing_flags={json.dumps(evidence['missing_native_capabilities'])}"
    elif evidence["status"] == "pass":
        passed = "EMULATED_PASS" if args.execution_mode == "simde" else "EXECUTED_PASS"
        details = (f"{passed} commit={evidence['commit']} "
                   f"main_union_checks={evidence['head_union_checks']} two_KV_GQA4_union_checks={evidence['two_KV_GQA4_union_checks']} "
                   f"overflow_rejections={evidence['overflow_rejections']} branches={json.dumps(evidence['branches'])} "
                   f"full_orchestrator={evidence['full_orchestrator']} gpu_verified={evidence['gpu_verified']}")
    else:
        raise ValueError("unknown server execution status")
    emit_native_notice("MagicPIG emulated CPU server" if args.execution_mode == "simde" else "MagicPIG native CPU server", f"execution={args.execution_mode} mode=bf16_disabled " + details)
    if (args.require_native or args.require_execution) and evidence["status"] == "skip":
        raise SystemExit("required MagicPIG native CPU orchestrator skipped; see server-evidence.json")


if __name__ == "__main__":
    main()
