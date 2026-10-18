#!/usr/bin/env python3
"""Build actual maintained Torch/FBGEMM CPU extensions, then independent oracles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def emit_native_notice(title: str, message: str) -> None:
    """Emit one bounded Actions notice, escaping workflow-command delimiters."""
    def escape(value: str) -> str:
        """Encode percent before CR/LF so data cannot inject another command."""
        return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")

    print(f"::notice title={escape(title)}::{escape(message[:1000])}")


def emit_mode_notices(results: list[dict[str, object]], commit: str) -> None:
    """Distinguish actual native success from capability skips for each build."""
    for result in results:
        mode = "bf16_enabled" if result["bf16"] else "bf16_disabled"
        if result["status"] == "skip":
            details = f"SKIPPED commit={commit} missing_flags={json.dumps(result['missing'])}"
        elif result["status"] == "pass":
            details = (f"EXECUTED_PASS commit={commit} heads={result['heads_checked']} "
                       f"dispatch={json.dumps(result['dispatch_traces'], separators=(',', ':'))} "
                       f"cxx11_abi={result['cxx11_abi']} fbgemm_bit_mismatches={result['fbgemm_model_bit_mismatches']}")
        else:
            raise ValueError("unknown native execution status")
        emit_native_notice("MagicPIG native CPU", f"mode={mode} {details}")


def build_extensions(build: Path, bf16: bool, cxx: str = "g++-14") -> tuple[ModuleType, ModuleType]:
    """Use the wheel ABI; FBGEMM conversion/dispatch links from libtorch_cpu."""
    import torch
    from torch.utils.cpp_extension import load
    os.environ["CXX"] = cxx
    os.environ["MAX_JOBS"] = "1"
    flags = ["-O0", "-g0", "-std=c++20", "-mavx512f", "-mavx512bw", "-mavx512dq", "-mavx512vl", "-mfma", "-fopenmp", "-DMAGICPIG_REAL_TORCH", "-DMAGICPIG_TRACE_DISPATCH", "-fsanitize=undefined", "-fno-sanitize-recover=all"]
    if bf16:
        flags.append("-mavx512bf16")
    modules: list[ModuleType] = []
    for stem, library in [("sparse", "sparse_attention"), ("lsh", "lsh")]:
        directory = build / (stem + ("_bf16" if bf16 else "_fp32"))
        directory.mkdir(parents=True, exist_ok=True)
        source = ROOT / "ports/magicpig/library" / library / (library + ".cc")
        module = load(name="magicpig_" + directory.name, sources=[str(source)], extra_include_paths=[str(ROOT / "third_party/magicpig/library/sparse_attention/3rdparty/FBGEMM/include")], extra_cflags=flags, extra_ldflags=["-fopenmp", "-fsanitize=undefined"], build_directory=str(directory), verbose=True)
        modules.append(module)
    assert torch.version.cuda is None, "CPU wheel required for this CPU boundary fixture"
    return modules[0], modules[1]


def main() -> None:
    """Emit execution and capability skips separately with source identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cxx", default="g++-14")
    parser.add_argument("--require-native", action="store_true", help="fail after recording evidence if either native build mode skips")
    args = parser.parse_args()
    flags = Path("/proc/cpuinfo").read_text()
    results: list[dict[str, object]] = []
    for bf16 in [False, True]:
        required = ["avx512f", "avx512bw", "avx512dq", "avx512vl", "fma"] + (["avx512_bf16"] if bf16 else [])
        missing = [flag for flag in required if flag not in flags]
        if missing:
            results.append({"bf16": bf16, "status": "skip", "missing": missing})
            continue
        sparse, lsh = build_extensions(args.build_dir.resolve(), bf16, args.cxx)
        sys.path.insert(0, str(ROOT / "tests/traceability"))
        from test_magicpig_extension_cpu import exercise
        results.append(exercise(sparse, lsh, bf16))
    sources = list((ROOT / "ports/magicpig/library").rglob("*.cc")) + list((ROOT / "ports/magicpig/library").rglob("*.h"))
    evidence = {"results": results, "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "gpu_verified": False, "fbgemm": "actual libtorch_cpu exported FBGEMM; archived declarations, wheel implementation", "sanitizer": "UBSan on maintained extension; unsanitized Torch wheel", "fixture_sha256": hashlib.sha256((ROOT / "tests/traceability/test_magicpig_extension_cpu.py").read_bytes()).hexdigest(), "descriptor_sha256": hashlib.sha256((ROOT / "tests/traceability/magicpig_extension_suite.json").read_bytes()).hexdigest()}
    args.build_dir.mkdir(parents=True, exist_ok=True)
    evidence["native_required"] = args.require_native
    (args.build_dir / "extension-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    emit_mode_notices(results, str(evidence["commit"]))
    if args.require_native and any(result["status"] == "skip" for result in results):
        raise SystemExit("required MagicPIG native build mode skipped; see extension-evidence.json")


if __name__ == "__main__":
    main()
