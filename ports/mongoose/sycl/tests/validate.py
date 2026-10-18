#!/usr/bin/env python3
"""Record distinct SYCL, fallback, and original CUDA compile-only probe results."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import cast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--acpp", default="acpp")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    args.evidence.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    for name, command in (("sycl_cpu", [str(args.binary.resolve())]),
                          ("torch_cpu_fallback", [sys.executable, str(Path(__file__).with_name("fallback.py"))])):
        result = subprocess.run(command, text=True, capture_output=True)
        (args.evidence / (name + ".log")).write_text(result.stdout + result.stderr)
        report[name] = {"exit_code": result.returncode, "status": "passed" if result.returncode == 0 else "failed"}
        if result.returncode:
            raise RuntimeError(f"{name} failed: {result.stdout} {result.stderr}")
    original = root / "ports/mongoose/mongoose_slide/slide_lib/simHash.py"
    tree = ast.parse(original.read_text())
    code = cast(str, next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "kernel" for t in node.targets)))
    probe = args.evidence / "original_fingerprint.cpp"
    probe.write_text("#include <pcuda.hpp>\n" + code)
    result = subprocess.run([args.acpp, "--acpp-targets=generic", "--acpp-pcuda", "-c", str(probe), "-o", str(args.evidence / "original_fingerprint.o")], text=True, capture_output=True)
    (args.evidence / "original_cuda_compile.log").write_text(result.stdout + result.stderr)
    report["original_cuda_cpu_compatibility"] = {
        "status": "compile_failed" if result.returncode else "compile_only_unexecuted",
        "exit_code": result.returncode, "original_body_executed": False,
        "kernel_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "note": "Exact unchanged CUDA string; no synthetic shuffle; compile-only probe never proves kernel execution."}
    files = list((root / "ports/mongoose/sycl").rglob("*.hpp")) + list((root / "ports/mongoose/sycl/tests").glob("*.*")) + [args.binary.resolve(), original, root / "ports/mongoose/lsh_lib/LSH.cpp"]
    report["sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()}
    report["native_gpu_executed"] = False
    (args.evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
