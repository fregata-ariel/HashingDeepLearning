#!/usr/bin/env python3
"""Required native NVIDIA checkpoint; stdlib only, no downloads or CPU fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(build: Path, arch: str, host_compiler: str, jobs: int, sanitizer: str = "compute-sanitizer") -> int:
    # A fresh directory prevents previous binaries/reports from proving this run.
    build.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "status": "FAIL", "profile": "gslide-native-cuda", "stages": [],
        "gpu_validated": False, "cuda_arch": arch,
        "diagnostics_required": ["memcheck", "racecheck", "initcheck", "synccheck"],
        "pending": ["shared SYCL NVIDIA parity/performance", "model checkpoint save/reload",
                    "MagicPIG CUDA/FlashInfer", "Scatterbrain CUDA", "MONGOOSE CUDA/RNG/DDP",
                    "paper accuracy/performance", "paid execution and TTL calibration"],
    }
    stages: list[dict[str, object]] = []
    report["stages"] = stages
    result = build / "result.json"
    stage = "metadata"

    def save() -> None:
        result.write_text(json.dumps(report, indent=2) + "\n")

    def command(name: str, args: list[str], timeout: int = 120, echo: bool = True) -> str:
        nonlocal stage
        stage = name
        log = build / (name + ".log")
        entry: dict[str, object] = {"name": name, "command": args, "status": "FAIL"}
        stages.append(entry)
        save()
        start = time.monotonic()
        try:
            # Stream to disk so failed commands and killed processes retain output.
            with log.open("w") as stream:
                completed = subprocess.run(args, cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=timeout)
            entry["exit_code"] = completed.returncode
            output = log.read_text(errors="replace")
            if echo:
                print(output[-6000:], end="", flush=True)
            completed.check_returncode()
            entry["status"] = "PASS"
            return output
        finally:
            entry["wall_seconds"] = time.monotonic() - start
            if log.exists():
                entry["log_sha256"] = digest(log)
            save()

    try:
        if not re.fullmatch(r"[0-9]{2,3}", arch) or jobs < 1 or jobs > 16:
            raise ValueError("invalid CUDA architecture or build parallelism")
        for tool in ("git", "cmake", "ctest", "nvcc", host_compiler,
                     "nvidia-smi", sanitizer, "/usr/bin/time"):
            if shutil.which(tool) is None:
                raise RuntimeError("required tool missing: " + tool)
        report["checkout"] = command("checkout", ["git", "rev-parse", "HEAD"]).strip()
        report["tree"] = command("tree", ["git", "rev-parse", "HEAD^{tree}"]).strip()
        command("git-status", ["git", "status", "--porcelain=v1"])
        command("working-tree", ["git", "diff", "--binary", "HEAD"], echo=False)
        report["environment"] = {key: os.environ.get(key) for key in
                                 ("CUDA_VISIBLE_DEVICES", "CUDAHOSTCXX", "CUDACXX")}
        inputs = sorted({p for directory in (ROOT / "ports/g-slide", ROOT / "tests/gslide_cuda",
                                             ROOT / "third_party/g-slide")
                         for p in directory.rglob("*") if p.is_file()} |
                        {ROOT / "tools/check_nvidia_gpu.py", ROOT / "tools/check_gslide_cpu.py",
                         ROOT / "tools/generate_gslide_cuda_original.py",
                         ROOT / "tools/bootstrap_nvidia_sanitizer.sh",
                         ROOT / "tests/traceability/test_nvidia_gpu_reporting.py",
                         ROOT / "papers/traceability/gslide-cuda.json",
                         ROOT / ".github/workflows/nvidia-gpu.yml"})
        # Bind new/untracked inputs as well as the tracked patch.
        report["inputs_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
        report["python"] = sys.version
        command("host-os", ["uname", "-a"])
        command("host-cpu", ["lscpu"], echo=False)
        command("nvcc-version", ["nvcc", "--version"])
        command("host-compiler", [host_compiler, "--version"])
        command("cmake-version", ["cmake", "--version"])
        command("sanitizer-version", [sanitizer, "--version"])
        command("gpu-before", ["nvidia-smi"])
        gpu = command("gpu-identity", ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version,memory.total",
                                       "--format=csv"])
        report["nvidia_smi"] = gpu
        native = build / "native"
        command("configure", ["cmake", "-S", str(ROOT / "tests/gslide_cuda"), "-B", str(native),
                              "-DCMAKE_BUILD_TYPE=Debug", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                              "-DCMAKE_CUDA_ARCHITECTURES=" + arch,
                              "-DCMAKE_CUDA_HOST_COMPILER=" + str(Path(shutil.which(host_compiler)).resolve())])
        command("build", ["/usr/bin/time", "-v", "-o", str(build / "build-resources.txt"),
                          "cmake", "--build", str(native), "--parallel", str(jobs)], 600)
        binary = native / "gslide_cuda_test"
        report["binary_sha256"] = digest(binary)
        report["generated_original_sha256"] = digest(native / "original.cu")
        output = command("native", ["/usr/bin/time", "-v", "-o", str(build / "native-resources.txt"),
                                    "ctest", "--test-dir", str(native), "-V", "--no-tests=error",
                                    "--output-junit", str(build / "native.xml")], 180)
        for marker in ("ORIGINAL_DENSE", "ORIGINAL_HASH", "DENSE_PASS", "HASH_PASS",
                       "STALE_PASS", "TRAINING_PASS", "GSLIDE_CUDA_PASS"):
            if marker not in output:
                raise RuntimeError("required native fixture did not complete: " + marker)
        report["native_numerical_pass"] = True
        for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
            args = [sanitizer, "--tool", tool, "--error-exitcode", "86"]
            # Debian/Ubuntu package relocates the injection libraries away from
            # the /usr/bin launcher; use its documented path option explicitly.
            packaged = Path("/usr/lib/nvidia-cuda-toolkit/compute-sanitizer")
            if Path(shutil.which(sanitizer)).resolve() == Path("/usr/bin/compute-sanitizer") and packaged.is_dir():
                args += ["--injection-path", str(packaged)]
            if tool == "memcheck":
                args += ["--leak-check", "full"]
            if tool == "racecheck":
                args += ["--racecheck-report", "all"]
            output = command(tool, args + [str(binary)], 240)
            clean = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                     if tool == "racecheck" else "ERROR SUMMARY: 0 errors")
            if "GSLIDE_CUDA_PASS" not in output or clean not in output:
                raise RuntimeError("diagnostic did not confirm a clean completed run: " + tool)
            if tool == "memcheck" and "LEAK SUMMARY: 0 bytes leaked in 0 allocations" not in output:
                raise RuntimeError("memcheck did not confirm zero leaked allocations")
        command("gpu-after", ["nvidia-smi"])
        report["status"] = "PASS"
        report["gpu_validated"] = True
        report["scope"] = "G-SLIDE bounded native fixtures only; see pending"
        print("NVIDIA_GPU_PASS evidence=" + str(result))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report["failed_stage"] = stage
        report["error"] = str(error)
        if stages and stages[-1]["name"] == stage:
            stages[-1]["status"] = "FAIL"
            stages[-1]["failure_reason"] = str(error)
        print("NVIDIA_GPU_FAIL stage=" + stage + " " + str(error), file=sys.stderr)
        return 1
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True, help="must not already exist")
    parser.add_argument("--cuda-arch", default="75")
    parser.add_argument("--cuda-host-compiler", default="g++-12")
    parser.add_argument("--compute-sanitizer", default="compute-sanitizer")
    parser.add_argument("--jobs", default=2, type=int)
    args = parser.parse_args()
    sys.exit(run(args.build_dir.resolve(), args.cuda_arch, args.cuda_host_compiler, args.jobs, args.compute_sanitizer))
