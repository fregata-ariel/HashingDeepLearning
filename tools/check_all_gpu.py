#!/usr/bin/env python3
"""Bounded, explicitly enumerated GPU profiles with retained failure evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sycl", action="store_true")
    parser.add_argument("--only", choices=("all", "gslide-native", "gslide-library", "mongoose-original", "mongoose-wrapper", "mongoose-fallback", "mongoose-rng", "scatterbrain", "magicpig", "sycl"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        parser.error("fresh output directory required")
    records: list[dict[str, object]] = []
    for name in ("python-packages.txt", "os-packages.txt", "pip-install.json"):
        manifest = Path("/opt") / name
        if manifest.exists():
            shutil.copyfile(manifest, output / name)

    def run(name: str, command: list[str], timeout: int = 600, summary: str | None = None) -> None:
        selected_name = "scatterbrain" if name.startswith("scatterbrain-") else name
        if name != "device" and args.only not in ("all", selected_name):
            records.append({"profile":name,"status":"NOT_RUN","reason":"another profile selected explicitly"})
            return
        started = time.perf_counter()
        with (output / f"{name}.log").open("w") as log:
            try:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                log.write(f"\nTIMEOUT after {timeout} seconds\n")
                code = 124
            except OSError as error:
                log.write(f"\nRequired command unavailable: {error}\n")
                code = 127
        if summary is not None and code == 0 and summary not in (output / f"{name}.log").read_text():
            code = 125
        record = {"profile": name, "command": command, "exit_code": code,
                  "status": "PASS" if code == 0 else "FAIL", "wall_seconds": time.perf_counter() - started,
                  "log_sha256": hashlib.sha256((output / f"{name}.log").read_bytes()).hexdigest()}
        records.append(record)
        print(f"{name}: {record['status']} ({record['wall_seconds']:.2f}s)", flush=True)
        # Save a progress report after each process so interruption is visible.
        (output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")

    run("device", ["nvidia-smi"])
    if args.only == 'gslide-library':
        run('gslide-library',[sys.executable,'tools/check_gslide_library.py','--build-dir',str(output/'gslide-library')],timeout=900,summary='GSLIDE_LIBRARY_PASS')
    run("gslide-native", [sys.executable, "tools/check_nvidia_gpu.py", "--build-dir", str(output / "gslide"),
        "--cuda-arch", "75", "--cuda-host-compiler", "g++-12", "--jobs", "2", "--compute-sanitizer", "/opt/compute-sanitizer/bin/compute-sanitizer"])
    for profile in ("mongoose-original", "mongoose-wrapper", "mongoose-fallback", "mongoose-rng", "scatterbrain", "magicpig"):
        run(profile, [sys.executable, "tests/gpu_lineages/run.py", "--profile", profile,
            "--output", str(output / f"{profile}.json"), "--upstream", "/opt/fast-transformers", "--build-dir", str(output / "scatterbrain-build")])
    scatterbrain = next(record for record in records if record["profile"] == "scatterbrain")
    for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
        name = "scatterbrain-" + tool
        if scatterbrain["status"] != "PASS":
            records.append({"profile":name,"status":"NOT_RUN","reason":"successful fresh Scatterbrain native build required"})
            continue
        clean = "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" if tool == "racecheck" else "ERROR SUMMARY: 0 errors"
        run(name, ["/opt/compute-sanitizer/bin/compute-sanitizer", "--tool", tool, "--target-processes", "all", "--error-exitcode", "86",
            sys.executable, "tests/gpu_lineages/run.py", "--profile", "scatterbrain", "--reuse-build",
            "--output", str(output / f"{name}.json"), "--upstream", "/opt/fast-transformers", "--build-dir", str(output / "scatterbrain-build")], summary=clean)
    if args.sycl:
        run("sycl", ["bash", "tools/check_sycl_gpu.sh", str(output / "sycl")], timeout=900)
    else:
        records.append({"profile":"sycl", "status":"NOT_RUN", "reason":"components image selected; use sycl target"})
    status = "FAIL" if any(r["status"] == "FAIL" for r in records) else "PARTIAL"
    inputs = [p for directory in ("ports/sycl-common", "ports/g-slide/sycl", "ports/mongoose/sycl", "ports/scatterbrain/sycl", "ports/scatterbrain/cuda", "tests/gslide_sycl", "tests/gpu_lineages", "docker/gpu") for p in (root / directory).rglob("*") if p.is_file() and p.suffix in (".hpp", ".cpp", ".cu", ".py", ".txt", ".sh")]
    inputs += [Path(__file__), root / "tools/check_sycl_gpu.sh", root / "tools/run_gpu_container.sh", root / "docker/gpu/Dockerfile"]
    binaries = [output / "sycl" / p for p in ("gslide/gslide_sycl_test", "gslide/gslide_sycl_training_test", "mongoose/mongoose_sycl_fixture", "scatterbrain/libscatterbrain_sycl.so")]
    report = {"status": status, "profiles": records, "source_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "sycl_binary_sha256":{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries if p.exists()}, "remaining": ["multi-device DDP/NCCL", "MagicPIG original BF16 server/model generation",
        "Scatterbrain causal hybrid", "general native G-SLIDE stress/checkpointing", "same-input CUDA/SYCL parity and performance"],
        "scope":"bounded components; PARTIAL never means all research acceptance passed"}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"GPU aggregate: {status}")
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
