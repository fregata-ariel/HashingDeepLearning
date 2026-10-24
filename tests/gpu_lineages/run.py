#!/usr/bin/env python3
"""Run one bounded GPU profile; always retain a result, including failures."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("mongoose-original", "mongoose-wrapper", "mongoose-fallback", "mongoose-rng", "scatterbrain", "magicpig"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--reuse-build", action="store_true", help="diagnostics only: verify the fixture, sources and native binary hashes from the preceding plain build")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing evidence")
    root = Path(__file__).resolve().parents[2]
    started = time.perf_counter()
    report: dict[str, object] = {"profile": args.profile, "status": "FAIL", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "python": sys.version, "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "sources": {}, "environment": {k: os.environ.get(k) for k in ("CUDA_HOME", "CUDA_PATH", "CC", "CXX", "MAX_JOBS", "TORCH_CUDA_ARCH_LIST")}}
    try:
        import torch
        assert torch.cuda.is_available(), "required NVIDIA device unavailable; no CPU fallback"
        assert torch.cuda.device_count() == 1, "this bounded profile is calibrated for one GPU"
        torch.set_num_threads(2)
        torch.manual_seed(179)
        torch.cuda.manual_seed_all(179)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report.update(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                      torch_cuda=torch.version.cuda, torch_arches=torch.cuda.get_arch_list(),
                      native_bf16=torch.cuda.is_bf16_supported(including_emulation=False))
        torch.cuda.reset_peak_memory_stats()
        if args.profile.startswith("mongoose-"):
            import mongoose
            report["result"] = mongoose.rng_replay() if args.profile == "mongoose-rng" else mongoose.packing(args.profile)
        elif args.profile == "scatterbrain":
            import scatterbrain
            assert args.upstream and args.build_dir
            native_results: dict[str, object] = {}
            report["result"] = native_results
            scatterbrain.check(args.upstream.resolve(), args.build_dir.resolve(), native_results, args.reuse_build)
        else:
            import magicpig
            components: dict[str, object] = {}
            report["result"] = components
            magicpig.check(components)
        torch.cuda.synchronize()
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["status"] = "PASS"
    except Exception:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        sources = list((root / "tests/gpu_lineages").glob("*.py"))
        for directory in ("ports/mongoose", "ports/scatterbrain", "third_party/magicpig/models"):
            sources.extend((root / directory).rglob("*.py"))
        report["sources"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        report["wall_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"GPU_PROFILE {args.profile}: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
