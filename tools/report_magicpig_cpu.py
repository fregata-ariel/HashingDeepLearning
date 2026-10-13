#!/usr/bin/env python3
"""Expose validated CPU execution/skip evidence through GitHub check notices."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import cast

SPARSE_SOURCE = "ports/magicpig/library/sparse_attention/sparse_attention.cc"
NATIVE_EXP_MARKER = "MAGICPIG-EXP-NATIVE-CPU cases=2821"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def record(value: object, label: str) -> dict[str, object]:
    require(isinstance(value, dict), f"invalid {label} record")
    return cast(dict[str, object], value)


def notices(summary: dict[str, object], commit: str, require_native_exp: bool = False, expected_sparse_source_hash: str | None = None) -> tuple[str, list[tuple[str, str]]]:
    """Reject incomplete/native-marker/leak-disabled evidence before reporting.

    Portable host exp never establishes the native #74 regression. Absence of
    native results is a capability skip, not a numerical pass. Compilation of
    a native body alone does not create an execution result in driver evidence.
    """
    require(summary.get("status") == "pass", "CPU evidence did not pass")
    require(type(summary.get("sanitize")) is bool, "missing sanitizer mode")
    phase = "sanitized" if summary["sanitize"] else "plain"
    raw_results = summary.get("results")
    require(isinstance(raw_results, list) and bool(raw_results), "missing CPU suite results")
    results = [record(item, "suite") for item in cast(list[object], raw_results)]
    require(all(item.get("status") == "pass" for item in results), "CPU suite did not pass")
    for item in results:
        if expected_sparse_source_hash is not None:
            sources = record(item.get("source_sha256", {}), "suite source hashes")
            if SPARSE_SOURCE in sources:
                require(sources[SPARSE_SOURCE] == expected_sparse_source_hash,
                        "tested sparse attention source differs from checkout")
        mode = item.get("mode")
        require(isinstance(mode, str) and mode in {"portable", "portable_python", "avx512", "bf16"}, "unknown CPU suite execution mode")
        if mode == "portable_python":
            require(item.get("sanitizer") == "not_applicable_python_adapter", "Python adapter sanitizer scope is not explicit")
        else:
            require(item.get("sanitize") is summary["sanitize"], "suite sanitizer mode differs from summary")
            require(item.get("leak_check_explicitly_disabled") is False, "CPU evidence disabled leak detection")
    require(any(item.get("mode") == "portable" for item in results),
            "combined CPU reporting requires actual portable C++ execution")
    capability = record(summary.get("capabilities"), "capability")
    capability_modes = record(capability.get("modes"), "capability modes")
    output: list[tuple[str, str]] = []
    for mode in ("avx512", "bf16"):
        decision = record(capability_modes.get(mode), mode)
        require(type(decision.get("eligible")) is bool, f"missing {mode} capability decision")
        executed = [item for item in results if item.get("mode") == mode]
        if executed:
            require(decision["eligible"] is True, f"{mode} executed without a confirmed capability")
            if mode == "avx512":
                softmax = [item for item in executed if item.get("suite") == "softmax"]
                exp_executed = len(softmax) == 1 and re.search(r"(?m)^MAGICPIG-EXP-NATIVE-CPU cases=2821(?:[ \t]|$)", str(softmax[0].get("stdout", ""))) is not None
                require(not require_native_exp or exp_executed, "enrolled native exponential execution marker missing")
                if exp_executed:
                    output.append((f"MagicPIG {phase} native exponential", f"ACTUALLY EXECUTED PASS; {NATIVE_EXP_MARKER}; commit={commit}"))
                else:
                    output.append((f"MagicPIG {phase} native exponential", f"PENDING #74; 2821-case repair regression not enrolled/executed; commit={commit}"))
            names = ",".join(sorted(str(item.get("suite", "unknown")) for item in executed))
            output.append((f"MagicPIG {phase} {mode}", f"ACTUALLY EXECUTED PASS; suites={len(executed)}; names={names}; commit={commit}"))
        else:
            require(decision["eligible"] is False, f"eligible {mode} has no execution result")
            reason = decision.get("reason")
            require(isinstance(reason, str) and bool(reason), f"missing {mode} skip reason")
            output.append((f"MagicPIG {phase} {mode}", f"SKIP; {reason}; native repair not verified; commit={commit}"))
    cpp_count = sum(item.get("mode") != "portable_python" for item in results)
    python_count = len(results) - cpp_count
    output.append((f"MagicPIG CPU {phase}",
                   f"PASS; suites={len(results)}; cpp_suites={cpp_count}; cpp_leaks_disabled=False; "
                   f"python_adapters={python_count} (not sanitizer instrumented); "
                   f"gcc_major={capability.get('gcc_major', 'unknown')}; "
                   f"sparse_attention_sha256={expected_sparse_source_hash or 'not supplied'}; commit={commit}"))
    return phase, output


def escape_notice(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def native_exp_enrolled(root: Path) -> bool:
    paths = [root / "papers/traceability.json", *sorted((root / "papers/traceability").glob("*.json"))]
    return any(experiment.get("id") == "MAGICPIG-EXP-NATIVE-CPU"
               for path in paths for experiment in json.loads(path.read_text())["experiments"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    root = Path(__file__).resolve().parents[1]
    require_exp = native_exp_enrolled(root)
    source_hash = hashlib.sha256((root / SPARSE_SOURCE).read_bytes()).hexdigest()
    output: list[tuple[str, str, str]] = []
    phases: set[str] = set()
    failed = False
    for path in args.evidence:
        phase = path.parent.name
        require(phase in {"plain", "sanitized"}, "evidence parent must identify plain or sanitized phase")
        require(phase not in phases, f"duplicate {phase} evidence")
        phases.add(phase)
        try:
            observed, entries = notices(record(json.loads(path.read_text()), "summary"), commit, require_exp, source_hash)
            require(observed == phase, "evidence sanitizer mode disagrees with its phase path")
        except (OSError, ValueError) as error:
            # A failed driver does not leave final success evidence. Preserve
            # its actual diagnostic without inferring a native capability skip
            # or substituting the preceding plain run for sanitizer execution.
            failed = True
            log = path.parent / "run.log"
            diagnostic = log.read_text(errors="replace")[-4000:] if log.is_file() else "phase log absent; execution not established"
            output.append(("error", f"MagicPIG CPU {phase} failure",
                           f"FAIL/INCOMPLETE; native validation not established; commit={commit}; {error}\n{diagnostic}"))
        else:
            output.extend(("notice", title, value) for title, value in entries)
    require(phases == {"plain", "sanitized"}, "both plain and sanitized execution evidence are required")
    require(len(output) <= 10, "reporting exceeds GitHub per-step annotation limit")
    for level, title, value in output:
        print(f"::{level} title={escape_notice(title)}::{escape_notice(value)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
