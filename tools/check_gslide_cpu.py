#!/usr/bin/env python3
"""Execute selected, unchanged CUDA bodies under a restricted serial CPU adapter.

This is an arithmetic/serial-state oracle, not a CUDA runtime or race simulator.
The maintained sources are extracted at every invocation; no copied kernel bodies
live in the harness. See ports/g-slide/CPU_VALIDATION.md for coverage boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "ports/g-slide"
SELECTED = {
    "src/lshKnl.cu": ["init_bins_knl", "init_hash_no_sw_knl", "get_hash_knl", "gather_buckets_knl"],
    "src/kernel.cu": ["relu_fwd_slide_in_knl", "softmax_fwd_bp_rowmajor_all_sm_knl", "bp_first_layer_knl", "update_weights_knl"],
    "src/GPUMultiLinkedHashTable.cu": ["GPUMultiLinkedHashTable::d_block_reduce_cnt", "GPUMultiLinkedHashTable::d_activate_labels_seq"],
}


@dataclass(frozen=True)
class Suite:
    name: str
    test_file: Path
    definitions: dict[str, list[str]]
    descriptor: Path | None = None


def load_suites() -> dict[str, Suite]:
    """Enroll each isolated suite; unknown fields/names fail rather than skip."""
    suites = {"baseline": Suite("baseline", ROOT / "tests/traceability/test_gslide_cpu_emulation.cpp", SELECTED)}
    for path in sorted((ROOT / "tests/traceability/gslide_suites").glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or set(data) != {"name", "test_file", "definitions"}:
            raise ValueError(f"invalid suite descriptor: {path}")
        name = data["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name != path.stem or name in suites:
            raise ValueError(f"invalid/duplicate suite name: {path}")
        relative = data["test_file"]
        if not isinstance(relative, str):
            raise ValueError(f"invalid test file: {path}")
        test = (ROOT / relative).resolve()
        if not test.is_relative_to(ROOT / "tests/traceability") or test.suffix != ".cpp" or not test.is_file():
            raise ValueError(f"missing or out-of-scope test file: {path}")
        extra = data["definitions"]
        if not isinstance(extra, dict):
            raise ValueError(f"invalid definitions: {path}")
        definitions = {source: list(symbols) for source, symbols in SELECTED.items()}
        for source, symbols in extra.items():
            if source not in SELECTED or not isinstance(symbols, list) or not symbols or not all(isinstance(symbol, str) and symbol for symbol in symbols):
                raise ValueError(f"invalid source/symbol selection: {path}")
            for symbol in symbols:
                if symbol not in definitions[source]:
                    definitions[source].append(symbol)
        suites[name] = Suite(name, test, definitions, path)
    return suites


def check_enrollment(suites: dict[str, Suite]) -> None:
    """CPU ledger entries must name executed suite files, not just exist."""
    data = json.loads((ROOT / "papers/traceability.json").read_text())
    experiments = list(data["experiments"])
    for path in sorted((ROOT / "papers/traceability").glob("*.json")):
        experiments.extend(json.loads(path.read_text())["experiments"])
    enrolled = {suite.test_file.resolve() for suite in suites.values()}
    mapped: set[Path] = set()
    for experiment in experiments:
        # CPU IDs are a stable execution-scope convention. G3 GPU experiments
        # have their own execution gate and must not be silently counted here.
        if experiment["paper"] == "g-slide-2022" and experiment["id"].endswith("-CPU"):
            path = (ROOT / experiment["test"]["file"]).resolve()
            if path not in enrolled:
                raise ValueError(f"CPU ledger test is not an enrolled suite: {experiment['id']}")
            mapped.add(path)
    for suite in suites.values():
        if suite.test_file.resolve() not in mapped:
            raise ValueError(f"suite has no CPU ledger mapping: {suite.name}")


def extract(source: str, symbol: str) -> tuple[str, int]:
    """Balance braces after masking comments/literals; reject ambiguous bodies."""
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                    lambda match: re.sub(r"[^\n]", " ", match.group()), source)
    pattern = re.compile(r"__device__\s+void\s+" + re.escape(symbol) + r"\s*\([^)]*\)\s*\{|__global__\s+void\s+" + re.escape(symbol) + r"\s*\([^)]*\)\s*\{")
    matches = [match for match in pattern.finditer(masked)
               if "d_rand_nodes" not in match.group()]
    if len(matches) != 1:
        raise ValueError(f"expected one selected definition of {symbol}, got {len(matches)}")
    start = matches[0].start()
    opening = matches[0].end() - 1
    depth = 1
    end = opening + 1
    while depth and end < len(masked):
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    if depth:
        raise ValueError(f"unbalanced definition of {symbol}")
    return source[start:end], source.count("\n", 0, start) + 1


def check_archive() -> int:
    pin = json.loads((ROOT / "papers/g-slide-source-pin.json").read_text())
    for relative, expected in pin["blobs"].items():
        content = (ROOT / "third_party/g-slide" / relative).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if digest != expected:
            raise ValueError(f"upstream archive differs: {relative}")
    actual = {str(path.relative_to(ROOT / "third_party/g-slide"))
              for path in (ROOT / "third_party/g-slide").rglob("*") if path.is_file()}
    if actual != set(pin["blobs"]):
        raise ValueError("upstream archive file set differs from pin")
    return len(actual)


def run_suite(suite: Suite, build: Path, sanitize: bool, archive_count: int) -> None:
    build.mkdir(parents=True, exist_ok=True)
    (build / "gslide_cpu_evidence.json").unlink(missing_ok=True)
    sections = ['#include "gslide_cpu_adapter.h"']
    utils = (PORT / "include/utils.h").read_text()
    sections.append(utils[utils.index("#define FOR_IDX_SYNC"):utils.index("#define CUDA_CHECK")])
    kernel = (PORT / "src/kernel.cu").read_text()
    for name in ("BETA1", "BETA2", "EPS", "MAX_INIT"):
        match = re.search(r"^#define " + name + r"\s+[^\n]+", kernel, re.MULTILINE)
        if match is None:
            raise ValueError(f"missing source constant {name}")
        sections.append(match.group())
    evidence: list[dict[str, str | int]] = []
    for relative, symbols in suite.definitions.items():
        path = PORT / relative
        source = path.read_text()
        for symbol in symbols:
            body, line = extract(source, symbol)
            sections.extend([f'#line {line} "{path.as_posix()}"', body])
            evidence.append({"file": str(path.relative_to(ROOT)), "symbol": symbol,
                             "body_sha256": hashlib.sha256(body.encode()).hexdigest()})
    # Include the exact resolved file that is hashed, even if another directory
    # has a test with the same basename.
    sections.append(f'#include {json.dumps(suite.test_file.as_posix())}')
    unit = build / "gslide_cpu_generated.cpp"
    unit.write_text("\n".join(sections) + "\n")
    executable = build / "gslide_cpu_oracle"
    flags = ["-std=c++14", "-O0", "-g", "-Wall", "-Wextra", "-Wno-sign-compare"]
    if sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-no-pie"]
    command = ["g++", *flags, "-I" + str(ROOT / "tests/traceability"),
               "-I" + str(PORT / "include"), str(unit), "-o", str(executable)]
    subprocess.run(command, check=True)
    subprocess.run([str(executable)], check=True)
    inputs = [Path(__file__), ROOT / "papers/g-slide-source-pin.json",
              ROOT / "tests/traceability/gslide_cpu_adapter.h",
              suite.test_file,
              PORT / "include/CscActNodes.h", PORT / "include/GPUMultiLinkedHashTable.h",
              PORT / "include/utils.h", *(PORT / relative for relative in suite.definitions)]
    if suite.descriptor:
        inputs.append(suite.descriptor)
    report = {"suite": suite.name, "mode": "CPU serial selected CUDA bodies", "gpu_validated": False,
              "shared_kernel_block_size": 1, "warp_reductions": "scalar adapters",
              "archive_blobs_verified": archive_count, "compiler_command": command,
              "compiler_version": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
              "sanitize": sanitize, "definitions": evidence,
              "inputs_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in inputs}}
    (build / "gslide_cpu_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"GSLIDE_CPU_PASS suite={suite.name} definitions={len(evidence)} archive_blobs={archive_count} sanitize={sanitize}")


def run(build: Path, sanitize: bool, selected: list[str] | None = None) -> None:
    # An interrupted/failed invocation must not leave a previous PASS summary
    # looking like this invocation's result in a reused build directory.
    (build / "gslide_cpu_summary.json").unlink(missing_ok=True)
    archive_count = check_archive()
    suites = load_suites()
    names = selected or list(suites)
    if len(set(names)) != len(names) or any(name not in suites for name in names):
        raise ValueError(f"invalid suite selection {names}; available: {list(suites)}")
    check_enrollment(suites)
    for name in names:
        run_suite(suites[name], build / name, sanitize, archive_count)
    summary = {"suites": names, "sanitize": sanitize, "gpu_validated": False,
               "archive_blobs_verified": archive_count}
    (build / "gslide_cpu_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"GSLIDE_CPU_SUITES_PASS suites={len(names)} sanitize={sanitize}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--suite", action="append", help="Run one named suite; repeat to select several. Default: all discovered suites.")
    args = parser.parse_args()
    if args.build_dir:
        run(args.build_dir.resolve(), args.sanitize, args.suite)
    else:
        with tempfile.TemporaryDirectory(prefix="gslide-cpu-") as directory:
            run(Path(directory), args.sanitize, args.suite)


if __name__ == "__main__":
    main()
