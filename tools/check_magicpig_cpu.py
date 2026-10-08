#!/usr/bin/env python3
"""Run independent tiny CPU oracles against selected maintained MagicPIG bodies.

Always execute portable suites. Native SIMD is opt-in and guarded in the parent
process before compiling/running SIMD translation units with global vectors.
This does not build Torch/FBGEMM extensions or execute GPU/model generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "ports/magicpig"
SOURCE = "library/sparse_attention/sparse_attention.cc"
ALLOWED_SOURCES = {SOURCE, "library/lsh/lsh.cc"}
PRELUDE = "#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cstring>\n#include <iostream>\n#include <vector>\n"


@dataclass(frozen=True)
class Suite:
    name: str
    test_file: Path
    definitions: dict[str, list[str]]
    descriptor: Path
    adapter_file: Path | None = None


def fixture_path(relative: str, suffix: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT / "tests/traceability") or path.suffix != suffix or not path.is_file():
        raise ValueError(f"invalid/missing fixture: {relative}")
    return path


def load_suites() -> dict[str, Suite]:
    suites: dict[str, Suite] = {}
    for descriptor in sorted((ROOT / "tests/traceability/magicpig_suites").glob("*.json")):
        data = json.loads(descriptor.read_text())
        required = {"name", "test_file", "definitions"}
        if not isinstance(data, dict) or not required <= set(data) or set(data) - required - {"adapter_file"}:
            raise ValueError(f"invalid descriptor: {descriptor}")
        name = data["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name != descriptor.stem or name in suites:
            raise ValueError(f"invalid/duplicate suite: {descriptor}")
        definitions = data["definitions"]
        if not isinstance(definitions, dict) or not definitions:
            raise ValueError(f"missing definitions: {descriptor}")
        for source, symbols in definitions.items():
            if source not in ALLOWED_SOURCES or not isinstance(symbols, list) or not symbols or len(set(symbols)) != len(symbols) or not all(isinstance(s, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9:]*", s) for s in symbols):
                raise ValueError(f"invalid source/symbol selection: {descriptor}")
        test = fixture_path(data["test_file"], ".cpp")
        adapter = fixture_path(data["adapter_file"], ".hpp") if "adapter_file" in data else None
        suites[name] = Suite(name, test, definitions, descriptor, adapter)
    if "baseline" not in suites:
        raise ValueError("baseline suite is not enrolled")
    return suites


def load_bf16_suite() -> Suite:
    path = ROOT / "tests/traceability/magicpig_native_suites/bf16_tile.json"
    data = json.loads(path.read_text())
    expected = {"name": "bf16_tile", "test_file": "tests/traceability/test_magicpig_bf16_cpu.cpp", "definitions": {SOURCE: ["qk_kernel_bf16_impl"]}}
    if data != expected:
        raise ValueError("invalid BF16 native suite descriptor")
    return Suite(data["name"], fixture_path(data["test_file"], ".cpp"), data["definitions"], path)


def check_enrollment(suites: dict[str, Suite], bf16: Suite) -> None:
    data = json.loads((ROOT / "papers/traceability.json").read_text())
    experiments = list(data["experiments"])
    for path in sorted((ROOT / "papers/traceability").glob("*.json")):
        experiments.extend(json.loads(path.read_text())["experiments"])
    enrolled = {s.test_file.resolve() for s in [*suites.values(), bf16]}
    mapped: set[Path] = set()
    for record in experiments:
        if record["paper"] == "magicpig-2024" and record["id"].endswith("-CPU"):
            path = (ROOT / record["test"]["file"]).resolve()
            if path not in enrolled:
                raise ValueError(f"CPU ledger test is not enrolled: {record['id']}")
            mapped.add(path)
    if mapped != enrolled:
        raise ValueError("suite has no CPU ledger mapping")


def extract_definition(source: str, symbol: str) -> tuple[str, int]:
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', lambda m: re.sub(r"[^\n]", " ", m.group()), source)
    pattern = re.compile(r"(?m)^(?:void|int|__m512)\s+" + re.escape(symbol) + r"\s*\([^)]*\)\s*\{")
    matches = list(pattern.finditer(masked))
    if len(matches) != 1:
        raise ValueError(f"expected one definition of {symbol}, got {len(matches)}")
    start, end = matches[0].start(), matches[0].end()
    depth = 1
    while depth and end < len(masked):
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    if depth:
        raise ValueError(f"unbalanced body: {symbol}")
    return source[start:end], source.count("\n", 0, start) + 1


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide_modes(flags: set[str], machine: str, gcc_major: int | None, native_requested: bool) -> dict[str, dict[str, bool | str]]:
    x86 = machine.lower() in {"x86_64", "amd64", "i386", "i486", "i586", "i686", "x86"}
    result: dict[str, dict[str, bool | str]] = {}
    for name, needed in (("avx512", {"avx512f", "fma"}), ("bf16", {"avx512f", "avx512bw", "avx512_bf16"})):
        if not native_requested:
            reason = "native checks not requested"
        elif not x86:
            reason = "unknown or non-x86 architecture"
        elif gcc_major is None:
            reason = "GCC compiler identity unavailable"
        elif name == "bf16" and gcc_major < 11:
            reason = "BF16 requires GCC >= 11"
        elif needed - flags:
            reason = "missing CPU/OS features: " + ",".join(sorted(needed - flags))
        else:
            reason = "CPU/OS and compiler capabilities confirmed"
        result[name] = {"eligible": reason == "CPU/OS and compiler capabilities confirmed", "reason": reason}
    return result


def checked(command: list[str], *, env: dict[str, str] | None = None, input_text: str | None = None) -> str:
    run = subprocess.run(command, input=input_text, capture_output=True, text=True, env=env)
    if run.returncode:
        raise RuntimeError(f"command failed ({run.returncode}): {command}\n{run.stdout}\n{run.stderr}")
    return run.stdout


def capabilities(cxx: str, build: Path, requested: bool) -> dict[str, object]:
    version = checked([cxx, "--version"]).splitlines()[0]
    macros = checked([cxx, "-dM", "-E", "-x", "c++", "-"], input_text="")
    major_match = re.search(r"^#define __GNUC__ (\d+)$", macros, re.M)
    major = int(major_match[1]) if major_match and "__clang__" not in macros else None
    machine = platform.machine()
    flag_sets = []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags") and ":" in line:
                flag_sets.append(set(line.split(":", 1)[1].split()))
    except OSError:
        pass
    flags: set[str] = set()
    probe_status = "not attempted"
    probe_command: list[str] = []
    if requested and major is not None and machine.lower() in {"x86_64", "amd64", "i386", "i486", "i586", "i686", "x86"}:
        # No ISA flags/global SIMD vectors: GCC builtins also check OS support.
        probe = build / "cpu_capabilities.cpp"
        probe.write_text('#include <cstdio>\nint main(){__builtin_cpu_init();\nif(__builtin_cpu_supports("avx512f"))puts("avx512f");\nif(__builtin_cpu_supports("fma"))puts("fma");\nif(__builtin_cpu_supports("avx512bw"))puts("avx512bw");\n#if __GNUC__ >= 11\nif(__builtin_cpu_supports("avx512bf16"))puts("avx512_bf16");\n#endif\n}\n')
        executable = build / "cpu_capabilities"
        probe_command = [cxx, "-std=c++17", "-O0", str(probe), "-o", str(executable)]
        try:
            checked(probe_command)
            flags = set(checked([str(executable)]).split())
            if flag_sets:
                flags &= set.intersection(*flag_sets)
            probe_status = "GCC builtin CPU/OS probe; intersected with /proc flags when present"
        except RuntimeError as error:
            probe_status = "failed; native checks disabled: " + str(error)
            flags = set()
    return {"machine": machine, "compiler_version": version, "gcc_major": major, "flags": sorted(flags), "proc_flags_available": bool(flag_sets), "probe_status": probe_status, "probe_command": probe_command, "probe_sha256": digest(build / "cpu_capabilities.cpp") if probe_command else None, "modes": decide_modes(flags, machine, major, requested)}


def compile_run(suite: Suite, mode: str, build: Path, cxx: str, sanitize: bool, disable_leaks: bool) -> dict[str, object]:
    directory = build / mode / suite.name
    directory.mkdir(parents=True, exist_ok=True)
    evidence_path = directory / "evidence.json"
    evidence_path.unlink(missing_ok=True)
    prelude = PRELUDE
    if sanitize and disable_leaks:
        prelude += 'extern "C" const char* __asan_default_options(){return "detect_leaks=0";}\n'
    sources: dict[str, str] = {}
    definitions = {p: list(s) for p, s in suite.definitions.items()}
    adapters = []
    if mode == "portable":
        adapter = ROOT / "tests/traceability/magicpig_fp32_adapter.hpp"
        prelude += '#include ' + json.dumps(str(adapter)) + '\n'
        adapters.append(adapter)
    else:
        prelude += '#include <immintrin.h>\n'
        if mode == "bf16":
            prelude += 'using bfloat16 = std::uint16_t;\n'
    if suite.adapter_file:
        prelude += '#include ' + json.dumps(str(suite.adapter_file)) + '\n'
        adapters.append(suite.adapter_file)
    pieces = [prelude]
    records = []
    if mode == "avx512":
        source = (PORT / SOURCE).read_bytes().decode()
        constants = re.findall(r"(?ms)^const __m512\s+(?:LOG2E_VEC|MAGIC_FLOAT_BIAS|ONE_VEC|LN2_PART_VEC|EXP_POLY_COEFFS)\b.*?;", source)
        if len(constants) != 5:
            raise ValueError("native polynomial constants missing/ambiguous")
        constant_block = "\n".join(constants)
        pieces.append(constant_block)
        records.append({"symbol": "polynomial_constants", "sha256": hashlib.sha256(constant_block.encode()).hexdigest()})
        definitions.setdefault(SOURCE, []).insert(0, "avx512_exp_ps")
    for relative, symbols in definitions.items():
        source_file = PORT / relative
        source = source_file.read_bytes().decode()
        sources[str(source_file.relative_to(ROOT))] = digest(source_file)
        for symbol in symbols:
            body, line = extract_definition(source, symbol)
            pieces.append(body)
            records.append({"source": relative, "symbol": symbol, "line": line, "sha256": hashlib.sha256(body.encode()).hexdigest()})
    pieces.append('#include ' + json.dumps(str(suite.test_file)) + '\n')
    translation = directory / "selected_bodies.cpp"
    translation.write_text("\n".join(pieces))
    executable = directory / "oracle"
    command = [cxx, "-std=c++17", "-O1", "-fno-tree-vectorize", "-fno-tree-slp-vectorize"]
    if sanitize:
        command += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-no-pie"]
    if mode == "avx512":
        command += ["-mavx512f", "-mfma", "-DMAGIC_NATIVE_AVX512=1"]
    if mode == "bf16":
        command += ["-mavx512f", "-mavx512bw", "-mavx512bf16"]
    command += [str(translation), "-o", str(executable)]
    checked(command)
    env = dict(os.environ)
    if disable_leaks:
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":detect_leaks=0"
    output = checked([str(executable)], env=env)
    print(output, end="" if output.endswith("\n") else "\n")
    headers = {str(p.relative_to(ROOT)): digest(p) for p in sorted(PORT.rglob("*.h"))}
    evidence = {"suite": suite.name, "mode": mode, "status": "pass", "sanitize": sanitize, "leak_check_explicitly_disabled": disable_leaks, "compiler_command": command, "translation_sha256": digest(translation), "source_sha256": sources, "recorded_native_header_sha256": headers, "definitions": records, "fixture": str(suite.test_file.relative_to(ROOT)), "fixture_sha256": digest(suite.test_file), "descriptor_sha256": digest(suite.descriptor), "adapter_sha256": {str(p.relative_to(ROOT)): digest(p) for p in adapters}, "stdout": output, "gpu_validated": False, "torch_extension_validated": False}
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def run(build: Path, cxx: str, sanitize: bool, native: bool, suites_requested: list[str], disable_leaks: bool) -> None:
    build.mkdir(parents=True, exist_ok=True)
    summary_path = build / "magicpig_cpu_evidence.json"
    summary_path.unlink(missing_ok=True)
    for old_evidence in build.glob("*/*/evidence.json"):
        old_evidence.unlink(missing_ok=True)
    suites = load_suites()
    bf16 = load_bf16_suite()
    check_enrollment(suites, bf16)
    if len(set(suites_requested)) != len(suites_requested) or set(suites_requested) - set(suites):
        raise ValueError("unknown or duplicate requested suite")
    selected = [suites[n] for n in suites_requested] if suites_requested else list(suites.values())
    # Preserve the independent immutable-source gate before any numerical test.
    checked([os.sys.executable, str(ROOT / "tools/check_magicpig_sources.py")])
    cap = capabilities(cxx, build, native)
    results = [compile_run(s, "portable", build, cxx, sanitize, disable_leaks) for s in selected]
    for mode in ("avx512", "bf16"):
        decision = cap["modes"][mode]
        if not decision["eligible"]:
            print(f"MAGICPIG_NATIVE_SKIP mode={mode} reason={decision['reason']}")
        elif mode == "avx512":
            native_suites = [s for s in selected if s.name == "baseline"]
            if not native_suites:
                print("MAGICPIG_NATIVE_SKIP mode=avx512 reason=baseline suite not selected")
            results.extend(compile_run(s, mode, build, cxx, sanitize, disable_leaks) for s in native_suites)
        else:
            results.append(compile_run(bf16, mode, build, cxx, sanitize, disable_leaks))
    summary = {"status": "pass", "driver_sha256": digest(Path(__file__)), "source_pin_sha256": digest(ROOT / "papers/magicpig-source-pin.json"), "fixture_seed": "fixed literals, no RNG", "capabilities": cap, "sanitize": sanitize, "results": results, "gpu_validated": False, "torch_extension_validated": False}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"MAGICPIG_CPU_PASS portable_suites={len(selected)} executed_modes={','.join(sorted({r['mode'] for r in results}))} sanitize={sanitize}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--sanitize", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--disable-leak-check", action="store_true", help="explicit local-only workaround; hosted CI keeps LeakSanitizer enabled")
    args = parser.parse_args()
    if args.disable_leak_check and not args.sanitize:
        parser.error("--disable-leak-check requires --sanitize")
    if args.build_dir:
        run(args.build_dir.resolve(), args.cxx, args.sanitize, args.native, args.suite, args.disable_leak_check)
    else:
        with tempfile.TemporaryDirectory(prefix="magicpig-cpu-") as directory:
            run(Path(directory), args.cxx, args.sanitize, args.native, args.suite, args.disable_leak_check)


if __name__ == "__main__":
    main()
