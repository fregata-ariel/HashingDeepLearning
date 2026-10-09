#!/usr/bin/env python3
"""Run independent tiny CPU oracles against selected maintained MagicPIG bodies.

Always execute portable suites. Native SIMD is opt-in and guarded in the parent
process before compiling/running SIMD translation units with global vectors.
This does not build Torch/FBGEMM extensions or execute GPU/model generation.
"""
from __future__ import annotations

import argparse
import ast
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
    native_avx512: bool = False
    probes: tuple[dict[str, str], ...] = ()


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
        if not isinstance(data, dict) or not required <= set(data) or set(data) - required - {"adapter_file", "native_avx512", "probes"}:
            raise ValueError(f"invalid descriptor: {descriptor}")
        name = data["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or name != descriptor.stem or name in suites:
            raise ValueError(f"invalid/duplicate suite: {descriptor}")
        definitions = data["definitions"]
        if not isinstance(definitions, dict) or not definitions:
            raise ValueError(f"missing definitions: {descriptor}")
        for source, symbols in definitions.items():
            if source not in ALLOWED_SOURCES or not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) and re.fullmatch(r"(?:[A-Za-z_]\w*::)*~?[A-Za-z_]\w*", s) for s in symbols) or len(set(symbols)) != len(symbols):
                raise ValueError(f"invalid source/symbol selection: {descriptor}")
        test = fixture_path(data["test_file"], ".cpp")
        adapter = fixture_path(data["adapter_file"], ".hpp") if "adapter_file" in data else None
        native = data.get("native_avx512", False)
        if not isinstance(native, bool):
            raise ValueError(f"invalid native selection: {descriptor}")
        probes = data.get("probes", [])
        if not isinstance(probes, list):
            raise ValueError(f"invalid probes: {descriptor}")
        arguments = set()
        for probe in probes:
            if not isinstance(probe, dict) or set(probe) != {"argument", "marker", "diagnostic", "symbol"} or not all(isinstance(v, str) for v in probe.values()):
                raise ValueError(f"invalid probe: {descriptor}")
            if not re.fullmatch(r"--probe-[a-z-]+", probe["argument"]) or probe["argument"] in arguments or not probe["marker"] or "\n" in probe["marker"] or probe["diagnostic"] not in {"asan_heap_buffer_overflow", "ubsan_null_load"} or probe["symbol"] not in {s for symbols in definitions.values() for s in symbols}:
                raise ValueError(f"invalid probe contract: {descriptor}")
            arguments.add(probe["argument"])
        suites[name] = Suite(name, test, definitions, descriptor, adapter, native, tuple(probes))
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


def load_python_suites() -> dict[str, dict[str, object]]:
    suites = {}
    required = {"name", "test_file", "source_file", "class_name", "blocks"}
    for path in sorted((ROOT / "tests/traceability/magicpig_python_suites").glob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or set(data) != required or data["name"] != path.stem or not re.fullmatch(r"[a-z][a-z0-9_-]*", data["name"]):
            raise ValueError("invalid Python suite descriptor")
        if data["source_file"] != "third_party/magicpig/models/attnserver.py" or data["class_name"] != "LSHSparseAttnServer":
            raise ValueError("unapproved Python source/class")
        if data["blocks"] != ["packing", "centering", "fill_hash", "decode_hash", "append_centering"]:
            raise ValueError("invalid Python block selection")
        fixture_path(data["test_file"], ".py")
        data["descriptor"] = path
        suites[data["name"]] = data
    return suites


def check_enrollment(suites: dict[str, Suite], bf16: Suite, python_suites: dict[str, dict[str, object]] | None = None) -> None:
    data = json.loads((ROOT / "papers/traceability.json").read_text())
    experiments = list(data["experiments"])
    for path in sorted((ROOT / "papers/traceability").glob("*.json")):
        experiments.extend(json.loads(path.read_text())["experiments"])
    enrolled = {s.test_file.resolve() for s in [*suites.values(), bf16]}
    enrolled.update(fixture_path(s["test_file"], ".py") for s in (python_suites or {}).values())
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
    parts = symbol.split("::")
    constructor = len(parts) >= 2 and parts[-1].lstrip("~") == parts[-2]
    prefix = "" if constructor else r"(?:static\s+)?(?:inline\s+)?(?:void|int|__m512|torch::Tensor)\s+"
    pattern = re.compile(r"(?m)^" + prefix + re.escape(symbol) + r"\s*\([^)]*\)\s*\{")
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


def validate_fastfill_probe(probe: subprocess.CompletedProcess[str], sanitize: bool, disable_leaks: bool) -> str:
    if "MAGICPIG-FASTFILL-DEFECT: expected {0,1,2,3}; actual {0}" not in probe.stdout:
        raise RuntimeError(f"fastfill characterization marker missing: {probe.stdout}\n{probe.stderr}")
    if sanitize and not disable_leaks:
        expected = "SUMMARY: AddressSanitizer: 64 byte(s) leaked in 4 allocation(s)"
        if probe.returncode != 1 or "LeakSanitizer: detected memory leaks" not in probe.stderr or expected not in probe.stderr or "LSH::fastfill" not in probe.stderr or "runtime error:" in probe.stderr or "ERROR: AddressSanitizer:" in probe.stderr:
            raise RuntimeError(f"unexpected fastfill sanitizer outcome ({probe.returncode}): {probe.stdout}\n{probe.stderr}")
        return "expected_leak_reproduced"
    if probe.returncode:
        raise RuntimeError(f"fastfill probe failed: {probe.stdout}\n{probe.stderr}")
    return "not_checked"


def validate_boundary_probe(probe: subprocess.CompletedProcess[str], contract: dict[str, str], records: list[dict[str, object]], translation: Path, executable: Path) -> dict[str, object]:
    """Accept only the specified diagnostic at the selected production body."""
    if probe.returncode != 1 or probe.stdout.strip() != contract["marker"]:
        raise RuntimeError("unexpected boundary probe exit or marker")
    diagnostic = probe.stderr
    if contract["diagnostic"] not in {"asan_heap_buffer_overflow", "ubsan_null_load"}:
        raise RuntimeError("unknown boundary probe diagnostic")
    if contract["diagnostic"] == "asan_heap_buffer_overflow":
        if diagnostic.count("ERROR: AddressSanitizer:") != 1 or diagnostic.count("SUMMARY: AddressSanitizer:") != 1 or "ERROR: AddressSanitizer: heap-buffer-overflow" not in diagnostic or "SUMMARY: AddressSanitizer: heap-buffer-overflow" not in diagnostic or "runtime error:" in diagnostic or "LeakSanitizer:" in diagnostic:
            raise RuntimeError("unexpected boundary probe sanitizer category")
    elif diagnostic.count("runtime error:") != 1 or "runtime error: load of null pointer of type 'float'" not in diagnostic or "AddressSanitizer:" in diagnostic or "LeakSanitizer:" in diagnostic:
        raise RuntimeError("unexpected boundary probe sanitizer category")
    selected = [r for r in records if r.get("symbol") == contract["symbol"]]
    if len(selected) != 1:
        raise RuntimeError("boundary probe symbol is not uniquely selected")
    record = selected[0]
    # UBSan reports a source location directly. ASan's selected caller may be
    # inlined into its intrinsic shim, so retain all exact-executable frames.
    method = "diagnostic_source_location"
    location_text = "\n".join(line for line in diagnostic.splitlines() if "runtime error:" in line)
    if contract["diagnostic"] == "asan_heap_buffer_overflow":
        frames = re.findall(r"(?m)^\s*#\d+\s+(0x[0-9a-fA-F]+)\b", diagnostic)
        if not frames:
            raise RuntimeError("boundary probe has no stack frames")
        # Some local sandboxes lack /proc/self/exe and ASan cannot symbolize.
        # Non-PIE addresses are resolved against this exact compiled artifact.
        resolved = checked(["addr2line", "-f", "-C", "-i", "-e", str(executable), *frames])
        location_text = "\n".join(location for function, location in zip(resolved.splitlines()[0::2], resolved.splitlines()[1::2]) if function.split("(", 1)[0] == contract["symbol"])
        method = "exact_executable_addr2line"
        if not location_text:
            raise RuntimeError("boundary probe diagnostic has no selected caller")
    locations = re.findall(re.escape(str(translation)) + r":(\d+)\b", location_text)
    if not any(int(record["translation_start_line"]) <= int(line) <= int(record["translation_end_line"]) for line in locations):
        raise RuntimeError("boundary probe diagnostic is outside selected body")
    return {"status": "known_defect_reproduced", "contract": contract, "returncode": probe.returncode, "stdout": probe.stdout, "stderr": diagnostic, "site_resolution": method, "resolved_locations": location_text if method == "exact_executable_addr2line" else None}


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
    if mode == "avx512" and (suite.name == "baseline" or any(s.startswith("softmax_kernel") for s in definitions.get(SOURCE, []))):
        source = (PORT / SOURCE).read_bytes().decode()
        constants = re.findall(r"(?ms)^const __m512\s+(?:LOG2E_VEC|MAGIC_FLOAT_BIAS|ONE_VEC|LN2_PART_VEC|EXP_POLY_COEFFS)\b.*?;", source)
        if len(constants) != 5:
            raise ValueError("native polynomial constants missing/ambiguous")
        constant_block = "\n".join(constants)
        pieces.append(constant_block)
        records.append({"symbol": "polynomial_constants", "sha256": hashlib.sha256(constant_block.encode()).hexdigest()})
        if "avx512_exp_ps" not in definitions.setdefault(SOURCE, []):
            definitions[SOURCE].insert(0, "avx512_exp_ps")
    if mode == "avx512" and suite.name == "lsh_retrieval":
        definitions.setdefault("library/lsh/lsh.cc", []).insert(0, "fast_memcpy_avx512")
    for relative, symbols in definitions.items():
        source_file = PORT / relative
        source = source_file.read_bytes().decode()
        sources[str(source_file.relative_to(ROOT))] = digest(source_file)
        for symbol in symbols:
            body, line = extract_definition(source, symbol)
            translation_line = ("\n".join(pieces) + "\n").count("\n") + 1
            pieces.append(body)
            records.append({"source": relative, "symbol": symbol, "line": line, "translation_start_line": translation_line, "translation_end_line": translation_line + body.count("\n"), "sha256": hashlib.sha256(body.encode()).hexdigest()})
    pieces.append('#include ' + json.dumps(str(suite.test_file)) + '\n')
    translation = directory / "selected_bodies.cpp"
    translation.write_text("\n".join(pieces))
    executable = directory / "oracle"
    command = [cxx, "-std=c++17", "-O1", "-fno-tree-vectorize", "-fno-tree-slp-vectorize"]
    if sanitize:
        command += ["-fsanitize=address,undefined", "-g", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-no-pie"]
    if mode == "avx512":
        command += ["-mavx512f", "-mfma", "-DMAGIC_NATIVE_AVX512=1"]
        if suite.name == "lsh_retrieval":
            command += ["-DMAGICPIG_NATIVE_LSH_COPY=1"]
    if mode == "bf16":
        command += ["-mavx512f", "-mavx512bw", "-mavx512bf16"]
    command += [str(translation), "-o", str(executable)]
    checked(command)
    env = dict(os.environ)
    if disable_leaks:
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":detect_leaks=0"
    output = checked([str(executable)], env=env)
    print(output, end="" if output.endswith("\n") else "\n")
    defect_probe = None
    if suite.name == "lsh_retrieval":
        probe = subprocess.run([str(executable), "--fastfill-probe"], capture_output=True, text=True, env=env)
        leak_status = validate_fastfill_probe(probe, sanitize, disable_leaks)
        defect_probe = {"status": "known_defect_reproduced", "leak_status": leak_status, "returncode": probe.returncode, "stdout": probe.stdout, "stderr": probe.stderr}
        print(f"MAGICPIG_FASTFILL_CHARACTERIZED mode={mode} leak_status={leak_status}")
    boundary_probes = []
    if sanitize and mode == "portable":
        for contract in suite.probes:
            probe = subprocess.run([str(executable), contract["argument"]], capture_output=True, text=True, env=env)
            boundary_probes.append(validate_boundary_probe(probe, contract, records, translation, executable))
        if boundary_probes:
            print(f"MAGICPIG_BOUNDARIES_CHARACTERIZED suite={suite.name} count={len(boundary_probes)}")
    headers = {str(p.relative_to(ROOT)): digest(p) for p in sorted(PORT.rglob("*.h"))}
    evidence = {"suite": suite.name, "mode": mode, "status": "pass", "sanitize": sanitize, "leak_check_explicitly_disabled": disable_leaks, "compiler_command": command, "translation_sha256": digest(translation), "source_sha256": sources, "recorded_native_header_sha256": headers, "definitions": records, "fixture": str(suite.test_file.relative_to(ROOT)), "fixture_sha256": digest(suite.test_file), "descriptor_sha256": digest(suite.descriptor), "adapter_sha256": {str(p.relative_to(ROOT)): digest(p) for p in adapters}, "stdout": output, "gpu_validated": False, "torch_extension_validated": False}
    evidence["defect_probe"] = defect_probe
    evidence["boundary_probes"] = boundary_probes
    evidence["boundary_probe_execution"] = "portable_sanitized" if sanitize and mode == "portable" else "not_checked_in_this_mode"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def validate_python_evidence(suite: dict[str, object], data: dict[str, object]) -> None:
    source = (ROOT / suite["source_file"]).read_text()
    if data.get("source") != suite["source_file"] or data.get("gpu_execution") is not False or data.get("torch_runtime") is not False or data.get("name") != suite["name"] or data.get("status") != "passed" or data.get("source_sha256") != digest(ROOT / suite["source_file"]) or set(data.get("selected_blocks", {})) != set(suite["blocks"]):
        raise ValueError("Python selected-block evidence does not match enrolled source")
    statements = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.stmt)]
    for block in data["selected_blocks"].values():
        start, end = block.get("start_line"), block.get("end_line")
        if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start <= end <= len(source.splitlines()):
            raise ValueError("invalid Python selected-block line range")
        nodes = sorted((n for n in statements if start <= n.lineno and n.end_lineno <= end), key=lambda n: n.lineno)
        text = "\n".join(ast.get_source_segment(source, n) or "" for n in nodes)
        if not nodes or hashlib.sha256(text.encode()).hexdigest() != block.get("sha256"):
            raise ValueError("Python selected-block hash mismatch")


def run_python(suite: dict[str, object], build: Path) -> dict[str, object]:
    directory = build / "portable_python" / suite["name"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "selected_blocks.json"
    path.unlink(missing_ok=True)
    fixture = fixture_path(suite["test_file"], ".py")
    command = [os.sys.executable, str(fixture), "--root", str(ROOT), "--evidence", str(path)]
    output = checked(command)
    data = json.loads(path.read_text())
    validate_python_evidence(suite, data)
    evidence = {"suite": suite["name"], "mode": "portable_python", "status": "pass", "sanitizer": "not_applicable_python_adapter", "fixture_sha256": digest(fixture), "descriptor_sha256": digest(suite["descriptor"]), "command": command, "selected_body_evidence": data, "stdout": output, "gpu_validated": False, "torch_extension_validated": False}
    (directory / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(output, end="" if output.endswith("\n") else "\n")
    return evidence


def run(build: Path, cxx: str, sanitize: bool, native: bool, suites_requested: list[str], disable_leaks: bool) -> None:
    build.mkdir(parents=True, exist_ok=True)
    summary_path = build / "magicpig_cpu_evidence.json"
    summary_path.unlink(missing_ok=True)
    for old_evidence in build.glob("*/*/evidence.json"):
        old_evidence.unlink(missing_ok=True)
    suites = load_suites()
    python_suites = load_python_suites()
    if set(suites) & set(python_suites):
        raise ValueError("duplicate C++/Python suite name")
    bf16 = load_bf16_suite()
    check_enrollment(suites, bf16, python_suites)
    if len(set(suites_requested)) != len(suites_requested) or set(suites_requested) - set(suites) - set(python_suites):
        raise ValueError("unknown or duplicate requested suite")
    selected = [suites[n] for n in suites_requested if n in suites] if suites_requested else list(suites.values())
    selected_python = [python_suites[n] for n in suites_requested if n in python_suites] if suites_requested else list(python_suites.values())
    # Preserve the independent immutable-source gate before any numerical test.
    checked([os.sys.executable, str(ROOT / "tools/check_magicpig_sources.py")])
    cap = capabilities(cxx, build, native)
    results = [compile_run(s, "portable", build, cxx, sanitize, disable_leaks) for s in selected]
    results.extend(run_python(s, build) for s in selected_python)
    for mode in ("avx512", "bf16"):
        decision = cap["modes"][mode]
        if not decision["eligible"]:
            print(f"MAGICPIG_NATIVE_SKIP mode={mode} reason={decision['reason']}")
        elif mode == "avx512":
            native_suites = [s for s in selected if s.name == "baseline" or s.native_avx512]
            if not native_suites:
                print("MAGICPIG_NATIVE_SKIP mode=avx512 reason=no native suite selected")
            results.extend(compile_run(s, mode, build, cxx, sanitize, disable_leaks) for s in native_suites)
        else:
            results.append(compile_run(bf16, mode, build, cxx, sanitize, disable_leaks))
    summary = {"status": "pass", "driver_sha256": digest(Path(__file__)), "source_pin_sha256": digest(ROOT / "papers/magicpig-source-pin.json"), "fixture_seed": "fixed literals, no RNG", "capabilities": cap, "sanitize": sanitize, "results": results, "gpu_validated": False, "torch_extension_validated": False}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"MAGICPIG_CPU_PASS portable_suites={len(selected) + len(selected_python)} executed_modes={','.join(sorted({r['mode'] for r in results}))} sanitize={sanitize}")


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
