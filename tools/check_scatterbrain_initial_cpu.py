#!/usr/bin/env python3
"""Build the pinned CPU dependency and verify the maintained initial Scatterbrain port.

This required suite fails on missing prerequisites/build/link/import/numerical
errors. It never substitutes Torch local products or converts failures to skips.
Only the build/package namespace is adapted; archived dependency bytes are exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = "third_party/scatterbrain-fast-transformers"
PIN = "papers/scatterbrain-native-dependency-pin.json"
SOURCE_NAMES = (
    "src/models/modules/attention/sblocal_attention.py",
    "src/models/modules/attention/feature_maps_sb.py",
    "src/models/modules/attention/projection_utils.py",
    "src/models/modules/attention/scatterbrain_utils.py",
    "src/models/modules/masking.py",
)
ANCHORS = {
    "LICENSE": (
        "2c828d5ff01b3971442ba1a5ea47b1119b52c186",
        "db2b1cdf9bda73650d860da722095b8f3a817be008e94e3522bad2c40979915d",
        1199,
    ),
    "fast_transformers/local_product/local_product_cpu.cpp": (
        "35776250726b940ff161e7f8d17d9403067219fa",
        "70dedc4ab7b367eaea622eb80a53d5caeb3399604d144dcf036cd5bed223c313",
        7356,
    ),
    "fast_transformers/local_product/__init__.py": (
        "56f69fa0696492730464e19a9ebe0f7c86db2131",
        "6ed2d8f4b7ae84c4f9852d9096db0ab6c3bba4644cc2bd4985a0b31e4ab6a504",
        3053,
    ),
}
PROFILE = {
    "python": "3.12",
    "torch": "2.14.1+cpu",
    "einops": "0.8.1",
    "setuptools": "84.0.0",
    "compiler": "GNU14",
    "cxx_standard": "C++20",
}
SETUP = """from setuptools import setup
from torch.utils.cpp_extension import CppExtension,BuildExtension
setup(name="scatterbrain_local_product_cpu",ext_modules=[CppExtension(name="fast_transformers.local_product.local_product_cpu",sources=["local_product_cpu.cpp"],extra_compile_args=["-O1","-g0","-fopenmp","-fno-fast-math","-std=c++20"],extra_link_args=["-fopenmp"])],cmdclass={"build_ext":BuildExtension.with_options(use_ninja=False)})
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(pairs):
    data = {}
    for key, value in pairs:
        require(key not in data, "duplicate pin JSON key: " + key)
        data[key] = value
    return data


def validate_dependency(root=ROOT):
    path = root / PIN
    require(not path.is_symlink(), "dependency pin symlink forbidden")
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), "dependency pin parent symlink forbidden")
    data = json.loads(path.read_text(), object_pairs_hook=unique)
    require(
        set(data)
        == {
            "schema_version",
            "repository",
            "commit",
            "archive_root",
            "scope",
            "license",
            "selected_blobs",
            "runtime_profile",
            "scope_notes",
        },
        "unknown/missing dependency pin fields",
    )
    require(
        type(data["schema_version"]) is int
        and data["schema_version"] == 1
        and data["repository"] == "https://github.com/idiap/fast-transformers"
        and data["commit"] == "f22c13716fc748bb21a7b226ada7f7b5f87f867f",
        "dependency lineage differs",
    )
    require(
        data["archive_root"] == ARCHIVE
        and data["scope"] == "selected-local-product-cpu-dependency"
        and data["license"] == "MIT",
        "dependency scope/license differs",
    )
    require(data["runtime_profile"] == PROFILE, "required runtime profile differs")
    require(
        isinstance(data["scope_notes"], list)
        and all(isinstance(x, str) for x in data["scope_notes"]),
        "invalid dependency notes",
    )
    expected = {
        name: {"sha": sha, "sha256": sha256, "size": size, "mode": "100644"}
        for name, (sha, sha256, size) in ANCHORS.items()
    }
    require(data["selected_blobs"] == expected, "dependency blob anchors differ")
    require(
        all(type(record["size"]) is int for record in data["selected_blobs"].values()),
        "dependency blob size must be an integer",
    )
    archive = root / ARCHIVE
    require(
        archive.is_dir() and not archive.is_symlink(),
        "missing/symlink dependency archive",
    )
    for parent in archive.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), "dependency archive parent symlink")
    expected_dirs = {
        str(p) for name in ANCHORS for p in Path(name).parents if str(p) != "."
    }
    actual_files = set()
    actual_dirs = set()
    for item in archive.rglob("*"):
        require(not item.is_symlink(), "dependency archive symlink forbidden")
        name = item.relative_to(archive).as_posix()
        if item.is_dir():
            actual_dirs.add(name)
        else:
            require(
                stat.S_ISREG(item.stat().st_mode), "dependency special file forbidden"
            )
            actual_files.add(name)
    require(
        actual_files == set(ANCHORS) and actual_dirs == expected_dirs,
        "dependency archive file/directory set differs",
    )
    for name, (sha, sha256, size) in ANCHORS.items():
        item = archive / name
        blob = item.read_bytes()
        require(
            len(blob) == size
            and hashlib.sha1(b"blob " + str(size).encode() + b"\0" + blob).hexdigest()
            == sha
            and digest(item) == sha256,
            "dependency blob identity differs: " + name,
        )
        require(
            not item.stat().st_mode & 0o111,
            "dependency executable mode differs: " + name,
        )
    return data


def test_dependency_contracts():
    """Reject altered bytes, coordinated pin edits, inventory and link changes."""

    def body(root):
        item = root / ARCHIVE / "LICENSE"
        item.write_bytes(item.read_bytes() + b"altered")

    def coordinated(root):
        body(root)
        item = root / ARCHIVE / "LICENSE"
        value = json.loads((root / PIN).read_text())
        value["selected_blobs"]["LICENSE"] = {
            "sha": hashlib.sha1(
                b"blob " + str(item.stat().st_size).encode() + b"\0" + item.read_bytes()
            ).hexdigest(),
            "sha256": digest(item),
            "size": item.stat().st_size,
            "mode": "100644",
        }
        (root / PIN).write_text(json.dumps(value))

    def extra_file(root):
        (root / ARCHIVE / "unselected.cpp").write_text("unselected")

    def extra_dir(root):
        (root / ARCHIVE / "unselected").mkdir()

    def executable(root):
        (root / ARCHIVE / "LICENSE").chmod(0o755)

    def file_link(root):
        item = root / ARCHIVE / "LICENSE"
        item.unlink()
        item.symlink_to(ROOT / ARCHIVE / "LICENSE")

    def archive_link(root):
        item = root / ARCHIVE
        shutil.rmtree(item)
        item.symlink_to(ROOT / ARCHIVE, target_is_directory=True)

    def parent_link(root):
        shutil.rmtree(root / "third_party")
        (root / "third_party").symlink_to(
            ROOT / "third_party", target_is_directory=True
        )

    def duplicate(root):
        item = root / PIN
        item.write_text(item.read_text().replace("{", '{"schema_version":1,', 1))

    def unknown(root):
        item = root / PIN
        value = json.loads(item.read_text())
        value["unknown"] = True
        item.write_text(json.dumps(value))

    def profile(root):
        item = root / PIN
        value = json.loads(item.read_text())
        value["runtime_profile"]["torch"] = "unverified"
        item.write_text(json.dumps(value))

    def bool_schema(root):
        item = root / PIN
        value = json.loads(item.read_text())
        value["schema_version"] = True
        item.write_text(json.dumps(value))

    def float_schema(root):
        item = root / PIN
        value = json.loads(item.read_text())
        value["schema_version"] = 1.0
        item.write_text(json.dumps(value))

    def float_size(root):
        item = root / PIN
        value = json.loads(item.read_text())
        value["selected_blobs"]["LICENSE"]["size"] = 1199.0
        item.write_text(json.dumps(value))

    mutations = (
        body,
        coordinated,
        extra_file,
        extra_dir,
        executable,
        file_link,
        archive_link,
        parent_link,
        duplicate,
        unknown,
        profile,
        bool_schema,
        float_schema,
        float_size,
    )
    for mutation in mutations:
        with tempfile.TemporaryDirectory(
            prefix="scatterbrain-dependency-contract-"
        ) as location:
            root = Path(location)
            (root / PIN).parent.mkdir(parents=True)
            shutil.copyfile(ROOT / PIN, root / PIN)
            shutil.copytree(ROOT / ARCHIVE, root / ARCHIVE)
            validate_dependency(root)
            mutation(root)
            try:
                validate_dependency(root)
            except (ValueError, OSError):
                continue
            raise AssertionError("dependency mutation accepted: " + mutation.__name__)
    print(
        "SCATTERBRAIN_NATIVE_DEPENDENCY_CONTRACTS_PASS rejected=" + str(len(mutations))
    )


def checked(command, *, env=None, log=None, cwd=None):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if log:
        log.write_text(result.stdout + "\nSTDERR\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\n{result.stdout[-12000:]}\n{result.stderr[-12000:]}"
        )
    return result.stdout


def runtime(cxx):
    import torch

    require(sys.version_info[:2] == (3, 12), "Python3.12 is required")
    require(
        torch.__version__ == PROFILE["torch"] and torch.version.cuda is None,
        "pinned CPU Torch is required",
    )
    for name in ("einops", "setuptools"):
        require(version(name) == PROFILE[name], "pinned " + name + " is required")
    compiler = shutil.which(cxx)
    require(bool(compiler), "required GNU14 compiler missing")
    compiler = str(Path(compiler).resolve())
    macros = checked([compiler, "-dM", "-E", "-x", "c++", "/dev/null"])
    require(
        re.search(r"^#define __GNUC__ 14$", macros, re.M) is not None
        and "__clang__" not in macros,
        "GNU14 compiler is required",
    )
    cc = shutil.which("gcc-14")
    require(bool(cc), "GNU14 C compiler missing")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_CXX11_ABI": torch._C._GLIBCXX_USE_CXX11_ABI,
        "torch_config": torch.__config__.show(),
        "einops": version("einops"),
        "setuptools": version("setuptools"),
        "compiler": compiler,
        "compiler_version": checked([compiler, "--version"]).splitlines()[0],
        "compiler_macros_sha256": hashlib.sha256(macros.encode()).hexdigest(),
        "cc": str(Path(cc).resolve()),
    }


def _run(build, cxx):
    build = build.resolve()
    for protected in (
        "third_party",
        "ports",
        "papers",
        "tests",
        "tools",
        ".git",
        ".agents",
        ".codex",
    ):
        require(
            not build.is_relative_to(ROOT / protected),
            "build directory overlaps protected sources",
        )
    build.mkdir(parents=True, exist_ok=True)
    summary = build / "scatterbrain_initial_cpu_evidence.json"
    summary.unlink(missing_ok=True)
    fixture_evidence = build / "fixture_evidence.json"
    fixture_evidence.unlink(missing_ok=True)
    dependency = validate_dependency()
    checked(
        [sys.executable, "-B", str(ROOT / "tools/check_scatterbrain_sources.py")],
        log=build / "source-gate.log",
    )
    profile = runtime(cxx)
    port = ROOT / "ports/scatterbrain"
    for name in SOURCE_NAMES:
        require(
            (port / name).is_file() and not (port / name).is_symlink(),
            "required maintained port missing: " + name,
        )
    port_hashes = {}
    for item in sorted(port.rglob("*.py")):
        require(not item.is_symlink(), "maintained port symlink forbidden")
        port_hashes[item.relative_to(port).as_posix()] = digest(item)
    stage = Path(tempfile.mkdtemp(prefix="native-build-", dir=build))
    package = stage / "fast_transformers/local_product"
    package.mkdir(parents=True)
    (stage / "fast_transformers/__init__.py").write_text(
        ""
    )  # Explicit namespace-only build helper.
    shutil.copyfile(
        ROOT / ARCHIVE / "fast_transformers/local_product/__init__.py",
        package / "__init__.py",
    )
    shutil.copyfile(
        ROOT / ARCHIVE / "fast_transformers/local_product/local_product_cpu.cpp",
        stage / "local_product_cpu.cpp",
    )
    (stage / "setup_cpu.py").write_text(SETUP)
    env = dict(
        os.environ,
        CC=profile["cc"],
        CXX=profile["compiler"],
        MAX_JOBS="1",
        OMP_NUM_THREADS="2",
        PYTHONDONTWRITEBYTECODE="1",
    )
    commands = []
    durations = {}
    start = time.monotonic()
    command = [
        sys.executable,
        "-B",
        "setup_cpu.py",
        "build_ext",
        "--inplace",
        "--parallel",
        "1",
    ]
    commands.append(command)
    checked(command, env=env, cwd=stage, log=build / "build.log")
    durations["build_seconds"] = time.monotonic() - start
    fixture = ROOT / "tests/traceability/test_scatterbrain_initial_cpu.py"
    start = time.monotonic()
    command = [
        sys.executable,
        "-B",
        str(fixture),
        "--native-root",
        str(stage),
        "--port-root",
        str(port),
        "--evidence",
        str(fixture_evidence),
    ]
    commands.append(command)
    output = checked(command, env=env, cwd=stage, log=build / "fixture.log")
    durations["fixture_seconds"] = time.monotonic() - start
    data = json.loads(fixture_evidence.read_text())
    require(
        data.get("status") == "passed"
        and data.get("native_CPU_executed") is True
        and data.get("maintained_port_executed") is True
        and data.get("substitute_local_product") is False,
        "fixture did not execute required native/maintained path",
    )
    binary = Path(data["extension_path"])
    require(
        binary.resolve().is_relative_to(stage)
        and binary.suffix == ".so"
        and digest(binary) == data["extension_sha256"],
        "native executable identity differs",
    )
    require(
        data["wrapper_sha256"]
        == ANCHORS["fast_transformers/local_product/__init__.py"][1],
        "executed native wrapper identity differs",
    )
    require(
        digest(stage / "local_product_cpu.cpp")
        == ANCHORS["fast_transformers/local_product/local_product_cpu.cpp"][1],
        "built native source identity differs",
    )
    validate_dependency()
    require(data["fixture_sha256"] == digest(fixture), "fixture identity differs")
    require(
        {
            name.replace("/", ".")[:-3]: sha
            for name, sha in port_hashes.items()
            if name in SOURCE_NAMES
        }
        == data["source_sha256"],
        "executed maintained source identities differ",
    )
    require(
        all(digest(port / name) == sha for name, sha in port_hashes.items()),
        "maintained source changed during execution",
    )
    evidence = {
        "status": "pass",
        "required_native_CPU": True,
        "runtime_profile": profile,
        "commands": commands,
        "durations": durations,
        "dependency_pin_sha256": digest(ROOT / PIN),
        "dependency_commit": dependency["commit"],
        "dependency_source_hashes": {
            name: digest(ROOT / ARCHIVE / name) for name in ANCHORS
        },
        "port_source_sha256": port_hashes,
        "initial_source_pin_sha256": digest(
            ROOT / "papers/scatterbrain-source-pin.json"
        ),
        "driver_sha256": digest(Path(__file__)),
        "fixture_sha256": digest(fixture),
        "build_helper_sha256": digest(stage / "setup_cpu.py"),
        "namespace_helper": "generated empty top-level fast_transformers package; original autograd wrapper unchanged",
        "native_binary": str(binary),
        "native_binary_sha256": digest(binary),
        "fixture_evidence": data,
        "gpu_validated": False,
        "memory_sanitizers_executed": False,
        "archive_runtime_claimed": False,
        "build_adaptation": "CPU-only GNU14/C++20 setuptools; original dependency source/wrapper unchanged",
    }
    summary.write_text(json.dumps(evidence, indent=2) + "\n")
    print(output, end="")
    print(
        "SCATTERBRAIN_INITIAL_CPU_PASS required_native=True maintained_port=True gpu=False"
    )


def run(build, cxx):
    build = build.resolve()
    for protected in (
        "third_party",
        "ports",
        "papers",
        "tests",
        "tools",
        ".git",
        ".agents",
        ".codex",
    ):
        require(
            not build.is_relative_to(ROOT / protected),
            "build directory overlaps protected sources",
        )
    try:
        return _run(build, cxx)
    except BaseException:
        for name in ("scatterbrain_initial_cpu_evidence.json", "fixture_evidence.json"):
            (build / name).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--cxx", default="g++-14")
    parser.add_argument("--verify-dependency-only", action="store_true")
    parser.add_argument("--test-dependency-contracts", action="store_true")
    args = parser.parse_args()
    if args.test_dependency_contracts:
        test_dependency_contracts()
        return
    if args.verify_dependency_only:
        validate_dependency()
        print("SCATTERBRAIN_NATIVE_DEPENDENCY_IDENTITY_PASS")
        return
    if args.build_dir:
        run(args.build_dir, args.cxx)
    else:
        with tempfile.TemporaryDirectory(prefix="scatterbrain-initial-cpu-") as d:
            run(Path(d), args.cxx)


if __name__ == "__main__":
    main()
