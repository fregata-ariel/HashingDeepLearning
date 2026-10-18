"""Verify and prepare one pinned official SIMDe snapshot for CPU emulation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
from types import ModuleType
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "ports/magicpig/compat/simde-source.json"


def source_pin() -> dict[str, str]:
    """Validate the fixed upstream URL, complete Git/SHA256 identities and license."""
    raw: object = json.loads(PIN.read_text())
    required = {"upstream_ref", "commit", "archive_url", "archive_sha256", "license", "license_sha256"}
    if not isinstance(raw, dict) or set(raw) != required or any(not isinstance(k, str) or not isinstance(v, str) for k, v in raw.items()):
        raise ValueError("invalid SIMDe source pin")
    pin = {str(k): str(v) for k, v in raw.items()}
    if not re.fullmatch(r"[0-9a-f]{40}", pin["commit"]) or any(not re.fullmatch(r"[0-9a-f]{64}", pin[field]) for field in ("archive_sha256", "license_sha256")):
        raise ValueError("SIMDe requires full commit/SHA256 identities")
    if pin["archive_url"] != "https://codeload.github.com/simd-everywhere/simde/tar.gz/" + pin["commit"] or pin["license"] != "MIT":
        raise ValueError("SIMDe requires the pinned official HTTPS archive/MIT license")
    if hashlib.sha256((PIN.parent / "SIMDE_LICENSE").read_bytes()).hexdigest() != pin["license_sha256"]:
        raise ValueError("maintained SIMDe license differs from source pin")
    return pin


def prepare_simde(cache: Path) -> Path:
    """Verify archive and every cached header before compilation; never execute downloads.

    Only regular files under upstream simde/ plus COPYING are extracted by
    explicit writes; links, absolute paths and parent traversal are excluded.
    Existing modified/extra cache files reject rather than silently validating.
    """
    pin = source_pin()
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (pin["commit"] + ".tar.gz")
    if archive.is_symlink():
        raise ValueError("SIMDe archive must not be a symbolic link")
    if not archive.exists():
        with urlopen(pin["archive_url"], timeout=60) as response:
            payload: bytes = response.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != pin["archive_sha256"]:
            raise ValueError("SIMDe download SHA256/size differs from source pin")
        archive.write_bytes(payload)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != pin["archive_sha256"]:
        raise ValueError("cached SIMDe archive SHA256 differs from source pin")
    source = cache / ("simde-" + pin["commit"])
    if source.is_symlink():
        raise ValueError("SIMDe cache root must not be a symbolic link")
    source.mkdir(exist_ok=True)
    expected: set[str] = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != "simde-" + pin["commit"] or ".." in parts or member.name.startswith("/"):
                raise ValueError("unsafe SIMDe archive path")
            relative = PurePosixPath(*parts[1:])
            if not member.isfile() or not parts[1:] or (parts[1] != "simde" and str(relative) != "COPYING"):
                continue
            destination = source / str(relative)
            if destination.is_symlink() or not destination.resolve().is_relative_to(source.resolve()):
                raise ValueError("unsafe SIMDe cache path")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("missing SIMDe archive file")
            payload = stream.read()
            if destination.exists():
                if destination.read_bytes() != payload:
                    raise ValueError("modified SIMDe cache file: " + str(relative))
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
            expected.add(str(relative))
    actual = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    if actual != expected or "simde/x86/avx512.h" not in expected:
        raise ValueError("unexpected or missing SIMDe cache files")
    if hashlib.sha256((source / "COPYING").read_bytes()).hexdigest() != pin["license_sha256"]:
        raise ValueError("upstream SIMDe license differs from source pin")
    return source


def source_record(cache: Path) -> dict[str, object]:
    """Record the actual verified include directory and complete extracted tree hash."""
    source = prepare_simde(cache)
    digest = hashlib.sha256()
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(source)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {**source_pin(), "include_root": str(source), "source_tree_sha256": digest.hexdigest(), "avx512_header_sha256": hashlib.sha256((source / "simde/x86/avx512.h").read_bytes()).hexdigest(), "pin_sha256": hashlib.sha256(PIN.read_bytes()).hexdigest(), "bootstrap_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def verify_emulated_binary(module: ModuleType, conversions: bool) -> dict[str, object]:
    """Reject AVX machine instructions or direct FBGEMM AVX512 imports in our module.

    Torch/FBGEMM shared libraries retain their own runtime CPU dispatch. This
    audit applies to the newly compiled extension, not the whole Torch wheel.
    """
    binary = Path(str(module.__file__))
    disassembly = subprocess.check_output(["objdump", "-d", "--no-show-raw-insn", "-Mintel", str(binary)], text=True)
    if re.search(r"\b(?:ymm|zmm)[0-9]+\b|\bk[0-7]\b", disassembly) or re.search(r"^\s*[0-9a-f]+:\s+v[a-z0-9]+\b", disassembly, re.MULTILINE):
        raise ValueError("emulated extension contains native AVX instructions")
    imports = subprocess.check_output(["nm", "-D", "--undefined-only", str(binary)], text=True)
    if any(name + "_avx512" in imports for name in ("FloatToBfloat16", "Bfloat16ToFloat")):
        raise ValueError("emulated extension imports unconditional FBGEMM AVX512 conversion")
    if conversions and any(name + "_simd" not in imports for name in ("FloatToBfloat16", "Bfloat16ToFloat")):
        raise ValueError("emulated attention must use real FBGEMM runtime conversion dispatch")
    return {"binary": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "native_avx_instructions": False, "unconditional_fbgemm_avx512_imports": False, "real_fbgemm_runtime_conversion_imports": conversions}
