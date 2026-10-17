#!/usr/bin/env python3
"""Check the coherent immutable SBSmyrf subset and its distinct license notices.

Reuses the initial source gate's strict JSON/path/blob/full-Git-tree primitives.
No network, build, dependency installation, numerical or GPU execution occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_scatterbrain_sources import (  # noqa: E402
    LINEAGE, NOTICE, blob_record, contained, count, digest, git_hash, identity,
    load_json, mapping, paths, require, validate_lineage,
)

PIN_PATH = "papers/scatterbrain-sbsmyrf-source-pin.json"
ARCHIVE = "third_party/scatterbrain-sbsmyrf"
NOTICE_ROOT = "papers/notices/scatterbrain-sbsmyrf"
COMMIT = "74cc6b14347b4edcf5c49098e80a9ef47c0b21bf"
TREE = "1821d435ef810ce88038ef81965c65cd0ac8c6bc"
CORE = {
    "src/models/attention/" + name for name in (
        "batching_utils.py", "feature_maps_sb.py", "hash_utils.py", "mask_utils.py",
        "projection_utils.py", "reformer_attention.py", "sbsmyrf_attention.py", "scatterbrain_utils.py")
} | {"src/models/modules/masking.py", "src/ops/permutation.py", "src/utils/padding.py"}
PACKAGES = {"src/__init__.py", "src/models/__init__.py", "src/models/modules/__init__.py", "src/utils/__init__.py"}
PIN_FIELDS = LINEAGE | {"schema_version", "scope", "archive_root", "selected_blobs", "licenses",
    "dependency_source_files", "component_license", "preserved_notices", "derivation_evidence", "scope_notes"}
# Required origin/notice bindings keep a coordinated removal from silently dropping
# one copyright notice while leaving another nonempty notice declaration behind.
NOTICE_BINDINGS = {
    "Fly-Apache-LICENSE": ("HazyResearch/scatterbrain", COMMIT, "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", "Apache-2.0", CORE | PACKAGES),
    "SMYRF-GPL-3.0-LICENSE": ("giannisdaras/smyrf", "bbbc82e0126233da07841275daa238d447c37f66", "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7", "GPL-3.0", {"src/models/attention/hash_utils.py", "src/models/attention/sbsmyrf_attention.py"}),
    "Reformer-MIT-LICENSE": ("lucidrains/reformer-pytorch", "2cbc36bb280c0a6de46d838baad11a533a015fc3", "145245790672b7d569096d08199b38a35d6f4d36", "MIT", {"src/models/attention/reformer_attention.py", "src/utils/padding.py"}),
    "Performer-MIT-LICENSE": ("lucidrains/performer-pytorch", "0a4a520e9195972f0a3fa1a577c78cfdfccf7338", "cad43213602892aadc2290cca7afa74cc9f6af64", "MIT", {"src/models/attention/scatterbrain_utils.py"}),
    "Idiap-MIT-LICENSE": ("idiap/fast-transformers", "f22c13716fc748bb21a7b226ada7f7b5f87f867f", "2c828d5ff01b3971442ba1a5ea47b1119b52c186", "MIT", {"src/models/attention/feature_maps_sb.py", "src/models/modules/masking.py"}),
}
DERIVATION_FIELDS = {"repository", "commit", "source_path", "sha", "sha256", "size", "source_url",
    "archive_path", "archive_start_line", "source_start_line", "line_count", "matched_lines_sha256", "license_notice"}


def validate_archive(root: Path, pin: dict[str, object], selected: dict[str, object]) -> None:
    archive = contained(root, pin["archive_root"])
    require(archive.is_dir(), "missing SBSmyrf archive")
    files, directories = set(), set()
    for path in archive.rglob("*"):
        require(not path.is_symlink(), "SBSmyrf archive symlink forbidden")
        name = path.relative_to(archive).as_posix()
        if path.is_dir():
            directories.add(name)
        else:
            require(stat.S_ISREG(path.stat().st_mode), "SBSmyrf archive special file forbidden")
            files.add(name)
    expected_dirs = {str(parent) for name in selected for parent in PurePosixPath(name).parents if str(parent) != "."}
    require(files == set(selected) and directories == expected_dirs, "SBSmyrf archive file/directory set differs")
    for name, value in selected.items():
        record = blob_record(value, "selected SBSmyrf blob", sha256=True)
        path = contained(archive, name)
        payload = path.read_bytes()
        require(len(payload) == record["size"] and git_hash("blob", payload) == record["sha"] and digest(path) == record["sha256"], "SBSmyrf archive blob identity differs")
        require(bool(path.stat().st_mode & 0o111) == (record["mode"] == "100755"), "SBSmyrf executable mode differs")
    for name in PACKAGES:
        require(contained(archive, name).read_bytes() == b"", "nonempty SBSmyrf package initializer")


def validate_notices(root: Path, pin: dict[str, object]) -> int:
    notices = mapping(pin["preserved_notices"], "SBSmyrf notices")
    require(set(notices) == {NOTICE_ROOT + "/" + name for name in NOTICE_BINDINGS}, "required SBSmyrf notice set differs")
    for name, value in notices.items():
        notice = mapping(value, "SBSmyrf notice", NOTICE)
        path = contained(root, name)
        repository, commit, sha, license_name, applies = NOTICE_BINDINGS[path.name]
        require(notice["repository"] == "https://github.com/" + repository and notice["commit"] == commit
                and notice["source_path"] == "LICENSE" and notice["sha"] == sha and notice["license"] == license_name,
                "SBSmyrf notice origin/license differs")
        require(set(paths(notice["applies_to"])) == applies, "SBSmyrf notice application differs")
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "SBSmyrf notice must be a regular file")
        payload = path.read_bytes()
        require(len(payload) == count(notice["size"]) and git_hash("blob", payload) == sha
                and digest(path) == identity(notice["sha256"], 64), "SBSmyrf notice identity differs")
    directory = contained(root, NOTICE_ROOT)
    require(directory.is_dir(), "missing SBSmyrf notice directory")
    actual = set()
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "SBSmyrf notice symlink forbidden")
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "unexpected SBSmyrf notice directory/special file")
        actual.add(path.relative_to(root).as_posix())
    require(actual == set(notices), "SBSmyrf notice file set differs")
    return len(notices)


def validate_derivation(root: Path, pin: dict[str, object]) -> int:
    evidence = pin["derivation_evidence"]
    require(isinstance(evidence, list) and len(evidence) == 1, "missing/invalid SBSmyrf derivation evidence")
    record = mapping(cast(list[object], evidence)[0], "derivation evidence", DERIVATION_FIELDS)
    require(record["repository"] == "https://github.com/giannisdaras/smyrf"
            and record["commit"] == "bbbc82e0126233da07841275daa238d447c37f66"
            and record["source_path"] == "smyrf/torch/utils.py"
            and record["sha"] == "89863929a031c993899cd855c1e2e82255d22665"
            and record["sha256"] == "b575ec24a06d34150be95d2fe25018eb0618d55fd4193bc1c3c42cf751fec38d"
            and count(record["size"]) == 9790, "historical SMYRF origin differs")
    require(record["source_url"] == str(record["repository"]) + "/blob/" + str(record["commit"]) + "/" + str(record["source_path"]), "historical SMYRF pinned URL differs")
    require(record["archive_path"] == "src/models/attention/hash_utils.py"
            and count(record["archive_start_line"]) == 2 and count(record["source_start_line"]) == 1
            and count(record["line_count"]) == 85
            and record["license_notice"] == NOTICE_ROOT + "/SMYRF-GPL-3.0-LICENSE", "SMYRF overlap/license binding differs")
    archive = contained(root, pin["archive_root"])
    source = contained(archive, record["archive_path"]).read_bytes().splitlines(keepends=True)
    # This validates retained archive text against the recorded donor-overlap
    # digest. Donor bytes remain separately pinned metadata, not an offline fetch.
    matched = b"".join(source[1:86])
    require(len(source) >= 86, "SMYRF overlap slice missing")
    require(hashlib.sha256(matched).hexdigest() == identity(record["matched_lines_sha256"], 64), "SMYRF overlap slice digest differs")
    for name, donor in (("hash_utils.py", "utils.py"), ("sbsmyrf_attention.py", "attn.py")):
        first = contained(archive, "src/models/attention/" + name).read_text().splitlines()[0]
        require(first == "# Adapted from https://github.com/giannisdaras/smyrf/blob/master/smyrf/torch/" + donor, "missing original SMYRF attribution")
    return 1


def validate(root: Path = ROOT) -> dict[str, int]:
    root = root.resolve()
    pin_path = contained(root, PIN_PATH)
    digest(pin_path)  # Reject nonregular pin files before attempting to read them.
    pin = mapping(load_json(pin_path), "SBSmyrf pin", PIN_FIELDS)
    require(type(pin["schema_version"]) is int and pin["schema_version"] == 1
            and pin["scope"] == "selected-sbsmyrf-source-subset", "unsupported SBSmyrf schema/scope")
    require(pin["repository"] == "https://github.com/HazyResearch/scatterbrain" and pin["commit"] == COMMIT
            and pin["tree"] == TREE and pin["archive_root"] == ARCHIVE, "coherent SBSmyrf lineage/archive differs")
    require(pin["component_license"] == "GPL-3.0", "derived SBSmyrf component license differs")
    selected = mapping(pin["selected_blobs"], "selected SBSmyrf blobs")
    require(set(selected) == CORE | PACKAGES | {"LICENSE", "README.md"}, "SBSmyrf closure differs")
    require(paths(pin["licenses"]) == ["LICENSE"] and set(paths(pin["dependency_source_files"])) == CORE | PACKAGES,
            "SBSmyrf license/dependency declaration differs")
    require(isinstance(pin["scope_notes"], list) and bool(pin["scope_notes"])
            and all(isinstance(note, str) and bool(note) for note in pin["scope_notes"]), "invalid SBSmyrf scope notes")
    _, lineage = validate_lineage(root, pin, selected)
    validate_archive(root, pin, selected)
    notice_count = validate_notices(root, pin)
    derivation_count = validate_derivation(root, pin)
    return {"selected_blobs": len(selected), "project_source_files": len(CORE), "empty_package_initializers": len(PACKAGES),
            "upstream_blobs": lineage["blobs"], "upstream_tree_objects": lineage["trees"],
            "preserved_notices": notice_count, "derivation_records": derivation_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-dir", type=Path)
    args = parser.parse_args()
    evidence_path = args.emit_dir / "scatterbrain_sbsmyrf_source_evidence.json" if args.emit_dir else None
    if evidence_path:
        evidence_path.unlink(missing_ok=True)
    counts = validate()
    pin = load_json(ROOT / PIN_PATH)
    result = {"status": "source_identity_pass", **counts, "upstream_commit": COMMIT, "upstream_tree": TREE,
              "pin_sha256": digest(ROOT / PIN_PATH), "inventory_sha256": pin["inventory_sha256"],
              "component_license": "GPL-3.0", "build_validated": False, "dependencies_installed": False,
              "numerical_runtime_validated": False, "paper_claims_validated": False, "gpu_validated": False,
              "donor_source_validation": "pinned_origin_metadata_and_retained_archive_slice_only",
              "checker_sha256": digest(Path(__file__)), "shared_checker_sha256": digest(ROOT / "tools/check_scatterbrain_sources.py")}
    if evidence_path:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(result, indent=2)+"\n")
    print("SCATTERBRAIN_SBSMYRF_SOURCE_PASS " + " ".join(f"{key}={value}" for key, value in counts.items()) + " component_license=GPL-3.0 numerical_runtime_validated=False")


if __name__ == "__main__":
    main()
