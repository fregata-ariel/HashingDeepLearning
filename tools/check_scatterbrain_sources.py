#!/usr/bin/env python3
"""Verify immutable Scatterbrain source, complete Git trees and preserved notices.

Source identity is not build, dependency, numerical, theorem or GPU validation.
Unknown JSON fields, duplicate keys, unsafe paths and symlinks fail closed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import stat
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["path", "mode", "type", "sha", "size", "import_status"]
LINEAGE = {"repository", "commit", "tree", "commit_object", "upstream_inventory",
           "inventory_sha256", "upstream_entries", "upstream_blobs"}
PIN = LINEAGE | {"schema_version", "scope", "archive_root", "selected_blobs",
                 "licenses", "dependency_source_files", "references", "preserved_notices", "scope_notes"}
REFERENCE = LINEAGE | {"name", "role", "reference_files"}
NOTICE = {"repository", "commit", "source_path", "sha", "sha256", "size", "license", "applies_to"}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def mapping(value: object, label: str, keys: set[str] | None = None) -> dict[str, object]:
    require(isinstance(value, dict), f"invalid {label} object")
    result = cast(dict[str, object], value)
    require(keys is None or set(result) == keys, f"unknown or missing {label} fields")
    return result


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, object]:
    return mapping(json.loads(path.read_text(), object_pairs_hook=unique_object), "pin")


def relative(value: object) -> str:
    require(isinstance(value, str), "path must be a string")
    value = cast(str, value)
    path = PurePosixPath(value)
    require(bool(value) and bool(path.parts) and not path.is_absolute() and
            str(path) == value and ".." not in path.parts and "\\" not in value and
            not any(ord(c) < 32 or ord(c) == 127 for c in value),
            f"unsafe/noncanonical path: {value!r}")
    return value


def contained(root: Path, value: object) -> Path:
    name = relative(value)
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        require(not current.is_symlink(), f"symlink path traversal: {name}")
    require(current.resolve().is_relative_to(root.resolve()), f"path escapes root: {name}")
    return current


def identity(value: object, length: int = 40) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None,
            "invalid object/digest identity")
    return cast(str, value)


def count(value: object) -> int:
    require(type(value) is int and cast(int, value) >= 0, "invalid nonnegative count")
    return cast(int, value)


def paths(value: object) -> list[str]:
    require(isinstance(value, list), "invalid path list")
    names = [relative(p) for p in cast(list[object], value)]
    require(len(set(names)) == len(names), "duplicate path declaration")
    return names


def git_hash(kind: str, content: bytes) -> str:
    return hashlib.sha1(kind.encode() + b" " + str(len(content)).encode() + b"\0" + content).hexdigest()


def digest(path: Path) -> str:
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), f"not a regular file: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob_record(value: object, label: str, sha256: bool = False) -> dict[str, object]:
    entry = mapping(value, label, {"sha", "mode", "size"} | ({"sha256"} if sha256 else set()))
    identity(entry["sha"])
    require(isinstance(entry["mode"], str) and entry["mode"] in {"100644", "100755"}, "selected object must be a regular Git blob")
    count(entry["size"])
    if sha256:
        identity(entry["sha256"], 64)
    return entry


def validate_lineage(root: Path, meta: dict[str, object], selected: dict[str, object]) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    require(isinstance(meta["repository"], str) and
            re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", cast(str, meta["repository"])) is not None,
            "invalid canonical source repository")
    commit, tree = identity(meta["commit"]), identity(meta["tree"])
    commit_path = contained(root, meta["commit_object"])
    require(commit_path.is_file() and stat.S_ISREG(commit_path.stat().st_mode), "commit object must be a regular file")
    payload = commit_path.read_bytes()
    require(git_hash("commit", payload) == commit, "commit object identity differs")
    require(payload.startswith(b"tree " + tree.encode() + b"\n"), "commit root tree differs")
    inventory = contained(root, meta["upstream_inventory"])
    require(digest(inventory) == identity(meta["inventory_sha256"], 64), "inventory digest differs")
    with inventory.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames == FIELDS, "invalid inventory header")
        rows = list(reader)
    entries: dict[str, dict[str, str]] = {}
    children: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        require(set(row) == set(FIELDS) and all(isinstance(v, str) for v in row.values()), "invalid inventory row")
        name = relative(row["path"])
        require(name not in entries, "duplicate inventory path")
        identity(row["sha"])
        if row["type"] == "tree":
            valid = row["mode"] == "040000" and row["size"] == "" and row["import_status"] == "context"
        elif row["type"] == "blob":
            valid = row["mode"] in {"100644", "100755", "120000"} and row["size"].isdigit() and row["import_status"] in {"selected", "excluded"}
            valid = valid and (row["import_status"] != "selected" or row["mode"] != "120000")
        elif row["type"] == "commit":
            valid = row["mode"] == "160000" and row["size"] == "" and row["import_status"] == "excluded"
        else:
            valid = False
        require(valid, "invalid inventory mode/type/selection")
        entries[name] = row
        parent = str(PurePosixPath(name).parent)
        children["" if parent == "." else parent].append(row)
    require(len(entries) == count(meta["upstream_entries"]), "upstream entry count differs")
    for parent in children:
        require(not parent or parent in entries and entries[parent]["type"] == "tree", "missing inventory parent tree")
    trees = {p: r["sha"] for p, r in entries.items() if r["type"] == "tree"}
    trees[""] = tree
    for path, expected in trees.items():
        ordered = sorted(children[path], key=lambda r: PurePosixPath(r["path"]).name.encode() + (b"/" if r["type"] == "tree" else b""))
        body = b"".join(format(int(r["mode"], 8), "o").encode() + b" " + PurePosixPath(r["path"]).name.encode() + b"\0" + bytes.fromhex(r["sha"]) for r in ordered)
        require(git_hash("tree", body) == expected, "Git tree identity differs")
    blobs = {p: r for p, r in entries.items() if r["type"] == "blob"}
    require(len(blobs) == count(meta["upstream_blobs"]), "upstream blob count differs")
    require({p for p, r in blobs.items() if r["import_status"] == "selected"} == set(selected), "selected subset differs from inventory")
    for path, value in selected.items():
        relative(path)
        entry = blob_record(value, "selected blob", sha256=True)
        row = blobs[path]
        require(entry["sha"] == row["sha"] and entry["mode"] == row["mode"] and entry["size"] == int(row["size"]),
                "selected blob metadata differs")
    return blobs, {"entries": len(entries), "blobs": len(blobs), "trees": len(trees)}


def validate(root: Path = ROOT) -> dict[str, int]:
    root = root.resolve()
    pin = mapping(load_json(contained(root, "papers/scatterbrain-source-pin.json")), "pin", PIN)
    require(pin["schema_version"] == 1 and type(pin["schema_version"]) is int and pin["scope"] == "selected-source-subset", "unsupported source pin schema/scope")
    selected = mapping(pin["selected_blobs"], "selected blobs")
    require(bool(selected), "empty selected archive")
    _, initial = validate_lineage(root, pin, selected)
    archive = contained(root, pin["archive_root"])
    require(archive.is_dir(), "missing source archive")
    files, directories = set(), set()
    for path in archive.rglob("*"):
        require(not path.is_symlink(), "archive symlink is forbidden")
        name = path.relative_to(archive).as_posix()
        if path.is_dir():
            directories.add(name)
        else:
            require(stat.S_ISREG(path.stat().st_mode), "archive special file is forbidden")
            files.add(name)
    expected_dirs = {str(parent) for name in selected for parent in PurePosixPath(name).parents if str(parent) != "."}
    require(files == set(selected) and directories == expected_dirs, "archive file/directory set differs")
    for name, value in selected.items():
        entry = blob_record(value, "selected blob", sha256=True)
        path = contained(archive, name)
        payload = path.read_bytes()
        require(len(payload) == entry["size"] and git_hash("blob", payload) == entry["sha"] and digest(path) == entry["sha256"], "archive blob identity differs")
        require(bool(path.stat().st_mode & 0o111) == (entry["mode"] == "100755"), "archive executable mode differs")
    licenses = paths(pin["licenses"])
    require(bool(licenses), "missing archive license declaration")
    for path in [*licenses, *paths(pin["dependency_source_files"])]:
        require(path in selected, "missing selected license/dependency source")
    require(isinstance(pin["scope_notes"], list) and all(isinstance(note, str) for note in pin["scope_notes"]), "invalid scope notes")
    require(isinstance(pin["references"], list), "invalid reference list")
    names, commits = set(), {pin["commit"]}
    for value in pin["references"]:
        reference = mapping(value, "reference", REFERENCE)
        require(isinstance(reference["name"], str) and bool(reference["name"]) and reference["name"] not in names, "invalid/duplicate reference name")
        names.add(reference["name"])
        require(reference["role"] == "reference-metadata-only" and reference["repository"] == pin["repository"], "invalid reference role/repository")
        identity(reference["commit"])
        require(reference["commit"] not in commits, "duplicate lineage commit")
        commits.add(reference["commit"])
        blobs, _ = validate_lineage(root, reference, {})
        for name, item in mapping(reference["reference_files"], "reference files").items():
            relative(name)
            expected = blob_record(item, "reference blob")
            require(name in blobs and expected == {"sha": blobs[name]["sha"], "mode": blobs[name]["mode"], "size": int(blobs[name]["size"])}, "reference blob fingerprint differs")
    notices = mapping(pin["preserved_notices"], "preserved notices")
    require(bool(notices), "missing preserved dependency notices")
    for name, value in notices.items():
        notice = mapping(value, "notice", NOTICE)
        relative(name)
        require(PurePosixPath(name).is_relative_to("papers/notices/scatterbrain"), "dependency notice outside dedicated directory")
        require(isinstance(notice["repository"], str) and re.fullmatch(r"https://github\.com/[\w.-]+/[\w.-]+", notice["repository"]) is not None, "invalid notice repository")
        identity(notice["commit"]); identity(notice["sha"]); relative(notice["source_path"])
        require(isinstance(notice["license"], str) and bool(notice["license"]), "missing notice license label")
        require(bool(paths(notice["applies_to"])) and all(p in selected for p in paths(notice["applies_to"])), "notice application outside selected archive")
        path = contained(root, name)
        payload = path.read_bytes()
        require(len(payload) == count(notice["size"]) and git_hash("blob", payload) == notice["sha"] and digest(path) == identity(notice["sha256"], 64), "dependency notice identity differs")
    notice_root = contained(root, "papers/notices/scatterbrain")
    actual_notices = {p.relative_to(root).as_posix() for p in notice_root.rglob("*") if not p.is_dir() or p.is_symlink()}
    require(actual_notices == set(notices), "preserved notice file set differs")
    return {"selected_blobs": len(selected), "upstream_blobs": initial["blobs"], "upstream_tree_objects": initial["trees"],
            "reference_lineages": len(names), "preserved_notices": len(notices)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-dir", type=Path)
    args = parser.parse_args()
    evidence_path = args.emit_dir / "scatterbrain_source_evidence.json" if args.emit_dir else None
    if evidence_path:
        evidence_path.unlink(missing_ok=True)
    counts = validate()
    pin = load_json(ROOT / "papers/scatterbrain-source-pin.json")
    evidence = {"status": "source_identity_pass", **counts, "upstream_commit": pin["commit"], "upstream_tree": pin["tree"],
                "pin_sha256": digest(ROOT / "papers/scatterbrain-source-pin.json"),
                "build_validated": False, "dependencies_installed": False, "numerical_runtime_validated": False,
                "paper_claims_validated": False, "gpu_validated": False}
    if evidence_path:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print("SCATTERBRAIN_SOURCE_PASS " + " ".join(f"{k}={v}" for k, v in counts.items()) + " build_validated=False numerical_runtime_validated=False gpu_validated=False")


if __name__ == "__main__":
    main()
