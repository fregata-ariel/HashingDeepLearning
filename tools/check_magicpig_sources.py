#!/usr/bin/env python3
"""Verify the scoped MagicPIG archive against the complete pinned Git tree.

This gate validates source identity and selection transparency. It does not
compile the reference, execute attention, or validate CUDA/AVX instructions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["path", "mode", "type", "sha", "size", "import_status"]


def relative(value: str) -> str:
    """Require one canonical repository-relative POSIX path."""
    path = PurePosixPath(value)
    if not value or not path.parts or path.is_absolute() or str(path) != value or ".." in path.parts or "\\" in value or any(c in value for c in "\t\n\r\0"):
        raise ValueError(f"unsafe/noncanonical path: {value!r}")
    return value


def git_hash(kind: str, content: bytes) -> str:
    return hashlib.sha1(kind.encode() + b" " + str(len(content)).encode() + b"\0" + content).hexdigest()


def validate(root: Path = ROOT) -> dict[str, int]:
    pin = json.loads((root / "papers/magicpig-source-pin.json").read_text())
    if pin["schema_version"] != 1 or pin["scope"] != "selected-source-subset":
        raise ValueError("unsupported source pin schema/scope")
    if not re.fullmatch(r"[0-9a-f]{40}", pin["commit"]) or not re.fullmatch(r"[0-9a-f]{40}", pin["tree"]):
        raise ValueError("invalid commit/tree identity")
    inventory = root / relative(pin["upstream_inventory"])
    with inventory.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError("invalid inventory header")
        rows = list(reader)
    entries: dict[str, dict[str, str]] = {}
    children: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if set(row) != set(FIELDS) or any(v is None for v in row.values()):
            raise ValueError("invalid inventory row")
        path = relative(row["path"])
        if path in entries or not re.fullmatch(r"[0-9a-f]{40}", row["sha"]):
            raise ValueError(f"duplicate/invalid inventory identity: {path}")
        if row["type"] == "tree":
            valid = row["mode"] == "040000" and row["size"] == "" and row["import_status"] == "context"
        elif row["type"] == "blob":
            valid = row["mode"] in {"100644", "100755"} and row["size"].isdigit() and row["import_status"] in {"selected", "excluded"}
        else:
            valid = False
        if not valid:
            raise ValueError(f"invalid inventory type/mode/scope: {path}")
        entries[path] = row
        parent = str(PurePosixPath(path).parent)
        children["" if parent == "." else parent].append(row)
    if len(entries) != pin["upstream_entries"]:
        raise ValueError("upstream entry count differs")
    for parent in children:
        if parent and (parent not in entries or entries[parent]["type"] != "tree"):
            raise ValueError(f"missing parent tree: {parent}")
    trees = {p: r["sha"] for p, r in entries.items() if r["type"] == "tree"}
    trees[""] = pin["tree"]
    for path, expected in trees.items():
        # Git orders directories by their basename with a trailing slash.
        ordered = sorted(children[path], key=lambda r: PurePosixPath(r["path"]).name.encode() + (b"/" if r["type"] == "tree" else b""))
        body = b"".join(format(int(r["mode"], 8), "o").encode() + b" " + PurePosixPath(r["path"]).name.encode() + b"\0" + bytes.fromhex(r["sha"]) for r in ordered)
        if git_hash("tree", body) != expected:
            raise ValueError(f"upstream tree identity differs: {path or '<root>'}")
    blobs = {p: r for p, r in entries.items() if r["type"] == "blob"}
    selected = {p: r for p, r in blobs.items() if r["import_status"] == "selected"}
    if len(blobs) != pin["upstream_blobs"] or set(selected) != set(pin["selected_blobs"]):
        raise ValueError("selected/upstream blob inventory differs")
    archive = root / relative(pin["archive_root"])
    if not archive.resolve().is_relative_to(root.resolve()):
        raise ValueError("archive escapes repository")
    actual = {str(p.relative_to(archive)) for p in archive.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(selected):
        raise ValueError("archive file set differs from selected subset")
    for path, row in selected.items():
        expected = pin["selected_blobs"][path]
        if expected != {"sha": row["sha"], "mode": row["mode"], "size": int(row["size"])}:
            raise ValueError(f"selected metadata differs: {path}")
        file = archive / path
        if file.is_symlink() or not file.resolve().is_relative_to(archive.resolve()):
            raise ValueError(f"archive symlink/escape: {path}")
        content = file.read_bytes()
        if len(content) != expected["size"] or git_hash("blob", content) != expected["sha"]:
            raise ValueError(f"archive blob identity differs: {path}")
    for path in pin["licenses"] + pin["dependency_source_files"]:
        if relative(path) not in selected:
            raise ValueError(f"missing selected license/dependency input: {path}")
    for dep in pin["dependency_subtrees"]:
        path = relative(dep["path"])
        if trees.get(path) != dep["sha"]:
            raise ValueError(f"dependency subtree identity differs: {path}")
    return {"selected_blobs": len(selected), "upstream_blobs": len(blobs), "tree_objects": len(trees)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-dir", type=Path)
    args = parser.parse_args()
    if args.emit_dir:
        (args.emit_dir / "magicpig_source_evidence.json").unlink(missing_ok=True)
    counts = validate()
    pin = json.loads((ROOT / "papers/magicpig-source-pin.json").read_text())
    evidence = {"status": "source_identity_pass", **counts, "upstream_commit": pin["commit"], "upstream_tree": pin["tree"], "inventory_sha256": hashlib.sha256((ROOT / pin["upstream_inventory"]).read_bytes()).hexdigest(), "build_validated": False, "numerical_runtime_validated": False, "gpu_validated": False}
    if args.emit_dir:
        args.emit_dir.mkdir(parents=True, exist_ok=True)
        (args.emit_dir / "magicpig_source_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("MAGICPIG_SOURCE_PASS " + " ".join(f"{k}={v}" for k, v in counts.items()) + " build_validated=False gpu_validated=False")


if __name__ == "__main__":
    main()
