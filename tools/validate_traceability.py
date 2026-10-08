#!/usr/bin/env python3
"""Validate and materialize the paper/code/test traceability ledger."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "papers" / "traceability.json"
MARKER = "TRACE_TEST_ID:"

def load() -> dict[str, Any]:
    with LEDGER.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("experiments"), list):
        raise ValueError("root ledger needs an experiment list")
    # Independent task PRs own independent fragments. The emitted artifact
    # includes the fully merged ledger; validate() still rejects duplicate IDs.
    for path in sorted((ROOT / "papers/traceability").glob("*.json")):
        fragment = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(fragment, dict) or set(fragment) != {"schema_version", "experiments"} or fragment["schema_version"] != 1 or not isinstance(fragment["experiments"], list):
            raise ValueError(f"invalid experiment fragment: {path}")
        data["experiments"].extend(fragment["experiments"])
    return data

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def file_at(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"missing file: {relative}")
    return path

def validate(data: dict[str, Any]) -> dict[str, str]:
    papers = data.get("papers")
    experiments = data.get("experiments")
    if not isinstance(papers, dict) or not isinstance(experiments, list):
        raise ValueError("ledger needs object 'papers' and list 'experiments'")
    hashes: dict[str, str] = {}
    for paper_id, paper in papers.items():
        for key in ("title", "version", "source_url", "local_file"):
            if key not in paper:
                raise ValueError(f"{paper_id}: missing {key}")
        local = paper["local_file"]
        if local is not None:
            hashes[local] = digest(file_at(local))

    seen: set[str] = set()
    for item in experiments:
        eid = item.get("id")
        if not isinstance(eid, str) or not eid:
            raise ValueError("experiment id must be a non-empty string")
        if eid in seen:
            raise ValueError(f"duplicate experiment id: {eid}")
        seen.add(eid)
        if item.get("paper") not in papers:
            raise ValueError(f"{eid}: unknown paper")
        if item.get("relation") not in {"direct", "variant", "approximation", "support"}:
            raise ValueError(f"{eid}: invalid relation")
        impl = item.get("implementation")
        test = item.get("test")
        if not isinstance(impl, dict) or not isinstance(test, dict):
            raise ValueError(f"{eid}: implementation/test must be objects")
        source = file_at(impl["file"]).read_text(encoding="utf-8", errors="replace")
        if impl["symbol"] not in source:
            raise ValueError(f"{eid}: symbol {impl['symbol']!r} not found")
        test_source = file_at(test["file"]).read_text(encoding="utf-8", errors="replace")
        if test.get("id") != eid:
            raise ValueError(f"{eid}: test id must equal experiment id")
        if f"{MARKER} {eid}" not in test_source:
            raise ValueError(f"{eid}: test marker not found")
    return hashes

def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

def emit(data: dict[str, Any], hashes: dict[str, str], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    snapshot = {"git_commit": head(), "paper_sha256": hashes, "traceability": data}
    (out / "traceability.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Paper / code / test traceability", "", f"Git commit: `{head()}`", "",
             "## Paper files", ""]
    if hashes:
        lines += [f"- `{sha}  {path}`" for path, sha in sorted(hashes.items())]
    else:
        lines += ["- No local paper files are registered in this commit."]
    lines += ["", "## Experiments", ""]
    for item in data["experiments"]:
        impl, test = item["implementation"], item["test"]
        lines += [f"### {item['id']}", "",
                  f"- Paper: `{item['paper']}` — {item['paper_location']}",
                  f"- Relation: `{item['relation']}`",
                  f"- Mechanism: {item['mechanism']}",
                  f"- Implementation: `{impl['file']}::{impl['symbol']}`",
                  f"- Test: `{test['file']}` (`{test['id']}`)",
                  f"- Conditions: `{json.dumps(item.get('conditions', {}), sort_keys=True)}`",
                  f"- Note: {item.get('note', '')}", ""]
    (out / "PAPER_CODE_TEST_MAP.md").write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-dir", type=Path)
    args = parser.parse_args()
    data = load()
    hashes = validate(data)
    print(f"TRACEABILITY_VALID experiments={len(data['experiments'])}")
    print(f"GIT_COMMIT {head()}")
    if hashes:
        for path, sha in sorted(hashes.items()):
            print(f"PAPER_SHA256 {sha}  {path}")
    else:
        print("PAPER_SHA256 none (no local paper files registered)")
    if args.emit_dir:
        emit(data, hashes, args.emit_dir)
        print(f"TRACEABILITY_ARTIFACT {args.emit_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
