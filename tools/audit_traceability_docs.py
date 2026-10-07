#!/usr/bin/env python3
"""Audit verified C/C++/Cython traceability documentation contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "papers" / "traceability.json"
DOCUMENTED_SUFFIXES = {".cpp", ".h", ".pyx", ".cu", ".cuh"}


def _load() -> dict[str, Any]:
    with LEDGER.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    data = _load()
    checked = 0

    for experiment in data["experiments"]:
        implementation = experiment.get("implementation")
        if not isinstance(implementation, dict):
            continue

        relative = implementation.get("file")
        symbol = implementation.get("symbol")
        if not isinstance(relative, str) or not isinstance(symbol, str):
            continue

        path = ROOT / relative
        if path.suffix not in DOCUMENTED_SUFFIXES:
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        experiment_id = experiment["id"]
        relation = experiment["relation"]

        if symbol not in text:
            raise ValueError(
                f"{experiment_id}: symbol {symbol!r} missing from {relative}"
            )

        marker = f"TRACE_TEST_ID: {experiment_id}"
        if marker not in text:
            raise ValueError(
                f"{experiment_id}: documentation marker {marker!r} "
                f"missing from {relative}"
            )

        # Non-direct mappings are the dangerous cases for overclaiming paper
        # equivalence. Require the implementation file itself to name that
        # weaker relation so a nearby/adjacent contract can explain it.
        if relation in {"support", "variant", "approximation"}:
            if relation.lower() not in text.lower():
                raise ValueError(
                    f"{experiment_id}: relation {relation!r} is not "
                    f"documented in {relative}"
                )

        checked += 1

    if checked == 0:
        raise ValueError("no C/C++/Cython traceability entries were audited")

    print(f"TRACEABILITY_DOC_AUDIT_PASS entries={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
