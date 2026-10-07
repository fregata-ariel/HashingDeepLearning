#!/usr/bin/env python3
"""Audit annotations for every maintained Python source under ports/."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def _annotation_uses_any(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    return any(
        isinstance(node, ast.Name) and node.id == "Any"
        or isinstance(node, ast.Attribute) and node.attr == "Any"
        for node in ast.walk(annotation)
    )


def _audit_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    relative: Path,
) -> list[str]:
    errors: list[str] = []
    context = f"{relative}:{node.lineno}:{node.name}"

    if node.returns is None:
        errors.append(f"{context}: missing return annotation")
    elif _annotation_uses_any(node.returns):
        errors.append(f"{context}: explicit Any return annotation")

    args = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    ]
    if node.args.vararg is not None:
        args.append(node.args.vararg)
    if node.args.kwarg is not None:
        args.append(node.args.kwarg)

    for arg in args:
        if arg.arg in {"self", "cls"}:
            continue
        if arg.annotation is None:
            errors.append(
                f"{context}: argument {arg.arg!r} is unannotated"
            )
        elif _annotation_uses_any(arg.annotation):
            errors.append(
                f"{context}: argument {arg.arg!r} uses explicit Any"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("ports"))
    parser.add_argument("--expected-files", type=int)
    args = parser.parse_args()

    root = args.root.resolve()
    files = sorted(root.rglob("*.py"))

    if args.expected_files is not None and len(files) != args.expected_files:
        raise ValueError(
            f"expected {args.expected_files} Python files, found {len(files)}"
        )

    function_count = 0
    errors: list[str] = []
    for path in files:
        relative = path.relative_to(root.parent)
        tree = ast.parse(
            path.read_text(encoding="utf-8"), filename=str(relative)
        )
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
                errors.extend(_audit_function(node, relative))

    if errors:
        raise ValueError("\n".join(errors))

    print(
        "PYTHON_ANNOTATION_AUDIT_PASS "
        f"files={len(files)} functions={function_count}"
    )
    for path in files:
        print(f"PYTHON_ENROLLED {path.relative_to(root.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
