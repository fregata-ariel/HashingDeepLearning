#!/usr/bin/env python3
"""Audit annotations for every maintained Python source under ports/."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPOSITORY / "tools/maintained_python_inventory.json"


@dataclass(frozen=True)
class TypingGroup:
    name: str
    files: tuple[Path, ...]
    stubs: tuple[Path, ...]
    mypy_path: tuple[Path, ...]


def _contained_path(base: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("inventory paths must be nonempty relative strings")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"unsafe inventory path: {relative}")
    path = (base / raw).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"inventory path escapes root: {relative}")
    return path


def load_inventory(inventory: Path, root: Path) -> list[TypingGroup]:
    """Enroll exact Python/stub paths, including deliberate zero-file groups.

    Group roots partition the maintained tree; additions, removals, same-count
    replacements and new stubs require review of the inventory. Immutable
    third_party Python and selected-AST test adapters are outside this tree.
    """
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"maintained root does not exist: {root}")
    repository = root.parent
    data = json.loads(inventory.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"version", "groups"} or data["version"] != 1 or not isinstance(data["groups"], list):
        raise ValueError("invalid maintained Python inventory schema")
    groups: list[TypingGroup] = []
    names: set[str] = set()
    roots: list[Path] = []
    enrolled: set[Path] = set()
    for record in data["groups"]:
        if not isinstance(record, dict) or set(record) != {"name", "root", "python_files", "stub_files", "mypy_path"}:
            raise ValueError("invalid typing group schema")
        name = record["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("invalid or duplicate typing group name")
        names.add(name)
        directory = _contained_path(repository, record["root"])
        if not directory.is_dir() or not directory.is_relative_to(root):
            raise ValueError(f"typing group outside maintained root: {name}")
        if any(directory.is_relative_to(p) or p.is_relative_to(directory) for p in roots):
            raise ValueError(f"overlapping typing group roots: {name}")
        roots.append(directory)
        selections: list[tuple[Path, ...]] = []
        for field, suffix in (("python_files", ".py"), ("stub_files", ".pyi")):
            values = record[field]
            if not isinstance(values, list) or not all(isinstance(p, str) for p in values) or values != sorted(set(values)):
                raise ValueError(f"{name}: {field} must be sorted unique paths")
            paths = tuple(_contained_path(directory, p) for p in values)
            if any(p.suffix != suffix or not p.is_file() for p in paths):
                raise ValueError(f"{name}: missing file or wrong suffix in {field}")
            selections.append(paths)
            enrolled.update(paths)
        search = record["mypy_path"]
        if not isinstance(search, list) or not search:
            raise ValueError(f"{name}: missing mypy search path")
        mypy_path = tuple(_contained_path(repository, p) for p in search)
        if len(set(mypy_path)) != len(mypy_path) or any(not p.is_dir() for p in mypy_path) or directory not in mypy_path:
            raise ValueError(f"{name}: invalid mypy search path")
        groups.append(TypingGroup(name, selections[0], selections[1], mypy_path))
    actual = {p.resolve() for suffix in ("*.py", "*.pyi") for p in root.rglob(suffix)}
    if actual != enrolled:
        extra = sorted(str(p.relative_to(repository)) for p in actual - enrolled)
        missing = sorted(str(p.relative_to(repository)) for p in enrolled - actual)
        raise ValueError(f"maintained Python inventory mismatch: unenrolled={extra}, missing={missing}")
    return groups


def strict_typecheck(groups: list[TypingGroup], repository: Path) -> None:
    """Use identical strict mypy flags for every enrolled group, even new ones."""
    for group in groups:
        sources = [*group.files, *group.stubs]
        if not sources:
            print(f"PYTHON_TYPING_EMPTY group={group.name} maintained_sources=0", flush=True)
            continue
        env = dict(os.environ)
        env["MYPYPATH"] = os.pathsep.join(str(p) for p in group.mypy_path)
        command = [sys.executable, "-m", "mypy", "--strict"]
        if group.name == "magicpig":
            # Preserve existing projects' NumPy typing (NDArray uses an Any
            # shape internally), while forbidding Any in the new native APIs.
            command.append("--disallow-any-explicit")
        command += ["--config-file", str(repository / "mypy.ini"), *(str(p) for p in sources)]
        subprocess.run(command, cwd=repository, env=env, check=True)
        print(f"PYTHON_TYPING_PASS group={group.name} sources={len(sources)}", flush=True)


def _annotation_uses_any(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        # Quoted forward annotations must not bypass the no-Any contract.
        try:
            parsed = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False  # mypy diagnoses malformed annotation syntax.
        return _annotation_uses_any(parsed)
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
    parser.add_argument("--root", type=Path, default=REPOSITORY / "ports")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--mypy", action="store_true", help="strict-check every enrolled source/stub group")
    args = parser.parse_args()

    root = args.root.resolve()
    groups = load_inventory(args.inventory, root)
    files = sorted(path for group in groups for path in group.files)

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
    for group in groups:
        print(f"PYTHON_INVENTORY group={group.name} python={len(group.files)} stubs={len(group.stubs)}")
    if args.mypy:
        strict_typecheck(groups, root.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
