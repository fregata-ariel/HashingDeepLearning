#!/usr/bin/env python3
"""Extract and validate the typed Reformer attention/hash core for M5 issue #24."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "ports"
    / "mongoose"
    / "mongoose_reformer"
    / "reformer_lib"
    / "reformer_pytorch.py"
)

TARGET_FUNCTIONS = {
    "sort_key_val",
    "batched_index_select",
    "mine_triplet_examples",
    "process_inputs_chunk",
    "chunked_sum",
    "default",
    "max_neg_value",
    "expand_dim",
    "merge_dims",
    "split_at_index",
}
TARGET_CLASSES = {
    "LSHAttention",
    "TripletLSHAttention",
    "FullQKAttention",
    "LSHSelfAttention",
}


def _function_name(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return node.name


def _require_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    context: str,
) -> None:
    if node.returns is None:
        raise ValueError(f"{context}.{_function_name(node)}: missing return annotation")

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
            raise ValueError(
                f"{context}.{_function_name(node)}: "
                f"argument {arg.arg!r} is unannotated"
            )


def _audit_selected(nodes: list[ast.stmt]) -> None:
    for top in nodes:
        context = getattr(top, "name", "<module>")
        for child in ast.walk(top):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _require_annotations(child, context)


PRELUDE = """from __future__ import annotations

from functools import partial, reduce
from operator import mul
from typing import Callable, Protocol, TypeAlias, TypeVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from mongoose_reformer.reformer_lib.scheduler import Scheduler

TOKEN_SELF_ATTN_VALUE = -5e4

_DefaultT = TypeVar("_DefaultT")
AttentionResult: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


class AttentionChunkFn(Protocol):
    def __call__(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> AttentionResult: ...


class _LocalAttentionCallable(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class LocalAttention(nn.Module):
    def __init__(
        self,
        *,
        window_size: int,
        causal: bool,
        dropout: float,
        shared_qk: bool,
        look_forward: int,
    ) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return q
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected: list[ast.stmt] = []
    found_functions: set[str] = set()
    found_classes: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in TARGET_FUNCTIONS:
                selected.append(node)
                found_functions.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name in TARGET_CLASSES:
            selected.append(node)
            found_classes.add(node.name)

    missing_functions = TARGET_FUNCTIONS - found_functions
    missing_classes = TARGET_CLASSES - found_classes
    if missing_functions or missing_classes:
        raise ValueError(
            "missing selected symbols: "
            f"functions={sorted(missing_functions)}, "
            f"classes={sorted(missing_classes)}"
        )

    _audit_selected(selected)

    rendered = PRELUDE + "\n\n"
    rendered += "\n\n".join(ast.unparse(node) for node in selected)
    rendered += "\n"

    args.output.write_text(rendered, encoding="utf-8")
    print(
        "REFORMER_ATTENTION_TYPE_AUDIT_PASS "
        f"functions={len(found_functions)} classes={len(found_classes)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
