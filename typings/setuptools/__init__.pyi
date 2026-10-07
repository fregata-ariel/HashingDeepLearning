from __future__ import annotations

from collections.abc import Sequence


class Extension:
    def __init__(
        self,
        name: str,
        sources: Sequence[str],
        *,
        language: str | None = ...,
        include_dirs: Sequence[str] = ...,
        extra_compile_args: Sequence[str] = ...,
    ) -> None: ...


def setup(
    *,
    name: str,
    ext_modules: Sequence[Extension],
) -> None: ...
