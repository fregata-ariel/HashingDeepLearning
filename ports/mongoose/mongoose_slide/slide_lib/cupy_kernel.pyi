from __future__ import annotations

class cupyKernel:
    def __init__(self, kernel: str, func_name: str) -> None: ...
    def __call__(
        self,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[int],
        strm: int,
    ) -> None: ...
