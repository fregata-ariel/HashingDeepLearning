from __future__ import annotations

from collections.abc import Sequence
from setuptools import Extension


def cythonize(
    module_list: Extension | Sequence[Extension],
) -> list[Extension]: ...
