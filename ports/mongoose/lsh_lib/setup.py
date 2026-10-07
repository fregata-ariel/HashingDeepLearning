from __future__ import annotations

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def build_extension() -> Extension:
    """Return the native clsh extension definition used by CI/runtime builds."""
    return Extension(
        "clsh",
        sources=["clsh.pyx", "LSH.cpp"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    )


def main() -> None:
    setup(
        name="clsh",
        ext_modules=cythonize(build_extension()),
    )


if __name__ == "__main__":
    main()
