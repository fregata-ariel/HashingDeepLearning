#!/usr/bin/env python3
"""Emit two exact archived kernels as native CUDA negative controls."""
import hashlib
from pathlib import Path
import sys
from check_gslide_cpu import ROOT, check_archive, extract


def generate(output: Path) -> None:
    check_archive()
    chunks = ['#include "utils.h"\n#include "CscActNodes.h"\n#include <cfloat>\n']
    for file, symbol, expected in [
        ("kernel.cu", "get_dense_acts_knl", "2c322cb2d1a0d0c57afdc84c71126580139b697d0658140ed0232a9ba14e9c32"),
        ("lshKnl.cu", "init_hash_knl", "70c8b02dc858b79de82a13f5ab0a30e058a89c41349020c2ebefb9581dac32ff"),
    ]:
        body, _ = extract((ROOT / "third_party/g-slide/src" / file).read_text(), symbol)
        if hashlib.sha256(body.encode()).hexdigest() != expected:
            raise ValueError("archived negative control changed: " + symbol)
        # Rename only the symbol; keep all arithmetic, indexing and barriers.
        chunks.append(body.replace(symbol + "(", "original_" + symbol + "(", 1))
    output.write_text("\n".join(chunks) + "\n")


if __name__ == "__main__":
    generate(Path(sys.argv[1]))
