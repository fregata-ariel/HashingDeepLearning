#!/usr/bin/env python3
"""Separate maintained Torch CPU fallback evidence; never CUDA evidence."""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mongoose_slide.slide_lib.simHash import SimHash

def main() -> None:
    cases = 0
    for k in (1, 2, 7, 31, 32):
        n, tables, d = 3, 2, 3
        data = torch.tensor([[1, 0, 0], [0, 1, 0], [1, -1, 1]], dtype=torch.float32)
        weights = torch.tensor([[(c + 2 * e) % 3 - 1 for c in range(k * tables)] for e in range(d)], dtype=torch.float32)
        expected = []
        for row in data.tolist():
            codes = []
            for table in range(tables):
                code = sum(2**bit for bit in range(k) if sum(float(row[e])*float(weights[e, table*k+bit]) for e in range(d)) > 0)
                codes.append(code if code < 2**31 else code - 2**32)
            expected.append(codes)
        hasher = SimHash(d, k, tables, weights=weights)
        result = hasher.hash(data)
        assert result.dtype == torch.int32
        assert torch.equal(result, torch.tensor(expected, dtype=torch.int32))
        assert torch.equal(hasher.hash(data, transpose=True), result.T)
        cases += 1
    print(json.dumps(dict(status="passed", path="maintained_torch_cpu_fallback", cases=cases, original_cuda_executed=False)))
if __name__ == "__main__":
    main()
