from __future__ import annotations

# Selected from HazyResearch/fly 74cc6b1 src/utils/padding.py; see ../NOTICE.
# Adapted from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/autopadder.py
import torch
import torch.nn.functional as F

def pad_to_multiple(tensor: torch.Tensor, multiple: int, dims: int | tuple[int, ...] = -1,
                    value: float = 0) -> torch.Tensor:
    if type(multiple) is not int or multiple <= 0:
        raise ValueError('padding multiple must be a positive integer')
    selected = [dims] if isinstance(dims, int) else list(dims)
    if not selected or any(d < -tensor.ndim or d >= tensor.ndim for d in selected):
        raise ValueError('padding dimension outside tensor rank')
    # convert dims from negative to positive
    selected = [d if d >= 0 else tensor.ndim + d for d in selected]
    padding = [0] * (2 * tensor.ndim)
    for d in selected:
        size = tensor.size(d)
        # Pytorch's JIT doesn't like divmod
        # m, remainder = divmod(size, multiple)
        m = size // multiple
        remainder = size - m * multiple
        if remainder != 0:
            padding[2 * (tensor.ndim - d - 1) + 1] = multiple - remainder
    if all(p == 0 for p in padding):
        return tensor
    else:
        return F.pad(tensor, tuple(padding), value=value)
