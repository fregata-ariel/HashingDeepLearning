from __future__ import annotations

# Selected from Fly 74cc6b1 mask_utils.py; modified types/imports; see ../NOTICE.
import torch.nn.functional as F
from typing import overload
from src.models.modules.masking import FullMask, LengthMask


@overload
def pad_mask(mask: FullMask, pad_length: int, left: bool = True, value: bool = True) -> FullMask: ...
@overload
def pad_mask(mask: LengthMask, pad_length: int, left: bool = True, value: bool = True) -> LengthMask: ...
def pad_mask(mask: FullMask | LengthMask, pad_length: int, left: bool = True,
             value: bool = True) -> FullMask | LengthMask:
    if type(pad_length) is not int or pad_length < 0:
        raise ValueError('mask padding must be a nonnegative integer')
    if isinstance(mask, FullMask):
        pad = (pad_length, 0) if left else (0, pad_length)
        return FullMask(F.pad(mask.bool_matrix, pad, value=value))
    maximum = mask.shape[-1] + pad_length
    lengths = mask.lengths + pad_length if value else mask.lengths
    return LengthMask(lengths, max_len=maximum, device=lengths.device)
