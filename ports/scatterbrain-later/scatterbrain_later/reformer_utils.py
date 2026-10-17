from __future__ import annotations

# Selected from HazyResearch/fly 74cc6b1 reformer_attention.py; modified for typed CPU maintenance.
# Source correspondence and adapted SMYRF/Reformer notices: ../SOURCE_PROVENANCE.json, ../NOTICE.
# Adapted from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py
import torch

def chunked_sum(tensor: torch.Tensor, chunks: int = 1) -> torch.Tensor:
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)

def max_neg_value(tensor: torch.Tensor) -> float:
    return -torch.finfo(tensor.dtype).max
