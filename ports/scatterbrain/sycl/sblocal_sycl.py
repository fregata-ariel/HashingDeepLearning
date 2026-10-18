"""Bounded initial SBLocal composition with an explicit SYCL local boundary.

Relation: variant of the paper, preserving the maintained source composition.
TRACE_TEST_ID: SCATTERBRAIN-LOCAL-HYBRID-SYCL-CPU
"""
from __future__ import annotations
import torch
import sys
from types import ModuleType
import sycl_local_product as local_backend
# Explicit alternate dependency selection. The numeric module is the actual
# ctypes/SYCL implementation, never a fake or a CPU arithmetic substitute.
existing = sys.modules.get("fast_transformers.local_product")
if existing is not None and existing is not local_backend:
    raise RuntimeError("select the SYCL boundary before importing another local_product backend")
if "fast_transformers" not in sys.modules:
    sys.modules["fast_transformers"] = ModuleType("fast_transformers")
sys.modules["fast_transformers.local_product"] = local_backend
from src.models.modules.attention.sblocal_attention import SBLocalAttention, AttentionWeights
from src.models.modules.masking import BaseMask, LengthMask


class SBLocalSYCLAttention(SBLocalAttention):
    def __init__(self, local_context: int, dim_heads: int, nb_features: int | None = None,
                 ortho_scaling: int = 0, causal: bool = False, softmax_temp: float | None = None,
                 attention_dropout: float = 0.0, softmax_eps: float = 0.0,
                 device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        local_backend.require_library()
        if causal or attention_dropout != 0 or softmax_eps != 0:
            raise ValueError("SYCL SBLocal supports noncausal/dropout0/eps0 only")
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError("SYCL SBLocal adapter requires CPU host tensors")
        if dtype not in (None, torch.float32):
            raise ValueError("SYCL SBLocal adapter requires FP32")
        super().__init__(local_context, dim_heads, nb_features, ortho_scaling, False,
                         softmax_temp, 0.0, 0.0, device, dtype)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: BaseMask | None = None, key_padding_mask: LengthMask | None = None,
                need_weights: bool = False, return_attn_unnormalized: bool = False) -> tuple[torch.Tensor, AttentionWeights]:
        if attn_mask is not None or self.causal or self.dropout.p != 0 or self.softmax_eps != 0:
            raise ValueError("SYCL SBLocal supports prefix padding without arbitrary/causal masks, dropout or eps")
        for tensor in (query, key, value):
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or tensor.ndim != 4 or not bool(torch.isfinite(tensor).all()):
                raise ValueError("SYCL SBLocal requires finite rank4 CPU FP32 Q/K/V")
        if key.shape[:3] != value.shape[:3] or query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
            raise ValueError("incompatible Q/K/V shapes")
        if any(d <= 0 for tensor in (query, key, value) for d in tensor.shape):
            raise ValueError("positive hybrid dimensions required")
        if key_padding_mask is not None:
            lengths = key_padding_mask.lengths
            if (lengths.device.type != "cpu" or lengths.dtype != torch.int64 or
                lengths.shape != (key.shape[0],) or
                not bool(((lengths > 0) & (lengths <= key.shape[1])).all())):
                raise ValueError("hybrid prefix lengths must be positive and within physical keys")
        result = super().forward(query, key, value, None, key_padding_mask, need_weights, return_attn_unnormalized)
        if not bool(torch.isfinite(result[0]).all()):
            raise ValueError("unsupported nonfinite hybrid row mass/output")
        return result
