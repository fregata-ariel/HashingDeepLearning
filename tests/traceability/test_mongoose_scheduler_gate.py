"""Integration oracle for MONGOOSE scheduler gating of triplet work."""
from __future__ import annotations

import torch
from torch import nn

from mongoose_reformer.reformer_lib.reformer_pytorch import LSHSelfAttention

# TRACE_TEST_ID: MONGOOSE-SCHEDULER-GATE


class _Gate:
    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls = 0

    def detect_change(self, _updated: torch.Tensor) -> bool:
        self.calls += 1
        return self.result


class _AttentionStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.triplet_flags: list[bool] = []
        self.loss_calls = 0

    def forward(
        self,
        qk: torch.Tensor,
        v: torch.Tensor,
        query_len: int | None = None,
        input_mask: torch.Tensor | None = None,
        triplet_examples: bool = False,
        **_kwargs: object,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        self.triplet_flags.append(triplet_examples)
        seq = qk.shape[1]
        attn = torch.zeros(
            qk.shape[0], seq, seq, dtype=qk.dtype, device=qk.device
        )
        buckets = torch.zeros(
            qk.shape[0], seq, dtype=torch.long, device=qk.device
        )
        if triplet_examples:
            return qk, attn, buckets, qk, qk, -qk
        return qk, attn, buckets, None, None, None

    def triplet_forward(
        self, _x: torch.Tensor, _p: torch.Tensor, _n: torch.Tensor
    ) -> torch.Tensor:
        self.loss_calls += 1
        return torch.tensor(2.0)


def _module() -> LSHSelfAttention:
    return LSHSelfAttention(
        dim=4,
        heads=1,
        bucket_size=2,
        n_hashes=1,
        causal=False,
        attn_chunks=1,
        use_full_attn=False,
        full_attn_thres=0,
        attn_type="triplet",
        max_seq_len=4,
        scheduler_hashes=1,
        thresh=0.01,
        dropout=0.0,
        post_attn_dropout=0.0,
    )


def test_scheduler_gate_controls_triplet_example_work() -> None:
    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) / 10.0

    blocked = _module()
    blocked_gate = _Gate(False)
    blocked_attn = _AttentionStub()
    blocked.scheduler = blocked_gate
    blocked.lsh_attn = blocked_attn

    out_blocked = blocked(x, calc_triplet=True)
    assert out_blocked.shape == x.shape
    assert blocked_gate.calls == 1
    assert blocked_attn.triplet_flags == [False]
    assert blocked_attn.loss_calls == 0
    assert blocked.triplet_loss == 0.0

    allowed = _module()
    allowed_gate = _Gate(True)
    allowed_attn = _AttentionStub()
    allowed.scheduler = allowed_gate
    allowed.lsh_attn = allowed_attn

    out_allowed = allowed(x, calc_triplet=True)
    assert out_allowed.shape == x.shape
    assert allowed_gate.calls == 1
    assert allowed_attn.triplet_flags == [True]
    assert allowed_attn.loss_calls == 1
    assert torch.is_tensor(allowed.triplet_loss)
    assert float(allowed.triplet_loss) == 2.0


if __name__ == "__main__":
    test_scheduler_gate_controls_triplet_example_work()
    print("TRACE_TEST_PASS MONGOOSE-SCHEDULER-GATE")
