"""Focused characterization test for the maintained MONGOOSE scheduler."""
from __future__ import annotations
import torch
from mongoose_reformer.reformer_lib.scheduler import Scheduler

# TRACE_TEST_ID: MONGOOSE-SCHEDULER-CHANGE

class _FixedHash:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
    def hash(self, _data: torch.Tensor) -> torch.Tensor:
        return self.value.clone()

def _scheduler(cached: torch.Tensor, current: torch.Tensor, thresh: float) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.hash_codes = cached.clone()
    scheduler.thresh_hash = _FixedHash(current)
    scheduler.thresh = thresh
    return scheduler

def test_strict_threshold_and_cache_update() -> None:
    cached = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
    updated = torch.tensor([[1, 0, 0, 0]], dtype=torch.int64)
    at_threshold = _scheduler(cached, updated, 0.25)
    assert at_threshold.detect_change(torch.empty(0)) is False
    assert torch.equal(at_threshold.hash_codes, cached)
    above_threshold = _scheduler(cached, updated, 0.20)
    assert above_threshold.detect_change(torch.empty(0)) is True
    assert torch.equal(above_threshold.hash_codes, updated)
    above_threshold.thresh_hash = _FixedHash(updated)
    assert above_threshold.detect_change(torch.empty(0)) is False

if __name__ == "__main__":
    test_strict_threshold_and_cache_update()
    print("TRACE_TEST_PASS MONGOOSE-SCHEDULER-CHANGE")
