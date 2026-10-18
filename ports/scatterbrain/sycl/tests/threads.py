#!/usr/bin/env python3
"""Regression: foreign threads, autograd, rejected calls and clean runtime exit."""
from __future__ import annotations
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import sycl_local_product as native
    native.load_library(args.library)
    torch.set_num_threads(2)

    def worker(worker_id: int) -> int:
        for iteration in range(3):
            nq, nk, window = 3 + worker_id % 2, 5, 3
            q = (torch.arange(nq*3, dtype=torch.float32).sin() * .2).reshape(1,1,nq,3).requires_grad_()
            k = (torch.arange(nk*3, dtype=torch.float32).cos() * .3).reshape(1,1,nk,3).requires_grad_()
            v = (torch.arange(nk*2, dtype=torch.float32).sin() * .4).reshape(1,1,nk,2).requires_grad_()
            lengths = torch.tensor([4])
            qref, kref, vref = (x.detach().double().requires_grad_() for x in (q,k,v))
            dots = native.local_dot_product(q,k,torch.zeros(nq,nk),lengths,window)
            output = native.local_weighted_average(dots.softmax(-1),v)
            logits = torch.stack([torch.stack([
                (qref[0,0,i]*kref[0,0,i-window//2+slot]).sum()
                if 0 <= i-window//2+slot < 4 else torch.tensor(float(torch.tensor(-1e24)),dtype=torch.float64)
                for slot in range(window)]) for i in range(nq)])
            weights = logits.softmax(-1)
            reference = torch.stack([sum((weights[i,slot]*vref[0,0,i-window//2+slot]
                for slot in range(window) if 0 <= i-window//2+slot < nk),torch.zeros(2,dtype=torch.float64)) for i in range(nq)]).reshape(1,1,nq,2)
            actual_grad = torch.autograd.grad(output.square().sum(),(q,k,v))
            expected_grad = torch.autograd.grad(reference.square().sum(),(qref,kref,vref))
            for actual,expected in ((output,reference), *zip(actual_grad,expected_grad)):
                assert bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                assert float((actual.detach().double()-expected.detach()).abs().max()) < 5e-6
            try:
                native.local_dot_product(q,k,torch.zeros(nq,nk),torch.tensor([nk+1]),window)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid prefix accepted in foreign thread")
        return 3

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(worker, range(4)))
    assert counts == [3,3,3,3]
    for _ in range(3):
        native.load_library(args.library)  # Idempotent, never dlclose/reload.
        gc.collect()
    try:
        native.load_library(args.library.with_name("different-library.so"))
    except ValueError:
        pass
    else:
        raise AssertionError("live library identity replacement accepted")
    print("SCATTERBRAIN_SYCL_THREADS_PASS workers=4 calls_per_worker=3 actual_all4_ops=True")

if __name__ == "__main__":
    main()
