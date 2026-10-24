"""Weight-free MagicPIG GPU dependencies; BF16 support remains explicit."""
from __future__ import annotations

import importlib.metadata
import math

import torch


def check(results: dict[str, object]) -> dict[str, object]:
    results.update({"flashinfer":importlib.metadata.version("flashinfer-python"),
        "native_bf16":torch.cuda.is_bf16_supported(including_emulation=False),
        "server_and_model_generation":"NOT_RUN: component fixture only"})
    results["original_bf16_server"] = "NOT_RUN" if results["native_bf16"] else "UNSUPPORTED: SM75 has no native BF16"
    # Tiny projection: exact integer-representable inputs isolate arithmetic
    # support from the later server/library dispatch and native BF16 claims.
    q = torch.arange(32, dtype=torch.float32).reshape(4,8).remainder(7)-3
    w = torch.arange(24, dtype=torch.float32).reshape(8,3).remainder(5)-2
    expected = q.double() @ w.double()
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        name = str(dtype)
        try:
            actual = q.to("cuda",dtype) @ w.to("cuda",dtype)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual.cpu().double(),expected,rtol=0,atol=0)
            results[name+"_projection"]="PASS (arithmetic; not a native BF16 instruction claim)"
        except RuntimeError as exc:
            if dtype != torch.bfloat16:
                raise
            results[name+"_projection"]="UNSUPPORTED: "+str(exc)
    # Pinned host buffers and explicit stream/event ownership across D2H/H2D.
    stream = torch.cuda.Stream()
    src = q.cuda()
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    host = torch.empty_like(q,pin_memory=True)
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        host.copy_(src,non_blocking=True)
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    assert torch.equal(host,q)
    host.add_(1)
    with torch.cuda.stream(stream):
        returned = host.to("cuda",non_blocking=True)
        completed = torch.cuda.Event()
        completed.record()
    completed.synchronize()
    assert torch.equal(returned.cpu(),q+1)
    results["pinned_transfers_nondefault_stream_events"]="PASS"
    import flashinfer

    # FP16 dependency acceptance is separate from original BF16 server support.
    heads,dim,length = 2,64,7
    query=torch.randn(1,heads,dim,device="cuda",dtype=torch.float16)*.1
    keys=torch.randn(length,heads,dim,device="cuda",dtype=torch.float16)*.1
    values=torch.randn_like(keys)
    cache=torch.zeros(1,2,heads,16,dim,device="cuda",dtype=torch.float16)
    indptr=torch.tensor([0,1],device="cuda",dtype=torch.int32)
    indices=torch.tensor([0],device="cuda",dtype=torch.int32)
    last=torch.tensor([length],device="cuda",dtype=torch.int32)
    batch_indices=torch.zeros(length,device="cuda",dtype=torch.int32)
    positions=torch.arange(length,device="cuda",dtype=torch.int32)
    results["append_api"]="FlashInfer 0.2.5 explicit batch_indices/positions; original server uses the older indptr API"
    flashinfer.append_paged_kv_cache(keys,values,batch_indices,positions,cache,indices,indptr,last,kv_layout="HND")
    torch.cuda.synchronize()
    assert torch.equal(cache[0,0,:,:length],keys.transpose(0,1)) and torch.equal(cache[0,1,:,:length],values.transpose(0,1))
    wrapper=flashinfer.BatchDecodeWithPagedKVCacheWrapper(torch.empty(16*1024*1024,device="cuda",dtype=torch.uint8),"HND")
    wrapper.plan(indptr,indices,last,heads,heads,dim,16,q_data_type=torch.float16,data_type=torch.float16)
    actual,lse=wrapper.run_return_lse(query,cache)
    scores=torch.einsum("hd,nhd->hn",query[0].cpu().double(),keys.cpu().double())/math.sqrt(dim)
    reference=torch.einsum("hn,nhd->hd",scores.softmax(-1),values.cpu().double())[None]
    torch.testing.assert_close(actual.cpu().double(),reference,rtol=2e-3,atol=2e-3)
    # FlashInfer state uses base-2 LSE. Merge two states with an independently
    # computed normalized weighted average, including unequal mass.
    a=torch.randn(1,heads,dim,device="cuda",dtype=torch.float16)
    b=torch.randn_like(a)
    sa=torch.tensor([[1.,-2.]],device="cuda")
    sb=torch.tensor([[3.,1.]],device="cuda")
    merged,merged_lse=flashinfer.merge_state(a,sa,b,sb)
    mass_a=2**sa.cpu().double(); mass_b=2**sb.cpu().double()
    reference_merge=(a.cpu().double()*mass_a[...,None]+b.cpu().double()*mass_b[...,None])/(mass_a+mass_b)[...,None]
    torch.testing.assert_close(merged.cpu().double(),reference_merge,rtol=2e-3,atol=2e-3)
    torch.testing.assert_close(merged_lse.cpu().double(),torch.log2(mass_a+mass_b),rtol=1e-5,atol=1e-5)
    torch.cuda.synchronize()
    results.update(fp16_paged_append_plan_decode="PASS",fp16_lse_merge="PASS",
                   decode_max_abs_error=float((actual.cpu().double()-reference).abs().max()),
                   original_bf16_server="NOT_RUN" if results["native_bf16"] else "UNSUPPORTED: SM75 has no native BF16")
    return results
