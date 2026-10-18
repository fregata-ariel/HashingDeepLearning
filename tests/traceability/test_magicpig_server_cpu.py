"""TRACE_TEST_ID: MAGICPIG-SERVER-CPU; full maintained CPU orchestration.

Real Torch resident math and actual maintained native sparse/full/retrieval
callees. Expected union attention enumerates each token in FP64 directly;
it does not call the implementation's merge, hash or partition helpers.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import ModuleType

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ports/magicpig"))
from magicpig_cpu.orchestration import CPUAttentionServer, ServerConfig
from magicpig_cpu.helpers import causal_mask, repeat_kv, rotate_half, rotary, rms_norm
from test_magicpig_extension_cpu import rejected


def quantize(value: float) -> float:
    """Independent scalar representation boundary: actual Torch BF16 input policy."""
    return float(torch.tensor(value, dtype=torch.bfloat16))


def partition(key: Tensor, value: Tensor) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]], list[float]]:
    """Literal sink1/local1 independent partition, BF16 mean/subtract semantics."""
    rows = key[:, 0].double().tolist(); vals = value[:, 0].double().tolist()
    dim = key.shape[-1]
    average = [quantize(sum(rows[i][d] for i in range(1, len(rows)-1))/(len(rows)-2)) for d in range(dim)]
    centered = [[quantize(x-average[d]) for d,x in enumerate(row)] for row in rows]
    return centered[1:-1], vals[1:-1], [centered[0], centered[-1]], [vals[0], vals[-1]], average


def hash_row(row: list[float], projection: Tensor, k: int) -> list[int]:
    """Scalar FP64 strict-sign oracle, with explicit BF16-safe sign margin."""
    result: list[int] = []
    for table in range(4):
        packed = 0
        for bit in range(k):
            dot = sum(row[d]*float(projection[d, table*k+bit]) for d in range(len(row)))
            assert dot == 0 or abs(dot) >= 1/65536
            if dot > 0:
                packed += 2**bit
        result.append(packed)
    return result


def union_oracle(query: list[float], keys: list[list[float]], values: list[list[float]], corrected_count: int) -> tuple[list[float], float]:
    """One FP64 softmax over concatenated supplied tokens; no LSE merge formula."""
    logits: list[float] = []
    for i,row in enumerate(keys):
        dot = sum(x*y for x,y in zip(query,row))
        logit = dot / math.sqrt(len(query))
        if i < corrected_count:
            cosine = dot / math.sqrt(sum(x*x for x in query)*sum(x*x for x in row))
            assert -1 < cosine < 1
            hit = (1-math.acos(cosine)/math.pi)**3
            probability = sum(math.comb(4,j)*hit**j*(1-hit)**(4-j) for j in range(2,5))
            logit -= math.log(probability+1e-4)
        logits.append(logit)
    maximum = max(logits)
    weights = [math.exp(x-maximum) for x in logits]
    total = sum(weights)
    result = [sum(weight*row[d] for weight,row in zip(weights,values))/total for d in range(len(query))]
    return result, (maximum+math.log(total))/math.log(2)


def helper_oracles() -> int:
    """Independent mask/head/rotation/RMS expectations on fixed tiny operands."""
    mask = causal_mask(3)
    for query in range(3):
        for key in range(3):
            assert float(mask[query,key]) == (0 if key <= query else torch.finfo(torch.float32).min)
    broadcast = torch.zeros((2, 2, 3, 3)) + mask
    assert bool((broadcast[:, :, 0, 1:] < -1e30).all())
    value = torch.arange(2*2*3*4).reshape(2,2,3,4).float()
    repeated = repeat_kv(value,3)
    for batch in range(2):
        for head in range(6):
            assert torch.equal(repeated[batch,head], value[batch,head//3])
    assert repeat_kv(value,1) is value
    assert rotate_half(torch.tensor([1.,2.,3.,4.])).tolist() == [-3.,-4.,1.,2.]
    positions = torch.tensor([[2,0,1],[1,2,0]])
    cosine = torch.tensor([[1.,1.,1.,1.],[0.,0.,0.,0.],[.6,.8,.6,.8]])
    sine = torch.tensor([[0.,0.,0.,0.],[1.,1.,1.,1.],[.8,.6,.8,.6]])
    actual = rotary(value,cosine,sine,positions)
    for b in range(2):
        for h in range(2):
            for p in range(3):
                row=value[b,h,p].tolist(); rot=[-row[2],-row[3],row[0],row[1]]; pos=int(positions[b,p])
                expected=[row[d]*float(cosine[pos,d])+rot[d]*float(sine[pos,d]) for d in range(4)]
                assert max(abs(float(actual[b,h,p,d])-expected[d]) for d in range(4)) < 1e-5
    x = torch.tensor([[[1.,-2.,3.,-4.],[.5,1.,-1.5,2.]]])
    weight = torch.tensor([.5,1.,1.5,2.]); epsilon=.01
    actual = rms_norm(x,weight,epsilon)
    for p,row in enumerate(x[0].tolist()):
        scale=math.sqrt(sum(v*v for v in row)/4+epsilon)
        assert max(abs(float(actual[0,p,d])-row[d]*float(weight[d])/scale) for d in range(4)) < 1e-6
    return 5


def exercise_server(sparse: ModuleType, lsh: ModuleType) -> dict[str, object]:
    """Actual complete two-step dense/sampled/baseline/native connection and reset."""
    config=ServerConfig(layers=3,heads=4,kvheads=1,dim=128,batch=2,capacity=12)
    projection=torch.zeros((128,12),dtype=torch.bfloat16)
    for col in range(12):
        projection[(col*7)%128,col]=1 if col%2 else -1
    cold_native = sparse.SparseAttentionServer()
    cold = CPUAttentionServer(config, cold_native)
    rejected(cold.alloc_buffer)
    rejected(cold.plan)
    rejected(lambda: cold.fill(-1, 0, torch.ones((4, 1, 128), dtype=torch.bfloat16), torch.ones((4, 1, 128), dtype=torch.bfloat16)))
    for length in [2, 3, 5]:
        operand = torch.ones((length, 1, 128), dtype=torch.bfloat16)
        rejected(lambda: cold.fill(1, 0, operand, operand))
        assert not cold.filled and not cold.resident[1].count_nonzero()
        assert not cold_native.get_key_cache(1).count_nonzero()
    total_heads=0; routes: list[str]=[]
    retrieval_counts: set[int] = set()
    for k in [0,3]:
        server=CPUAttentionServer(config,sparse.SparseAttentionServer(),lsh.LSH() if k else None,k=k,projection=projection if k else None)
        if k:
            assert server.projection is not None and server.projection.data_ptr() != projection.data_ptr()
        for reuse in range(2):
            prompts: dict[tuple[int,int],tuple[Tensor,Tensor]]={}
            expected_parts: dict[tuple[int,int],tuple[list[list[float]],list[list[float]],list[list[float]],list[list[float]],list[float]]]={}
            # Interleaved fill then interleaved build detects shared hash-buffer routing bugs.
            for request in [1,0]:
                for layer in [2,0,1]:
                    length=7+request
                    base=torch.arange(length*128).reshape(length,1,128)
                    key=(((base*7+layer*5+request*3)%41-20).float()/32).bfloat16()
                    value=(((base*11+layer*3+request)%37-18).float()/32).bfloat16()
                    server.fill(layer,request,key,value)
                    prompts[layer,request]=(key,value)
                    if not layer:
                        expected_parts[layer,request]=([],[],key[:,0].double().tolist(),value[:,0].double().tolist(),[])
                    if layer:
                        expected_parts[layer,request]=partition(key,value)
                        if k:
                            offkey=expected_parts[layer,request][0]
                            assert server.chunk_ranges[layer,request] == [(start,min(start+2,len(offkey))) for start in range(0,len(offkey),2)]
                            assert server.hashes[layer,request][0].t().tolist() == [hash_row(row,projection,3) for row in offkey]
            for layer in [1,2,0]:
                for request in [0,1]:
                    server.build_table(layer,request)
                    if layer and k:
                        codes,payload_ids=server.sorted_payloads[layer,request]
                        original=server.hashes[layer,request]
                        for table in range(4):
                            assert set(payload_ids[0,table].tolist()) == set(range(original.shape[-1]))
                            assert codes[0,table].tolist() == sorted(original[0,table].tolist())
                            for i,index in enumerate(payload_ids[0,table].tolist()):
                                assert int(codes[0,table,i]) == int(original[0,table,index])
            for step in range(2):
                server.plan(); rejected(server.plan)
                for layer in [2,0,1]:
                    qbase=torch.arange(2*4*128).reshape(2,4,1,128)
                    query=(((qbase*13+step*3+layer)%43-21).float()/32).bfloat16()
                    newbase=torch.arange(2*128).reshape(2,1,1,128)
                    newkey=(((newbase*3+step+layer*7)%31-15).float()/32).bfloat16()
                    newvalue=(((newbase*5+step*7+layer)%29-14).float()/32).bfloat16()
                    snapshot=server.resident[layer].clone(); lengths=list(server.lengths[layer])
                    rejected(lambda: server.decode(torch.zeros_like(query),newkey,newvalue,layer))
                    assert torch.equal(server.resident[layer],snapshot) and server.lengths[layer]==lengths
                    actual=server.decode(query,newkey,newvalue,layer).reshape(2,4,128)
                    routes.append(server.last_branch)
                    for request in range(2):
                        if layer:
                            offkey,offvalue,reskey,resvalue,average=expected_parts[layer,request]
                            reskey.append([quantize(float(newkey[request,0,0,d])-average[d]) for d in range(128)])
                            resvalue.append(newvalue[request,0,0].double().tolist())
                            assert server.resident[layer][request,0,0,:len(reskey)].double().tolist() == reskey
                        else:
                            _,_,reskey,resvalue,_=expected_parts[layer,request]
                            reskey.append(newkey[request,0,0].double().tolist())
                            resvalue.append(newvalue[request,0,0].double().tolist())
                            assert server.resident[layer][request,0,0,:len(reskey)].double().tolist() == reskey
                            assert server.resident[layer][request,1,0,:len(resvalue)].double().tolist() == resvalue
                        for head in range(4):
                            q=query[request,head,0].double().tolist()
                            ids: list[int]=[]
                            if layer:
                                if k:
                                    qnorm=math.sqrt(sum(x*x for x in q))
                                    # Original query normalization is BF16 and uses BF16 norm.
                                    denominator=quantize(qnorm)
                                    normalized=[quantize(x/denominator) for x in q]
                                    qcode=hash_row(normalized,projection,3)
                                    ids=[i for i,row in enumerate(offkey) if sum(a==b for a,b in zip(hash_row(row,projection,3),qcode))>=2]
                                    retrieval_counts.add(len(ids))
                                    flat=request*4+head
                                    assert int(server.nnz[flat]) == len(ids)
                                    assert set(server.results[flat,:len(ids)].tolist()) == set(ids)
                                else:
                                    ids=list(range(len(offkey)))
                                    assert int(server.nnz[request*4+head]) == len(ids)
                                expected,lse=union_oracle(q,[offkey[i] for i in ids]+reskey,[offvalue[i] for i in ids]+resvalue,len(ids) if k else 0)
                                if ids:
                                    _,offlse=union_oracle(q,[offkey[i] for i in ids],[offvalue[i] for i in ids],len(ids) if k else 0)
                                    assert abs(float(server.last_offload_lse[request,head])-offlse) <= .03
                                else:
                                    assert torch.isneginf(server.last_offload_lse[request,head])
                            else:
                                expected,lse=union_oracle(q,reskey,resvalue,0)
                            assert all(math.isfinite(x) for x in expected)
                            assert max(abs(float(actual[request,head,d])-expected[d]) for d in range(128)) <= .008
                            _,resident_lse=union_oracle(q,reskey,resvalue,0)
                            assert abs(float(server.last_resident_lse[request,head])-resident_lse) < 1e-5
                            total_heads+=1
            rejected(server.plan)
            server.clear()
            assert not server.pending and not server.filled and not server.built and not server.hashes
            assert not server.nnz.count_nonzero() and all(not cache.count_nonzero() for cache in server.resident)
            assert all(lengths==[0,0] for lengths in server.lengths+server.offload_lengths)
    assert 0 in retrieval_counts and any(count > 0 for count in retrieval_counts)
    # Packing boundary/strict zero tie independent scalar oracle for every K1..11.
    for k in range(1,12):
        proj=torch.zeros((128,k*4),dtype=torch.bfloat16)
        for col in range(k*4): proj[col,col]=1
        server=CPUAttentionServer(config,sparse.SparseAttentionServer(),lsh.LSH(),k=k,projection=proj)
        operand=torch.zeros((2,128),dtype=torch.bfloat16)
        operand[0,:k*4]=torch.tensor([0 if i%3==0 else 1 if i%3==1 else -1 for i in range(k*4)])
        actual=server._hash(operand)
        for row in range(2): assert actual[row].tolist()==hash_row(operand[row].double().tolist(),proj,k)
    return {"status":"pass","head_union_checks":total_heads,"full_orchestrator":True,"callee_boundary":"actual maintained native CPU plus actual Torch resident attention/CPU merge; FlashInfer uncalled","branches":sorted(set(routes)),"reuses":4,"steps":2,"helper_oracles":helper_oracles(),"packing_K":list(range(1,12)),"sampled_nnz_values":sorted(retrieval_counts),"gpu_verified":False}
