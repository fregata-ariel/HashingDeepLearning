#!/usr/bin/env python3
"""TRACE_TEST_ID: MAGICPIG-CPU-COMPOSITION-CPU; connected pinned CPU body workload.

Actual Python SimHash output crosses a checked protocol into actual maintained
LSH, correction and attention; its results return to the selected merge AST.
The only FlashInfer operations are explicitly named CPU cache/merge substitutes.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any
from test_magicpig_simhash_cpu import Tensor, selected_blocks, execute
from test_magicpig_cache_merge_cpu import (select, run_selected, Slot, IndexedValues,
    CapturePlan, Zeroable, host_merge)

SOURCE='third_party/magicpig/models/attnserver.py'
BLOCKS=['packing','centering','fill_hash','decode_hash','append_centering','partition',
        'center_store','sparse_cache','plan','merge','clear']
D=128


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def maximum(a: list[float], b: list[float]) -> float:
    require(len(a)==len(b), 'oracle shape mismatch')
    return max((abs(x-y) for x,y in zip(a,b)), default=0.0)


class MeanTensor(Tensor):
    def __setitem__(self, request: int, mean: Tensor) -> None:
        require(mean.shape==self.shape[1:], 'stored mean shape')
        width=math.prod(mean.shape)
        self.data[request*width:(request+1)*width]=mean.data
    def zero_(self) -> None:
        self.data=[0.0]*len(self.data)


class CacheLayer(list):
    def zero_(self) -> None:
        for slots in self:
            for slot in slots:slot.zero_()


class Bridge:
    def __init__(self, path: Path, groups: int) -> None:
        self.p=subprocess.Popen([str(path),'--composition-bridge',str(groups)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
        self.commands=0
        assert self.p.stdout
        for _ in range(2):
            require(self.p.stdout.readline().strip()=='NOT USING AVX512 BFLOAT16', 'bridge portable startup marker')

    def call(self, command: str, *values: Any) -> dict[str, Any]:
        payload=[command]
        for value in values:
            payload.extend(str(x) for x in value) if isinstance(value,list) else payload.append(str(value))
        assert self.p.stdin and self.p.stdout
        self.p.stdin.write(' '.join(payload)+'\n');self.p.stdin.flush()
        line=self.p.stdout.readline()
        if not line:
            error=self.p.stderr.read() if self.p.stderr else ''
            raise RuntimeError('bridge failed: '+error)
        self.commands+=1
        return json.loads(line)

    def close(self) -> None:
        assert self.p.stdin and self.p.stdout and self.p.stderr
        self.p.stdin.close()
        remaining=self.p.stdout.read();error=self.p.stderr.read();rc=self.p.wait()
        require(rc==0 and not error and not remaining,
                f'bridge exit/ASan/UBSan/LSan failed rc={rc}: {error} {remaining}')


class ClearNative:
    def __init__(self, bridge: Bridge, command: str) -> None:
        self.bridge,self.command=bridge,command
    def clear(self) -> None:
        self.bridge.call(self.command)


def vectors(cycle: int) -> tuple[list[list[float]],list[list[float]]]:
    center=[.5+cycle*.25,-.25+cycle*.125]
    offsets=[(1,.5),(2,-.5),(-1,-.5),(-2,.5)]
    if cycle: offsets.reverse()
    points=[(-.5,1),*offsets,(.75,-1)]
    keys=[[x+center[0],y+center[1]]+[0.0]*(D-2) for x,y in points]
    values=[[((t*3+d*5+cycle*7)%17-8)/16 for d in range(D)] for t in range(6)]
    return keys,values


def oracle(q: list[float], raw_keys: list[list[float]], vals: list[list[float]],
           ids: list[int], mean: list[float], correction: bool) -> dict[str, Any]:
    # Fresh direct FP64 summation over the union, not the host merger, C++
    # transform implementation, returned logits or returned probabilities.
    dots=[]; logits=[]; inclusion=[]
    for i in ids:
        key=[x-c for x,c in zip(raw_keys[i],mean)]
        dot=sum(x*y for x,y in zip(q,key));dots.append(dot)
        logit=dot/math.sqrt(D)
        if correction:
            cosine=dot/math.sqrt(sum(x*x for x in q)*sum(x*x for x in key))
            require(-1<cosine<1,'oracle supported acos domain')
            p=1-math.acos(cosine)/math.pi
            # K=1, L=2: independently enumerate accepted hit state (1,1).
            w=sum((p if a else 1-p)*(p if b else 1-p)
                  for a in (0,1) for b in (0,1) if a+b>=2)
            inclusion.append(w);logit-=math.log(w+1e-4)
        logits.append(logit)
    peak=max(logits);weights=[math.exp(x-peak) for x in logits];z=sum(weights)
    output=[sum(w*vals[i][d] for w,i in zip(weights,ids))/z for d in range(D)]
    return {'output':output,'lse':(peak+math.log(z))/math.log(2),
            'qk':dots,'logits':logits,'weights':[w/z for w in weights], 'inclusion':inclusion}


def union_oracle(q: list[float], keys: list[list[float]], vals: list[list[float]],
                 sparse_ids: list[int], dense_ids: list[int], mean: list[float], corrected: bool) -> dict[str, Any]:
    rows=[]
    for ids,corr in ((sparse_ids,corrected),(dense_ids,False)):
        ref=oracle(q,keys,vals,ids,mean,corr)
        rows.extend(zip(ref['logits'],ids))
    peak=max(s for s,_ in rows);weights=[math.exp(s-peak) for s,_ in rows];z=sum(weights)
    return {'output':[sum(w*vals[i][d] for w,(_,i) in zip(weights,rows))/z for d in range(D)],
            'lse':(peak+math.log(z))/math.log(2)}


def merge(block: list[ast.stmt], result: dict[str, Any], groups: int) -> tuple[list[list[float]],list[float]]:
    env={'gpu_hidden_states':[tuple(result['dense'][h*D:(h+1)*D]) for h in range(groups)],
         'gpu_lse':result['dense_lse'],
         'cpu_hidden_states':[tuple(result['sparse'][h*D:(h+1)*D]) for h in range(groups)],
         'cpu_lse':result['sparse_lse']}
    run_selected(block,env,flashinfer=SimpleNamespace(merge_state=host_merge))
    return [list(x) for x in env['hidden_states']],env['_']


def workload(path: Path, blocks: dict[str,list[ast.stmt]], groups: int, sampled: bool,
             metrics: dict[str,Any]) -> None:
    bridge=Bridge(path,groups)
    buffers={k:Zeroable() for k in ('nnz','results_lsh_cpu','max_value_expsum','output_cuda',
        'max_value_expsum_cuda','output','pinned_hashcode','pinned_query')}
    obj=SimpleNamespace(**buffers,K=1,L=2,device='cpu',num_layers=1,batch_size=1,
        num_attention_heads=groups,num_key_value_heads=1,head_dim=D,num_sink_tokens=1,num_local_tokens=1,
        avg_k=[MeanTensor([0.0]*D,(1,1,1,D))],
        flashinfer_kv_cache=[CacheLayer([[Slot(1,32,D),Slot(1,32,D)]])],
        kv_last_page_len=IndexedValues(1),dense_kv_last_page_len=IndexedValues(1),
        decode_wrapper=CapturePlan(),dense_decode_wrapper=CapturePlan(),
        kv_page_indptr=[0,1],kv_page_indices=[0],dense_kv_page_indptr=[0,1],dense_kv_page_indices=[0],
        page_size=32,dense_page_size=32,
        lsh_retriever=ClearNative(bridge,'CLEAR_LSH'),attn_server=ClearNative(bridge,'CLEAR_ATTN'))
    obj.hash_func=Tensor([1.0 if sampled and d==0 else 0.0 for d in range(D) for _ in range(2)],(D,2))
    execute(blocks['packing'],{'self':obj})
    previous_cycle_output=None
    try:
        for cycle in range(2):
            keys,vals=vectors(cycle)
            env={'self':obj,'layer_idx':0,'request_id':0,'seq_len':6,
                'key_cache':Tensor(sum(keys,[]),(6,1,D)),'value_cache':Tensor(sum(vals,[]),(6,1,D))}
            for name in ('partition','centering','center_store','sparse_cache'):
                run_selected(blocks[name],env)
            mean=[sum(keys[t][d] for t in range(1,5))/4 for d in range(D)]
            require(env['avg_k'].data==mean,'independent mean')
            require(env['offload_key'].data==[keys[t][d]-mean[d] for t in range(1,5) for d in range(D)],'centered offload')
            hash_env={'self':obj,'offload_key':env['offload_key'],'start':0,'end':4}
            execute(blocks['fill_hash'],hash_env)
            codes=[int(x) for x in hash_env['hash_code'].data]
            expected_codes=[int(keys[t][0]-mean[0]>0) if sampled else 0 for _ in range(2) for t in range(1,5)]
            require(codes==expected_codes,'independent literal SimHash codes')
            bridge.call('LOAD',4,codes,env['offload_key'].data,env['offload_value'].data)
            obj.dense_kv_last_page_len.values=[6]
            old_codes=None;old_dense=None;previous_step_output=None
            for step in range(2):
                # Append uses the original mean throughout decode. Values expose
                # omitted append/old cache and the cycle changes both ID order and values.
                new_key=[mean[0]+1.5+step/4,mean[1]-.75+step/8]+[0.0]*(D-2)
                new_value=[((d*3+step*5+cycle*7)%19-9)/16 for d in range(D)]
                keys.append(new_key);vals.append(new_value)
                run_selected(blocks['plan'],{'self':obj},torch=SimpleNamespace(bfloat16='bfloat16'))
                append={'self':obj,'layer_idx':0,'key_states':Tensor(new_key,(1,1,1,D))}
                run_selected(blocks['append_centering'],append)
                require(append['key_states'].data==[a-b for a,b in zip(new_key,mean)],'append centering')
                pos=obj.kv_last_page_len.values[0]-1
                for slot,tensor in zip(obj.flashinfer_kv_cache[0][0],
                    (append['key_states'].reshape(1,1,D),Tensor(new_value,(1,1,D)))):
                    slot[...,pos:pos+1,:].copy_(tensor)
                dense_ids=[0,5]+list(range(6,7+step))
                require(pos+1==len(dense_ids) and obj.dense_kv_last_page_len.values==[7+step],'append/page lengths')
                dense_keys=obj.flashinfer_kv_cache[0][0][0].value[:,:pos+1,:].data
                dense_vals=obj.flashinfer_kv_cache[0][0][1].value[:,:pos+1,:].data
                require(dense_keys==[keys[t][d]-mean[d] for t in dense_ids for d in range(D)],'independent resident cache')
                require(dense_vals==[vals[t][d] for t in dense_ids for d in range(D)],'independent resident values')
                queries=[[(-1 if step else 1)*(1+h/8),19/64]+[0.0]*(D-2) for h in range(groups)]
                query_env={'self':obj,'query_states':Tensor(sum(queries,[]),(1,groups,1,D))}
                execute(blocks['decode_hash'],query_env)
                qcodes=[int(x) for x in query_env['q_hashcode'].data]
                expected_q=[int(q[0]>0) if sampled else 0 for q in queries for _ in range(2)]
                require(qcodes==expected_q,'independent query hashes')
                expected_selected=[t for t in range(4) if sum(codes[l*4+t]==qcodes[l] for l in range(2))>=2]
                require(bool(expected_selected),'nonempty selected subset')
                result=bridge.call('QUERY',0 if sampled else 1,len(dense_ids),qcodes,sum(queries,[]),dense_keys,dense_vals)
                require(result['counts']==[len(expected_selected)]*groups and result['selected']==expected_selected*groups,'independent exact retrieved sets')
                merged,lse=merge(blocks['merge'],result,groups)
                for h,q in enumerate(queries):
                    selected=[i+1 for i in expected_selected]
                    sparse_ref=oracle(q,keys,vals,selected,mean,sampled)
                    dense_ref=oracle(q,keys,vals,dense_ids,mean,False)
                    ref=union_oracle(q,keys,vals,selected,dense_ids,mean,sampled)
                    n=len(selected);start=h*n
                    for field,bound in (('qk',1e-6),('logits',2e-5),('weights',2e-5)):
                        require(maximum(result[field][start:start+n],sparse_ref[field])<bound,f'{field} independent oracle sampled={sampled} groups={groups} cycle={cycle} step={step} h={h} actual={result[field][start:start+n]} expected={sparse_ref[field]}')
                    require(maximum(result['sparse'][h*D:(h+1)*D],sparse_ref['output'])<.004,'sparse output FP64')
                    require(maximum(result['dense'][h*D:(h+1)*D],dense_ref['output'])<.004,'resident output FP64')
                    require(abs(result['sparse_lse'][h]-sparse_ref['lse'])<3e-5,'sparse base2 LSE')
                    require(abs(result['dense_lse'][h]-dense_ref['lse'])<3e-5,'resident base2 LSE')
                    error=maximum(merged[h],ref['output'])
                    metrics['max_output_error']=max(metrics['max_output_error'],error)
                    require(error<.004 and abs(lse[h]-ref['lse'])<4e-5,'independent corrected union')
                    if not sampled:
                        direct=oracle(q,keys,vals,list(range(len(keys))),mean,False)
                        require(maximum(ref['output'],direct['output'])<1e-12 and abs(ref['lse']-direct['lse'])<1e-12,'full-selection direct dense equality')
                        require(maximum(merged[h],direct['output'])<.004,'actual full merge dense output')
                        metrics['full_selection_heads']+=1
                    metrics['head_checks']+=1
                # Mutation sensitivity uses actual bridge execution and the same
                # selected merge AST, then the unchanged independent union oracle.
                if sampled:
                    original_union = union_oracle(queries[0],keys,vals,[i+1 for i in expected_selected],dense_ids,mean,True)['output']
                    bad=bridge.call('QUERY',2,len(dense_ids),qcodes,sum(queries,[]),dense_keys,dense_vals)
                    bad_merge,_=merge(blocks['merge'],bad,groups)
                    diff=maximum(bad_merge[0],original_union)
                    require(diff>.01,'omitted correction must fail oracle tolerance');metrics['mutations']['omit_correction']+=1
                    if step:
                        stale=bridge.call('QUERY',0,len(dense_ids),old_codes,sum(queries,[]),dense_keys,dense_vals)
                        require(stale['selected']!=result['selected'],'stale query lookup must fail exact set');metrics['mutations']['stale_lookup']+=1
                        stale_cache=bridge.call('QUERY',0,len(dense_ids)-1,qcodes,sum(queries,[]),*old_dense)
                        stale_merge,_=merge(blocks['merge'],stale_cache,groups)
                        require(maximum(stale_merge[0],original_union)>.01,'stale resident cache must fail independent oracle');metrics['mutations']['stale_cache']+=1
                    shifted=list(dense_keys)
                    shifted[-D]+=4.0
                    stale_center=bridge.call('QUERY',0,len(dense_ids),qcodes,sum(queries,[]),shifted,dense_vals)
                    centered_merge,_=merge(blocks['merge'],stale_center,groups)
                    require(maximum(centered_merge[0],original_union)>.004,'stale append centering must fail independent oracle');metrics['mutations']['stale_centering']+=1
                if previous_step_output is not None:
                    require(maximum(previous_step_output,merged[0])>.01,'decode step observable output')
                previous_step_output=merged[0]
                old_codes=qcodes;old_dense=(dense_keys,dense_vals)
                metrics['decode_steps']+=1
            if previous_cycle_output is not None:
                require(maximum(previous_cycle_output,merged[0])>.01,'reuse cycle observable output')
            previous_cycle_output=merged[0]
            run_selected(blocks['clear'],{'self':obj})
            require(obj.kv_last_page_len.values==obj.dense_kv_last_page_len.values==[0],'clear lengths')
            require(all(x==0 for x in obj.avg_k[0].data),'clear mean')
            require(all(x==0 for layer in obj.flashinfer_kv_cache for slots in layer for slot in slots for x in slot.value.data),'clear cache')
            require(bridge.call('LOOKUP',qcodes)['counts']==[0]*groups,'actual LSH clear')
            metrics['clear_cycles']+=1
    finally:
        bridge.close()
    metrics['bridge_commands']+=bridge.commands


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--bridge',type=Path,required=True);parser.add_argument('--evidence',type=Path)
    args=parser.parse_args();source=(args.root/SOURCE).read_text()
    sim=selected_blocks(source);cache=select(source)
    blocks={name:(sim[name] if name in ('packing','fill_hash','decode_hash') else cache[name]) for name in BLOCKS}
    metrics={'head_checks':0,'full_selection_heads':0,'decode_steps':0,'clear_cycles':0,'bridge_commands':0,'max_output_error':0.0,
             'mutations':dict.fromkeys(('omit_correction','stale_lookup','stale_cache','stale_centering'),0)}
    for groups in (1,4,8):
        for sampled in (False,True):workload(args.bridge.resolve(),blocks,groups,sampled,metrics)
    require(metrics['head_checks']==104 and metrics['full_selection_heads']==52 and metrics['decode_steps']==24 and metrics['clear_cycles']==12, 'complete composition matrix')
    require(metrics['mutations']=={'omit_correction':12,'stale_lookup':6,'stale_cache':6,'stale_centering':12}, 'complete mutation matrix')
    statements=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.stmt)];hashes={}
    for name,nodes in blocks.items():
        start,end=nodes[0].lineno,nodes[-1].end_lineno
        selected=sorted((n for n in statements if start<=n.lineno and n.end_lineno<=end),key=lambda n:n.lineno)
        hashes[name]={'start_line':start,'end_line':end,'sha256':hashlib.sha256('\n'.join(ast.get_source_segment(source,n) or '' for n in selected).encode()).hexdigest()}
    result={'id':'MAGICPIG-CPU-COMPOSITION-CPU','name':'cpu_composition','status':'passed','source':SOURCE,
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'selected_blocks':hashes,**metrics,
        'seed':'fixed literal coordinate projections and dyadic keys/values, no RNG',
        'bridge_executable_sha256':hashlib.sha256(args.bridge.read_bytes()).hexdigest(),
        'bridge_fixture_sha256':hashlib.sha256((args.root/'tests/traceability/test_magicpig_composition_bridge.cpp').read_bytes()).hexdigest(),
        'bridge_path':str(args.bridge.resolve()),'gpu_execution':False,'torch_runtime':False,'flashinfer_execution':False,
        'sanitizer_scope':'instrumented bridge when --sanitize driver is used; Python host adapter has no ASan instrumentation',
        'full_selection_mode':'literal zero projection retrieves all; explicit actual qk/full-softmax/wv baseline with p=1 and no +1e-4 correction; differs from sparse all-candidate path'}
    print(json.dumps(result,sort_keys=True))
    if args.evidence:args.evidence.parent.mkdir(parents=True,exist_ok=True);args.evidence.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
