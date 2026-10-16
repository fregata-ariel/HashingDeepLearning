#!/usr/bin/env python3
"""Whole-module frozen SBSmyrf reference characterization, with a scalar oracle.

Run through tools/check_scatterbrain_later_cpu.py for pinned fetch/hash gates.
No copied upstream implementation, native extension, GPU or maintained port.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import unittest

import torch

SOURCE = None
EMIT = None
PIN = '74cc6b14347b4edcf5c49098e80a9ef47c0b21bf'

torch.set_num_threads(1)


def demand(ok, message):
    if not ok:
        raise AssertionError(message)


def maximum(a, b):
    demand(len(a) == len(b), 'oracle shape mismatch')
    demand(all(math.isfinite(x) for x in a), 'nonfinite actual value')
    demand(all(math.isfinite(y) for y in b), 'nonfinite reference value')
    errors = [abs(x-y) for x,y in zip(a,b)]
    demand(all(math.isfinite(error) for error in errors), 'nonfinite comparison error')
    return max(errors, default=0.)


def phi(x, projection, temp):
    row = [math.sqrt(temp)*sum(u*v for u,v in zip(w,x)) for w in projection]
    norm = temp*sum(u*u for u in x)/2
    return [math.exp(y-norm)/math.sqrt(2*len(projection)) for y in [*row,*[-z for z in row]]]


def oracle(q, k, v, projection, permutations, qcluster, kcluster, lengths, temp):
    # Inputs are plain nested lists. No source feature vectors, residual, round
    # counts, returned mask or normalization code is reused here.
    batch, T, heads, E = len(q),len(q[0]),len(q[0][0]),len(q[0][0][0])
    S,D = len(k[0]),len(v[0][0][0])
    qpos,kpos = permutations
    padded_q = ((T+qcluster-1)//qcluster)*qcluster
    padded_k = ((S+kcluster-1)//kcluster)*kcluster
    demand(len(qpos)==len(kpos)>0, 'hash round shape')
    for side, size in [(qpos,padded_q),(kpos,padded_k)]:
        for round_positions in side:
            demand(len(round_positions)==batch*heads, 'hash batch/head shape')
            for permutation in round_positions:
                demand(all(type(i) is int for i in permutation) and sorted(permutation)==list(range(size)), 'hash permutation bijection')
    support=[];multiplicity=[]
    for bh in range(batch*heads):
        counts={}
        for r in range(len(qpos)):
            for c in range(len(qpos[r][bh])//qcluster):
                qs=qpos[r][bh][c*qcluster:(c+1)*qcluster]
                ks=kpos[r][bh][c*kcluster:(c+1)*kcluster]
                for i in qs:
                    for j in ks:
                        if i<T and j<lengths[bh//heads]: counts[i,j]=counts.get((i,j),0)+1
        support.append(set(counts));multiplicity.append(counts)
    output=[[[[0.]*D for h in range(heads)] for i in range(T)] for b in range(batch)]
    normalized=[[[[0.]*S for i in range(T)] for h in range(heads)] for b in range(batch)]
    raw_kernel=[[[[0.]*S for i in range(T)] for h in range(heads)] for b in range(batch)]
    raw_lowrank=[[[[0.]*S for i in range(T)] for h in range(heads)] for b in range(batch)]
    support_mask=[[[[False]*S for i in range(T)] for h in range(heads)] for b in range(batch)]
    numerator=[[[[0.]*D for h in range(heads)] for i in range(T)] for b in range(batch)]
    denominator=[[[[0.] for h in range(heads)] for i in range(T)] for b in range(batch)]
    hybrid=[]; low_rank=[]; residual=[]
    for b in range(batch):
        for h in range(heads):
            bh=b*heads+h
            kp=[phi(k[b][j][h],projection,temp) for j in range(lengths[b])]
            for i in range(T):
                qp=phi(q[b][i][h],projection,temp)
                row=[]
                for j in range(lengths[b]):
                    approx=sum(x*y for x,y in zip(qp,kp[j]))
                    exact=math.exp(temp*sum(x*y for x,y in zip(q[b][i][h],k[b][j][h])))
                    selected=(i,j) in support[bh]
                    row.append(exact if selected else approx)
                    raw_kernel[b][h][i][j]=row[-1]
                    raw_lowrank[b][h][i][j]=approx
                    support_mask[b][h][i][j]=selected
                    hybrid.append(row[-1]);low_rank.append(approx)
                    if selected: residual.append(exact-approx)
                mass=sum(row)
                denominator[b][i][h][0]=mass
                if mass:
                    normalized[b][h][i][:len(row)]=[x/mass for x in row]
                    for d in range(D):
                        numerator[b][i][h][d]=sum(w*v[b][j][h][d] for j,w in enumerate(row))
                        output[b][i][h][d]=numerator[b][i][h][d]/mass
    flat=lambda a:torch.tensor(a,dtype=torch.float64).flatten().tolist()
    for label,values in [('hybrid',hybrid),('low rank',low_rank),('residual',residual)]:
        demand(all(math.isfinite(x) for x in values), 'nonfinite reference '+label)
    return {'output':flat(output),'weights':flat(normalized),'kernel':flat(raw_kernel),'raw_lowrank':flat(raw_lowrank),
            'numerator':flat(numerator),'denominator':flat(denominator),'support':flat(support_mask),'hybrid':hybrid,'low_rank':low_rank,
            'selected_pairs':sum(len(x) for x in support),'max_pair_multiplicity':max((c for row in multiplicity for c in row.values()),default=0),
            'negative_residuals':sum(x<0 for x in residual),'zero_residuals':sum(x==0 for x in residual)}


def inputs(dtype, T=3, S=5):
    B,H,E,D=2,2,4,3
    make=lambda shape,a,b,m,scale:(((torch.arange(math.prod(shape))*a+b)%m)-(m//2)).reshape(shape).to(dtype)/scale
    return make((B,T,H,E),5,3,17,16),make((B,S,H,E),3,2,19,16),make((B,S,H,D),7,4,13,8)


def run_case(dtype, lengths, *, need_weights, clusters=(2,3), T=3, S=5, fixed_positions=None, rounds=3, mixed_precision_characterization=False, zero_queries_keys=False, features=8):
    torch.manual_seed(2357)
    q,k,v=inputs(dtype,T,S)
    demand(all(x.device.type=='cpu' for x in [q,k,v]),'CPU fixture device')
    if zero_queries_keys:q.zero_();k.zero_()
    model=SBSmyrfAttention(rounds,*clusters,4,nb_features=features,softmax_temp=.5,attention_dropout=0.,dtype=dtype)
    captured={};original=model.hash_vectors
    def observe(query,key,key_padding_mask=None):
        result=original(query,key,key_padding_mask) if fixed_positions is None else tuple(torch.tensor(x,dtype=torch.long) for x in fixed_positions)
        captured['positions']=tuple(t.detach().clone().tolist() for t in result)
        return result
    model.hash_vectors=observe
    mask=LengthMask(torch.tensor(lengths),max_len=S)
    # Observe actual forward locals at its normalization line, before optional
    # returned-weight code reshapes them. No arithmetic/source is replaced.
    code=model.forward.__func__.__code__
    lines=Path(code.co_filename).read_text().splitlines()
    observation_line=next(i for i,line in enumerate(lines,1) if 'out_normalized = out /' in line)
    intermediates={}
    def trace(frame,event,arg):
        if frame.f_code is code and event=='line' and frame.f_lineno==observation_line:
            for name in ['out','normalization','normalization_log_scale']:
                intermediates[name]=frame.f_locals[name].detach().clone()
        return trace
    old_trace=sys.gettrace();sys.settrace(trace)
    try:out,weights=model(q,k,v,key_padding_mask=mask,need_weights=need_weights,return_attn_unnormalized=need_weights)
    finally:sys.settrace(old_trace)
    reference=oracle(q.tolist(),k.tolist(),v.tolist(),model.feature_map.projection_matrix.tolist(),captured['positions'],*clusters,lengths,.5)
    error=maximum(out.detach().flatten().tolist(),reference['output'])
    # Source duplicate counts are explicitly converted to float32 before log.
    # Mixed-round float64 is a separate source-precision characterization; it
    # must not be reported as satisfying the single-round strict FP64 budget.
    budget=3e-6 if dtype==torch.float32 or mixed_precision_characterization else 2e-11
    demand(set(intermediates)=={'out','normalization','normalization_log_scale'}, 'missing actual source intermediates')
    scale=intermediates['normalization_log_scale'].exp()
    raw_numerator=intermediates['out']*scale
    raw_denominator=intermediates['normalization']*scale
    reshape=lambda x:x.reshape(2,2,-1,x.shape[-1]).permute(0,2,1,3)[:,:T].flatten().tolist()
    numerator_error=maximum(reshape(raw_numerator),reference['numerator'])
    denominator_error=maximum(reshape(raw_denominator),reference['denominator'])
    # Even full-support ratios can cancel the float32 log-count error while
    # their raw kernel/HV/H1 retain it. Report that separately from output.
    raw_budget=3e-5 if dtype==torch.float32 or rounds>1 else 2e-10
    demand(numerator_error<raw_budget, f'numerator independent oracle {numerator_error}')
    demand(denominator_error<raw_budget, f'denominator independent oracle {denominator_error}')
    demand(all(x==0. or x>=1e-6 for x in intermediates['normalization'].flatten().tolist()),'outside unclamped reference domain')
    demand(error<budget,f'output independent oracle {error}')
    result={'dtype':str(dtype),'global_default_dtype':str(torch.get_default_dtype()),'hash_execution':'actual E2LSH' if fixed_positions is None else 'fixed permutation control','lengths':lengths,'need_weights':need_weights,'rounds':rounds,'output_max_error':error,'oracle_budget':budget,
            'selected_pairs':reference['selected_pairs'],'max_pair_multiplicity':reference['max_pair_multiplicity'],
            'negative_residuals':reference['negative_residuals'],'numerator_max_error':numerator_error,'denominator_max_error':denominator_error,
            'normalizer_domain':'unclamped positive rows or exact zero masked rows','raw_oracle_budget':raw_budget,
            'raw_mixed_precision_characterization':dtype==torch.float64 and rounds>1,
            'raw_meets_strict_fp64_budget':max(numerator_error,denominator_error)<2e-10 if dtype==torch.float64 else None}
    if dtype==torch.float64:
        result['strict_fp64_budget']=2e-11
        result['meets_strict_fp64_budget']=error<2e-11
        result['mixed_precision_characterization']=mixed_precision_characterization
    if need_weights:
        weights,raw,lowrank,residual_mask=weights
        kernel_error=maximum(raw.detach().flatten().tolist(),reference['kernel'])
        lowrank_error=maximum(lowrank.detach().flatten().tolist(),reference['raw_lowrank'])
        demand(kernel_error<raw_budget,f'kernel independent oracle {kernel_error}')
        demand(lowrank_error<raw_budget,f'lowrank independent oracle {lowrank_error}')
        weight_error=maximum(weights.detach().flatten().tolist(),reference['weights'])
        demand(weight_error<budget,f'weights independent oracle {weight_error}')
        result['weight_max_error']=weight_error
        result['kernel_max_error']=kernel_error;result['lowrank_max_error']=lowrank_error
        if zero_queries_keys:
            support=reference['support'];actual=residual_mask.flatten().tolist()
            missing=sum(bool(selected) and not marked for selected,marked in zip(support,actual))
            demand(reference['zero_residuals']>0 and missing>0,'selected zero residual mask characterization')
            result['selected_zero_residuals']=reference['zero_residuals']
            result['selected_pairs_absent_from_returned_residual_mask']=missing
    else:demand(weights is None,'need_weights=False contract')
    return result,(model,q,k,v,captured,reference)


def gradients(state):
    model,q,k,v,captured,reference=state
    # Freeze support obtained from an earlier actual hash execution, matching
    # the source's no-grad treatment of support. All forward arithmetic remains
    # the actual unchanged SBSmyrf forward implementation.
    positions=tuple(torch.tensor(x,dtype=torch.long) for x in captured['positions'])
    model.hash_vectors=lambda *args,**kwargs:positions
    q=q.detach().requires_grad_();k=k.detach().requires_grad_();v=v.detach().requires_grad_()
    lengths=[5,2];mask=LengthMask(torch.tensor(lengths),max_len=5)
    out,_=model(q,k,v,key_padding_mask=mask,need_weights=False)
    loss_weights=((torch.arange(out.numel())*3+1)%11-5).reshape(out.shape).to(torch.float64)/8
    (out*loss_weights).sum().backward()
    raw=[q.detach().tolist(),k.detach().tolist(),v.detach().tolist()]
    projection=model.feature_map.projection_matrix.tolist();weight=loss_weights.flatten().tolist()
    def objective():
        values=oracle(*raw,projection,captured['positions'],2,3,lengths,.5)['output']
        demand(all(math.isfinite(x) for x in values), 'nonfinite gradient reference')
        result = sum(a*b for a,b in zip(values,weight))
        demand(math.isfinite(result), 'nonfinite gradient objective')
        return result
    errors=[]
    probes=[(0,(0,0,0,0)),(0,(1,2,1,3)),(1,(0,4,0,2)),(1,(1,0,1,1)),(2,(0,2,0,0)),(2,(1,1,1,2))]
    for tensor,coordinate in probes:
        parent=raw[tensor]
        for index in coordinate[:-1]:parent=parent[index]
        last=coordinate[-1];initial=parent[last];step=1e-6
        parent[last]=initial+step;plus=objective();parent[last]=initial-step;minus=objective();parent[last]=initial
        expected=(plus-minus)/(2*step);actual=[q,k,v][tensor].grad[coordinate].item()
        error=maximum([actual],[expected])
        errors.append(error);demand(error<2e-6,'conditional independent gradient')
    demand(all(torch.isfinite(x.grad).all().item() for x in [q,k,v]),'finite Q/K/V gradients')
    return {'coordinate_checks':len(probes),'max_error':max(errors),'source_support':'captured actual hash, frozen for derivative comparison'}


def run():
    evidence={'source_pin':PIN,'source_tree':'1821d435ef810ce88038ef81965c65cd0ac8c6bc',
              'dependencies':{p:importlib.metadata.version(p) for p in ['torch','numpy','einops','tqdm']},
              'default_dtype':str(torch.get_default_dtype()),'cases':[],'observations':[],'gpu_execution':False,'native_extension_execution':False}
    evidence['finite_guard']=finite_guard_controls()
    def known_failure(label,fn,fragment):
        try:fn()
        except RuntimeError as error:
            demand(fragment in str(error),'unexpected known-failure diagnostic '+label)
            return {'case':label,'status':'known_source_failure_reproduced','exception':type(error).__name__,'message':str(error)}
        raise AssertionError('expected frozen source failure absent: '+label)
    evidence['observations'].append(known_failure('float64 actual hash under float32 default',
        lambda:run_case(torch.float64,[5,2],need_weights=False),'double != float'))
    for dtype in [torch.float64,torch.float32]:
        previous=torch.get_default_dtype();torch.set_default_dtype(dtype)
        try:
            for lengths in [[5,5],[5,2],[0,0]]:
                item,state=run_case(dtype,lengths,need_weights=False,mixed_precision_characterization=dtype==torch.float64);evidence['cases'].append(item)
                if dtype==torch.float64 and lengths==[5,2]:evidence['gradient_mixed_rounds']=gradients(state)
                if dtype==torch.float32 and lengths==[5,2]:frozen_positions=state[4]['positions']
        finally:torch.set_default_dtype(previous)
    item,_=run_case(torch.float32,[5,2],need_weights=True);evidence['cases'].append(item)
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    try:
        item,state=run_case(torch.float64,[5,2],need_weights=False,rounds=1);evidence['cases'].append(item)
        evidence['gradient_single_round']=gradients(state)
    finally:torch.set_default_dtype(previous)
    # Full support: one paired cluster per round. Independent hybrid oracle then
    # equals exact dense attention for every valid key.
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    try:item,_=run_case(torch.float64,[5,2],need_weights=False,clusters=(4,6));item['full_support']=True;evidence['cases'].append(item)
    finally:torch.set_default_dtype(previous)
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
    try:
        item,_=run_case(torch.float64,[5,2],need_weights=True,rounds=1,features=4,zero_queries_keys=True)
        item['selected_zero_residual_control']=True;evidence['cases'].append(item)
    finally:torch.set_default_dtype(previous)
    evidence['observations'].append(known_failure('float64 need_weights with fixed support under float32 default',
        lambda:run_case(torch.float64,[5,2],need_weights=True,fixed_positions=frozen_positions),
        'scatter(): Expected self.dtype to be equal to src.dtype'))
    evidence['observations'].append(known_failure('cold deterministic eval',cold_eval,
        'subscript d has size 8 for operand 1 which does not broadcast with previously seen size 4'))
    odd=odd_features();demand(odd=={'requested':7,'effective':6},'odd feature source contract')
    redraw=redraw_state();demand(redraw=={'requested_redraw':1,'actual_redraw':math.pi,'initial_shape':[4,8],
        'drawn_shape':[4,4],'subsequent_five_draws_changed':[False]*5},'redraw source contract')
    evidence['observations'].extend([{'case':'odd effective features','status':'observed','result':odd},
                                     {'case':'redraw state','status':'observed','result':redraw}])
    closure = CLOSURE
    evidence['source_sha256']={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in closure}
    imported=set()
    for name,module in tuple(sys.modules.items()):
        if name=='src' or name.startswith('src.'):
            filename=getattr(module,'__file__',None)
            if filename:
                path=Path(filename).resolve()
                demand(path.is_relative_to(SOURCE.resolve()),'source import escaped pinned root')
                imported.add(path.relative_to(SOURCE.resolve()).as_posix())
    demand(imported==set(CLOSURE)|set(PACKAGES),'unexpected or missing actual project import closure')
    evidence['actual_imported_project_files']=sorted(imported)
    evidence['status']='supported_cases_passed_with_separate_source_failure_observations'
    EMIT.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print('SCATTERBRAIN_LATER_CPU_PASS cases=' + str(len(evidence['cases'])))


def cold_eval():
    feature=SBPerformerFeatures(4,8,deterministic_eval=True,dtype=torch.float64)
    feature.eval();feature.new_feature_map('cpu')
    return {'projection_shape':list(feature.projection_matrix.shape),'output_shape':list(feature.forward_queries(torch.ones(1,3,4,dtype=torch.float64))[0].shape)}


def odd_features():
    feature=SBPerformerFeatures(4,7,dtype=torch.float64);feature.new_feature_map('cpu')
    return {'requested':7,'effective':feature.forward_queries(torch.ones(1,3,4,dtype=torch.float64))[0].shape[-1]}


def redraw_state():
    torch.manual_seed(17)
    feature=SBPerformerFeatures(4,8,redraw=1,dtype=torch.float64)
    initial_shape=list(feature.projection_matrix.shape)
    feature.new_feature_map('cpu');first=feature.projection_matrix.clone();changed=[]
    for _ in range(5):
        feature.new_feature_map('cpu');changed.append(not torch.equal(first,feature.projection_matrix))
    return {'requested_redraw':1,'actual_redraw':feature.redraw,'initial_shape':initial_shape,'drawn_shape':list(first.shape),'subsequent_five_draws_changed':changed}


def finite_guard_controls():
    rejected=0
    for bad in [float('nan'),float('inf'),float('-inf')]:
        for side in ['actual','reference']:
            actual=[0.,1.,2.,3.];reference=actual.copy()
            (actual if side=='actual' else reference)[-1]=bad
            try:maximum(actual,reference)
            except AssertionError as error:
                demand('nonfinite '+side in str(error),'finite guard wrong rejection')
                rejected+=1
            else:raise AssertionError('finite guard accepted '+side)
    try:maximum([1.7e308],[-1.7e308])
    except AssertionError as error:
        demand('nonfinite comparison error' in str(error),'overflow wrong rejection');rejected+=1
    else:raise AssertionError('finite guard accepted error overflow')
    return {'late_nonfinite_and_overflow_rejections':rejected}


class FiniteComparisonContracts(unittest.TestCase):
    def test_late_nan_infinity_and_error_overflow_reject(self):
        self.assertEqual(finite_guard_controls()['late_nonfinite_and_overflow_rejections'],7)

    def test_finite_difference_and_shape_contract(self):
        self.assertEqual(maximum([1.,4.],[1.,2.]),2.)
        self.assertEqual(maximum([],[]),0.)
        with self.assertRaisesRegex(AssertionError,'shape mismatch'):
            maximum([1.],[])


def main():
    global SOURCE,EMIT,CLOSURE,PACKAGES,SBSmyrfAttention,SBPerformerFeatures,LengthMask
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--emit',type=Path,required=True)
    args=parser.parse_args();SOURCE=args.source.resolve();EMIT=args.emit
    EMIT.unlink(missing_ok=True);EMIT.parent.mkdir(parents=True,exist_ok=True)
    # Metadata belongs to the maintained driver, not the downloaded package.
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from tools.check_scatterbrain_later_cpu import CLOSURE,PACKAGES
    sys.path.insert(0,str(SOURCE))
    from src.models.attention.sbsmyrf_attention import SBSmyrfAttention
    from src.models.attention.feature_maps_sb import SBPerformerFeatures
    from src.models.modules.masking import LengthMask
    try:run()
    except AssertionError as error:
        EMIT.write_text(json.dumps({'status':'oracle_rejected','reason':str(error)},allow_nan=False)+'\n')
        print('SCATTERBRAIN_LATER_ORACLE_REJECTED '+str(error),file=sys.stderr)
        raise SystemExit(2) from error


if __name__=='__main__':
    try:main()
    except Exception as error:
        # Never put downloaded source excerpts from a traceback into artifacts.
        print('SCATTERBRAIN_LATER_RUNTIME_FAILURE '+type(error).__name__,file=sys.stderr)
        raise SystemExit(1) from None
