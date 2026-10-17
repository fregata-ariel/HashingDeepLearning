"""Independent scalar-oracle checks of the maintained later SBSmyrf CPU component.

The frozen-reference oracle is loaded without importing its historical src
namespace. Reference parity executes in a separate process. No native/GPU path.
TRACE_TEST_ID: SCATTERBRAIN-SBSMYRF-HYBRID-CPU.
TRACE_TEST_ID: SCATTERBRAIN-SBSMYRF-DTYPE-CPU.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Callable, Protocol, TypeAlias, cast

import torch

if TYPE_CHECKING:
    from scatterbrain_later.attention import SBSmyrfAttention
    from src.models.modules.masking import LengthMask

ROOT = Path(__file__).resolve().parents[2]
Positions: TypeAlias = tuple[list[list[list[int]]], list[list[list[int]]]]
Values: TypeAlias = list[list[list[list[float]]]]
State: TypeAlias = tuple['SBSmyrfAttention', torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object], dict[str, object]]


class Cases(Protocol):
    def __call__(self, dtype: torch.dtype, lengths: list[int], *, need_weights: bool,
                 rounds: int = 3, features: int = 8, zero_queries_keys: bool = False,
                 clusters: tuple[int,int] = (2,3)
                 ) -> tuple[dict[str, object], State]: ...


class HashSamples(Protocol):
    alpha: torch.Tensor
    beta: torch.Tensor


class InvalidConstructor(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


class HybridOracle(Protocol):
    def __call__(self, q: Values, k: Values, v: Values, projection: list[list[float]],
                 permutations: Positions, qcluster: int, kcluster: int,
                 lengths: list[int], temp: float) -> dict[str,object]: ...


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def finite_metric(data: dict[str, object], name: str) -> float:
    value = data[name]
    require(type(value) in (int, float), 'invalid metric '+name)
    number = cast(float, value)
    require(math.isfinite(number), 'nonfinite metric '+name)
    return number


def oracle_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location('independent_frozen_hybrid_oracle',
        ROOT/'tests/traceability/test_scatterbrain_later_cpu.py')
    require(spec is not None and spec.loader is not None, 'oracle loader')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_fp64(result: dict[str,object]) -> None:
    for name in ('output_max_error','numerator_max_error','denominator_max_error'):
        require(finite_metric(result,name)<(2e-11 if name=='output_max_error' else 2e-10),'strict FP64 '+name)
    if result['need_weights']:
        for name in ('weight_max_error','kernel_max_error','lowrank_max_error'):
            require(finite_metric(result,name)<2e-10,'strict FP64 '+name)
    result['maintained_strict_fp64_pass']=True


def snapshot(runner: Cases, model_class: type[SBSmyrfAttention], dtype: torch.dtype
             ) -> tuple[dict[str, object], State, dict[str, object]]:
    captured: dict[str, object] = {}
    def profile(frame: FrameType, event: str, arg: object) -> None:
        if event != 'return':
            return
        if frame.f_code.co_name == '__init__' and frame.f_code.co_filename.endswith('/hash_utils.py'):
            instance = cast(HashSamples, frame.f_locals['self'])
            captured.update(alpha=instance.alpha.tolist(), beta=instance.beta.tolist(),
                alpha_dtype=str(instance.alpha.dtype), beta_dtype=str(instance.beta.dtype),
                alpha_device=str(instance.alpha.device), beta_device=str(instance.beta.device))
        if frame.f_code is model_class.hash_vectors.__code__:
            model = cast('SBSmyrfAttention', frame.f_locals['self'])
            q = cast(torch.Tensor, frame.f_locals['query']); k = cast(torch.Tensor, frame.f_locals['key'])
            captured.update(query_transform=model.hash_fn.Q(q).tolist(), key_transform=model.hash_fn.K(k).tolist())
        if frame.f_code is model_class.forward.__code__ and arg is not None:
            out, weights = cast(tuple[torch.Tensor, tuple[torch.Tensor, ...]], arg)
            captured['output'] = out.detach().tolist()
            captured['weights_and_raw'] = [w.detach().tolist() for w in weights]
    previous = sys.getprofile(); sys.setprofile(profile)
    try: result, state = runner(dtype, [5, 2], need_weights=True)
    finally: sys.setprofile(previous)
    captured['positions'] = state[4]['positions']
    captured['projection'] = state[0].feature_map.projection_matrix.tolist()
    return result, state, captured


def invalid_domains(model: SBSmyrfAttention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    mask_class: type[LengthMask]) -> int:
    cases: list[tuple[str, Callable[[], object]]] = [
        ('dtype mismatch',lambda:model.forward(q,k.float(),v)),
        ('unsupported half',lambda:model.forward(q.half(),k.half(),v.half())),
        ('unsupported device',lambda:model.forward(q.to('meta'),k,v)),
        ('head dimension',lambda:model.forward(q[...,:-1],k,v)),
        ('batch/head routing',lambda:model.forward(q,k,v[:,:,:1])),
        ('cluster count',lambda:model.forward(q,k[:,:3],v[:,:3])),
        ('empty sequence',lambda:model.forward(q[:,:0],k,v)),
        ('nonfinite query',lambda:model.forward(q*float('nan'),k,v)),
        ('nonfinite values',lambda:model.forward(q,k,v*float('inf'))),
        ('overflowed norm',lambda:model.forward(q*1e308,k,v)),
        ('bad prefix lengths',lambda:model.forward(q,k,v,key_padding_mask=mask_class(torch.tensor([-1,6]),max_len=5))),
        ('bad length dtype',lambda:model.forward(q,k,v,key_padding_mask=mask_class(torch.tensor([5.,2.]),max_len=5))),
        ('missing weights flag',lambda:model.forward(q,k,v,return_attn_unnormalized=True)),
    ]
    rejected = 0
    for label, call in cases:
        before = {n:t.clone() for n,t in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        try:call()
        except ValueError:rejected += 1
        else:raise AssertionError('invalid domain accepted '+label)
        require(torch.equal(rng,torch.get_rng_state()),'invalid domain consumed RNG '+label)
        require(all(torch.equal(t,model.state_dict()[n]) for n,t in before.items()),'invalid domain mutated model '+label)
    return rejected


def weights_value_gradients(state: State, oracle: ModuleType) -> dict[str,object]:
    model,q,k,v,_,_=state
    model.feature_map.deterministic_eval=True;model.eval()
    q=q.detach().requires_grad_();k=k.detach().requires_grad_();v=v.detach().requires_grad_()
    mask=cast(type['LengthMask'],importlib.import_module('src.models.modules.masking').LengthMask)(torch.tensor([5,2]),max_len=5)
    out,weights=model.forward(q,k,v,key_padding_mask=mask,need_weights=True)
    require(isinstance(weights,torch.Tensor),'normalized weight return type')
    assert isinstance(weights,torch.Tensor)
    via_weights=torch.einsum('bhts,bshd->bthd',weights,v)
    maximum=cast(Callable[[list[float],list[float]],float],oracle.maximum)
    output_error=maximum(out.detach().flatten().tolist(),via_weights.detach().flatten().tolist())
    output_budget=2e-11 if q.dtype==torch.float64 else 2e-6
    gradient_budget=2e-10 if q.dtype==torch.float64 else 2e-6
    require(output_error<output_budget,'output equals returned weights times values')
    loss_weights=((torch.arange(out.numel())*3+1)%11-5).reshape(out.shape).to(torch.float64)/8
    out_gradients=torch.autograd.grad((out*loss_weights).sum(),(q,k,v),retain_graph=True)
    weight_gradients=torch.autograd.grad((via_weights*loss_weights).sum(),(q,k,v))
    errors=[maximum(a.flatten().tolist(),b.flatten().tolist()) for a,b in zip(out_gradients,weight_gradients)]
    require(all(error<gradient_budget for error in errors),'returned weights QKV gradient equivalence')
    return {'output_equals_weightsV_error':output_error,'returned_weights_QKV_gradient_max_errors':errors,
            'weightsV_output_budget':output_budget,'weightsV_QKV_gradient_budget':gradient_budget}


def training_and_dtype_move(runner: Cases, mask_class: type[LengthMask], oracle: ModuleType) -> dict[str,object]:
    _,state=runner(torch.float32,[5,2],need_weights=True)
    model,q,k,v,captured,_=state
    require(model.feature_map.redraw==1,'shared feature-state repair #106 required')
    model.double();model.train()
    q=q.double();k=k.double();v=v.double()
    before=model.feature_map.projection_matrix.clone()
    out,weights=model.forward(q,k,v,key_padding_mask=mask_class(torch.tensor([5,2]),max_len=5),
                             need_weights=True,return_attn_unnormalized=True)
    require(not torch.equal(before,model.feature_map.projection_matrix),'training forward did not redraw shared features')
    require(model.feature_map.projection_matrix.dtype==torch.float64,'moved module redrew wrong feature dtype')
    reference=cast(HybridOracle,oracle.oracle)(cast(Values,q.tolist()),cast(Values,k.tolist()),cast(Values,v.tolist()),
        cast(list[list[float]],model.feature_map.projection_matrix.tolist()),cast(Positions,captured['positions']),2,3,[5,2],.5)
    maximum=cast(Callable[[list[float],list[float]],float],oracle.maximum)
    output_error=maximum(out.flatten().tolist(),cast(list[float],reference['output']))
    require(output_error<2e-11,'continued training independent output oracle')
    require(isinstance(weights,tuple),'continued training raw weights')
    assert isinstance(weights,tuple)
    raw_errors=[maximum(actual.flatten().tolist(),cast(list[float],reference[name]))
                for actual,name in zip(weights[:3],('weights','kernel','raw_lowrank'))]
    require(all(error<2e-10 for error in raw_errors),'continued training independent raw oracle')
    return {'actual_training_second_forward':True,'projection_redrawn':True,'moved_feature_dtype':'torch.float64',
            'output_max_error':output_error,'weights_kernel_lowrank_max_errors':raw_errors}


def run(source: Path | None, port: Path, shared: Path) -> dict[str, object]:
    require(torch.get_default_dtype()==torch.float32,'fixture requires ordinary float32 global default')
    torch.set_num_threads(1)
    if source is not None:
        sys.path.insert(0,str(source))
        module = importlib.import_module('src.models.attention.sbsmyrf_attention')
    else:
        sys.path[:0] = [str(port),str(shared)]
        module = importlib.import_module('scatterbrain_later.attention')
    model_class = cast(type['SBSmyrfAttention'],module.SBSmyrfAttention)
    mask_class = cast(type['LengthMask'],importlib.import_module('src.models.modules.masking').LengthMask)
    oracle = oracle_module()
    setattr(oracle,'SBSmyrfAttention',model_class)
    setattr(oracle,'LengthMask',mask_class)
    runner = cast(Cases,oracle.run_case)
    first, state, parity = snapshot(runner,model_class,torch.float32)
    if source is not None:
        return {'status':'frozen_fp32_reference','snapshot':parity,'metric':first}
    finite_controls=cast(Callable[[],dict[str,object]],oracle.finite_guard_controls)()
    # Full support cancels a common duplicate-count error in the ratio. Its raw
    # HV/H1 gate must reject restoration of the historical float32 logarithm.
    full,_ = runner(torch.float64,[5,2],need_weights=True,clusters=(4,6))
    strict_fp64(full); full['full_support']=True
    cases = [full]
    gradients: list[dict[str, object]] = []
    for dtype in (torch.float32,torch.float64):
        for rounds in (1,3):
            for lengths in ([5,5],[5,2],[0,0]):
                for need_weights in (False,True):
                    result,state = runner(dtype,lengths,need_weights=need_weights,rounds=rounds)
                    if dtype==torch.float64:
                        strict_fp64(result)
                    cases.append(result)
                    if lengths==[5,2] and not need_weights:
                        state[0].feature_map.deterministic_eval=True; state[0].eval()
                        # The reused scalar oracle perturbs Python float raw
                        # lists by1e-6 and evaluates in FP64 math; it never
                        # differences rounded FP32 forward evaluations.
                        gradient = cast(Callable[[State],dict[str,object]],oracle.gradients)(state)
                        gradient_budget=2e-8 if dtype==torch.float64 else 2e-6
                        require(finite_metric(gradient,'max_error')<gradient_budget,'conditional QKV gradient')
                        gradients.append({'dtype':str(dtype),'rounds':rounds,'independent_gradient_budget':gradient_budget,
                            'reference_difference_precision':'FP64 scalar raw lists, central step1e-6',
                            **gradient,**weights_value_gradients(state,oracle)})
    zero,_ = runner(torch.float64,[5,2],need_weights=True,rounds=1,features=4,zero_queries_keys=True)
    require(zero['selected_pairs_absent_from_returned_residual_mask']==24,'zero residual support distinction')
    cases.append(zero)
    fp64, state, fp64_snapshot = snapshot(runner,model_class,torch.float64)
    require(fp64_snapshot['alpha_dtype']==fp64_snapshot['beta_dtype']=='torch.float64','actual FP64 hash allocation')
    require(fp64_snapshot['alpha_device']==fp64_snapshot['beta_device']=='cpu','actual CPU hash allocation')
    require(torch.get_default_dtype()==torch.float32,'production changed global default dtype')
    training=training_and_dtype_move(runner,mask_class,oracle)
    model,q,k,v,_,_ = state
    rejected=invalid_domains(model,q,k,v,mask_class)
    invalid_constructor: list[dict[str,object]] = [{'causal':True},{'attention_dropout':.1},{'softmax_eps':1e-6},
        {'n_hashes':0},{'softmax_temp':0.},{'r':float('nan')},{'nb_features':1}]
    for extra in invalid_constructor:
        kwargs: dict[str,object] = dict(n_hashes=3,q_cluster_size=2,k_cluster_size=3,d_head=4,nb_features=8)
        kwargs.update(extra)
        before=torch.get_rng_state().clone()
        # Deliberately invalid runtime API dictionaries are a narrow constructor
        # boundary; source signature remains strict and no Any is introduced.
        constructor=cast(InvalidConstructor,model_class)
        try:constructor(**kwargs)
        except ValueError:rejected+=1
        else:raise AssertionError('invalid constructor accepted')
        require(torch.equal(before,torch.get_rng_state()),'invalid constructor consumed RNG')
    filename=module.__file__
    assert filename is not None
    own = Path(filename).resolve()
    require(own.is_relative_to(port.resolve()),'maintained module substituted')
    imported: dict[str,str] = {}
    for name,item in tuple(sys.modules.items()):
        if name.startswith(('scatterbrain_later','src.models.modules')):
            filename=getattr(item,'__file__',None)
            if filename:
                path=Path(filename).resolve()
                require(path.is_relative_to(port.resolve()) or path.is_relative_to(shared.resolve()),'project import escaped maintained roots')
                imported[name]=str(path)
    return {'status':'maintained_cpu_pass','cases':cases,'gradients':gradients,'fp32_snapshot':parity,
        'fp64_hash_snapshot':fp64_snapshot,'invalid_domain_rejections':rejected,'finite_guard_controls':finite_controls,
        'training_and_dtype_move':training,
        'global_default_dtype':str(torch.get_default_dtype()),'actual_imports':imported,
        'boundary':'actual noncausal CPU float32/float64; captured support/projection only for conditional derivatives; no native/GPU/theorem claim'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--port-root',type=Path,default=ROOT/'ports/scatterbrain-later')
    parser.add_argument('--shared-root',type=Path,default=ROOT/'ports/scatterbrain')
    parser.add_argument('--emit',type=Path,required=True)
    args=parser.parse_args(); args.emit.unlink(missing_ok=True)
    args.emit.parent.mkdir(parents=True,exist_ok=True)
    try:result=run(args.source,args.port_root,args.shared_root)
    except (AssertionError,ValueError,RuntimeError) as error:
        result={'status':'rejected','reason':str(error),'exception':type(error).__name__}
        args.emit.write_text(json.dumps(result,allow_nan=False)+'\n')
        print('SBSMYRF_CPU_REJECTED '+str(error),file=sys.stderr)
        raise SystemExit(2) from None
    args.emit.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('SBSMYRF_CPU_PASS '+str(result['status']))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('SBSMYRF_CPU_RUNTIME_FAILURE '+type(error).__name__,file=sys.stderr)
        raise SystemExit(1) from None
