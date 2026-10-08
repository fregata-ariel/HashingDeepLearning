#!/usr/bin/env python3
"""TRACE_TEST_ID: MAGICPIG-SIMHASH-CPU. execute pinned fill/decode statement blocks on CPU.

The deliberately restricted list tensor adapter is not PyTorch emulation in full:
FP64 arithmetic, IEEE binary16 packing, and exact positive int16 casts only.
No model, CUDA, FlashInfer, randomness, or imports from the archived module.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Any, Callable


def product(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


class Tensor:
    """Minimal row-major tensor for the source operations selected below."""
    def __init__(self, data: list[float], shape: tuple[int, ...]) -> None:
        assert len(data) == product(shape)
        self.data = data
        self.shape = shape

    def index(self, indices: tuple[int, ...]) -> float:
        offset = 0
        for index, size in zip(indices, self.shape):
            offset = offset * size + index
        return self.data[offset]

    def __getitem__(self, key: Any) -> Tensor:
        keys = key if isinstance(key, tuple) else (key,)
        keys = keys + (slice(None),) * (len(self.shape) - len(keys))
        axes = [list(range(n))[k] if isinstance(k, slice) else [k] for n, k in zip(self.shape, keys)]
        shape = tuple(len(a) for a, k in zip(axes, keys) if isinstance(k, slice))
        return Tensor([self.index(tuple(i)) for i in itertools.product(*axes)], shape)

    def reshape(self, *shape: int) -> Tensor:
        values = list(shape)
        if -1 in values:
            values[values.index(-1)] = len(self.data) // math.prod(x for x in values if x != -1)
        return Tensor(list(self.data), tuple(values))

    def transpose(self, a: int, b: int) -> Tensor:
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        data = []
        for pos in itertools.product(*(range(n) for n in shape)):
            old = list(pos)
            old[a], old[b] = old[b], old[a]
            data.append(self.index(tuple(old)))
        return Tensor(data, tuple(shape))

    def contiguous(self) -> Tensor:
        return self

    def float(self) -> Tensor:
        return self

    def int(self) -> Tensor:
        return Tensor([float(int(x)) for x in self.data], self.shape)

    def to(self, dtype: str | None = None, device: str | None = None) -> Tensor:
        if dtype == 'float16':
            return Tensor([struct.unpack('e', struct.pack('e', x))[0] for x in self.data], self.shape)
        if dtype == 'int16':
            if any(not 0 <= x <= 32767 for x in self.data):
                raise OverflowError('outside supported positive int16 range')
            return self.int()
        return self

    def _binary(self, other: Tensor, op: Callable[[float, float], float]) -> Tensor:
        rank = max(len(self.shape), len(other.shape))
        left = (1,) * (rank-len(self.shape)) + self.shape
        right = (1,) * (rank-len(other.shape)) + other.shape
        assert all(a == b or a == 1 or b == 1 for a, b in zip(left, right))
        shape = tuple(max(a,b) for a,b in zip(left,right))
        data = []
        for pos in itertools.product(*(range(n) for n in shape)):
            li = tuple(0 if n == 1 else p for p,n in zip(pos,left))[-len(self.shape):]
            ri = tuple(0 if n == 1 else p for p,n in zip(pos,right))[-len(other.shape):]
            data.append(op(self.index(li),other.index(ri)))
        return Tensor(data,shape)

    def __sub__(self, other: Tensor) -> Tensor:
        return self._binary(other,lambda a,b:a-b)

    def __truediv__(self, other: Tensor) -> Tensor:
        return self._binary(other,lambda a,b:a/b)

    def __gt__(self, value: int) -> Tensor:
        return Tensor([float(x > value) for x in self.data],self.shape)

    def gt(self, value: int) -> Tensor:
        return self > value

    def _reduce(self, dim: int, keepdim: bool, norm: bool) -> Tensor:
        dim %= len(self.shape)
        shape = self.shape[:dim]+self.shape[dim+1:]
        data = []
        for pos in itertools.product(*(range(n) for n in shape)):
            vals = [self.index(pos[:dim]+(j,)+pos[dim:]) for j in range(self.shape[dim])]
            data.append(math.sqrt(sum(x*x for x in vals)) if norm else sum(vals)/len(vals))
        if keepdim:
            shape = shape[:dim]+(1,)+shape[dim:]
        return Tensor(data,shape)

    def mean(self, dim: int, keepdim: bool = False) -> Tensor:
        return self._reduce(dim,keepdim,False)

    def norm(self, p: int, dim: int, keepdim: bool = False) -> Tensor:
        assert p == 2
        return self._reduce(dim,keepdim,True)


class TorchAdapter:
    float16 = 'float16'
    int16 = 'int16'

    @staticmethod
    def Tensor(data: list[int]) -> Tensor:
        return Tensor([float(x) for x in data],(len(data),))

    @staticmethod
    def cat(tensors: list[Tensor], dim: int) -> Tensor:
        assert dim == 0 and all(t.shape[1:] == tensors[0].shape[1:] for t in tensors)
        return Tensor(sum((t.data for t in tensors),[]),(sum(t.shape[0] for t in tensors),)+tensors[0].shape[1:])

    @staticmethod
    def matmul(left: Tensor, right: Tensor) -> Tensor:
        assert len(right.shape) == 2 and left.shape[-1] == right.shape[0]
        width = left.shape[-1]
        data = [sum(left.data[i*width+j]*right.index((j,k)) for j in range(width))
                for i in range(len(left.data)//width) for k in range(right.shape[1])]
        return Tensor(data,left.shape[:-1]+(right.shape[1],))

    @staticmethod
    def mv(left: Tensor, right: Tensor) -> Tensor:
        assert len(left.shape) == 2 and right.shape == (left.shape[1],)
        result = TorchAdapter.matmul(left,right.reshape(-1,1))
        # torch.mv(float16,float16) returns float16, including rounding the sum.
        return result.reshape(left.shape[0]).to(dtype='float16')


def selected_blocks(source: str) -> dict[str,list[ast.stmt]]:
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'LSHSparseAttnServer')
    methods = {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    def target(node: ast.stmt) -> str:
        return ast.unparse(node.targets[0]) if isinstance(node,ast.Assign) else ''
    def span(body: list[ast.stmt], first: str, last: str) -> list[ast.stmt]:
        start = next(i for i,n in enumerate(body) if target(n) == first)
        ends = [i for i,n in enumerate(body) if i >= start and target(n) == last]
        return body[start:(ends[0] if last == 'kn' else ends[-1])+1]
    fill_if = next(n for n in methods['fill'].body if isinstance(n,ast.If))
    decode_if = next(n for n in methods['decode'].body if isinstance(n,ast.If))
    loop = next(n for n in fill_if.orelse if isinstance(n,ast.For))
    blocks = {
        'packing':span(methods['__init__'].body,'self.binary_pack','self.binary_pack'),
        'centering':span(fill_if.orelse,'sink_tokens_key','kn'),
        'fill_hash':span(loop.body,'hash_code','hash_code'),
        'decode_hash':span(decode_if.orelse,'norm_q','q_hashcode'),
        'append_centering':[next(n for n in decode_if.orelse if target(n) == 'key_states')],
    }
    assert {k:len(v) for k,v in blocks.items()} == {'packing':2,'centering':14,'fill_hash':6,'decode_hash':6,'append_centering':1}
    return blocks


def execute(block: list[ast.stmt], env: dict[str,Any]) -> None:
    exec(compile(ast.Module(body=block,type_ignores=[]),'magicpig-selected-block','exec'),{'torch':TorchAdapter,'__builtins__':{'int':int,'range':range}},env)


def attention(query: list[float], keys: list[list[float]], values: list[float]) -> tuple[list[float],float]:
    scores = [sum(a*b for a,b in zip(query,key)) for key in keys]
    peak = max(scores)
    exp = [math.exp(x-peak) for x in scores]
    probabilities = [x/sum(exp) for x in exp]
    return probabilities,sum(p*v for p,v in zip(probabilities,values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--evidence',type=Path)
    args = parser.parse_args()
    path = args.root/'third_party/magicpig/models/attnserver.py'
    source = path.read_text()
    blocks = selected_blocks(source)
    projection = Tensor([1,0,1,1,-1,0,0,1,1,-1,0,1],(2,6))
    obj = SimpleNamespace(K=3,L=2,device='cpu',num_key_value_heads=2,batch_size=2,num_attention_heads=2,head_dim=2)
    env: dict[str,Any] = {'self':obj}
    execute(blocks['packing'],env)
    assert obj.binary_pack.data == [1,2,4]
    # Literal expected tables: strict >0 makes zero projections false; bits little-endian.
    raw = Tensor([1,0,0,1,-1,0,0,-1],(2,2,2))
    env.update(offload_key=raw,start=0,end=2)
    obj.hash_func = projection
    execute(blocks['fill_hash'],env)
    assert env['hash_code'].shape == (2,2,2)
    assert env['hash_code'].data == [5,6,1,4,0,0,2,1]
    fill_literal = list(env['hash_code'].data)
    execute(blocks['fill_hash'],env)
    assert env['hash_code'].data == fill_literal
    env.update(start=1,end=2)
    execute(blocks['fill_hash'],env)
    assert env['hash_code'].data == [6,4,0,1]
    env.update(start=0,end=2)
    env['query_states'] = raw.reshape(2,2,1,2)
    execute(blocks['decode_hash'],env)
    assert env['q_hashcode'].shape == (4,2)
    assert env['q_hashcode'].data == [5,1,6,4,0,2,0,1]
    execute(blocks['decode_hash'],env)
    assert env['q_hashcode'].data == [5,1,6,4,0,2,0,1]
    # K/L alternative distinguishes table width and grouping; same literal projection.
    obj.K,obj.L = 2,3
    execute(blocks['packing'],env)
    env.update(offload_key=raw,start=0,end=2)
    execute(blocks['fill_hash'],env)
    assert env['hash_code'].data == [1,2,3,1,0,2,0,0,0,2,1,0]
    execute(blocks['decode_hash'],env)
    assert env['q_hashcode'].data == [1,3,0,2,1,2,0,0,1,0,2,0]
    # Center all partitions using the offloaded mean, not a separate sink/local mean.
    obj.num_sink_tokens,obj.num_local_tokens = 1,1
    cache = Tensor([2,1, 4,-1, 4,3, 2,1, 8,1, 6,5, 10,-1, 8,3],(4,2,2))
    env.update(key_cache=cache,value_cache=cache,seq_len=4)
    execute(blocks['centering'],env)
    assert env['avg_k'].data == [6,2,4,3]
    assert env['offload_key'].data == [-2,1,2,-1,-2,-2,2,2]
    assert env['key'].data == [-4,-1,4,-3,0,-4,4,0]
    # Stored request means are supplied explicitly; actual cache assignment is outside the selected block.
    obj.avg_k = [Tensor(env['avg_k'].data+[1,-2,-3,4],(2,2,1,2))]
    env.update(key_states=Tensor([12,2,7,6,14,4,8,7],(2,2,1,2)),layer_idx=0)
    execute(blocks['append_centering'],env)
    assert env['key_states'].data == [6,0,3,3,13,6,11,3]
    max_error = 0.0
    for head in range(2):
        mean = env['avg_k'].data[head*2:head*2+2]
        keys = [cache.data[(t*2+head)*2:(t*2+head+1)*2] for t in range(4)]
        keys.append([12,2] if head == 0 else [7,6])
        shifted = [[x-m for x,m in zip(k,mean)] for k in keys]
        a = attention([.25,-.5],keys,[1,-2,3,4,-1])
        b = attention([.25,-.5],shifted,[1,-2,3,4,-1])
        max_error = max(max_error,max(abs(x-y) for x,y in zip(a[0],b[0])),abs(a[1]-b[1]))
    assert max_error < 1e-12
    # Characterize exact range without relying on undefined/out-of-range int casts.
    exact = []
    losses = []
    for k in range(1,16):
        obj.K = k
        execute(blocks['packing'],env)
        env.update(offload_key=Tensor([1],(1,1,1)),start=0,end=1)
        obj.num_key_value_heads,obj.L = 1,1
        obj.hash_func = Tensor([1]*k,(1,k))
        if k <= 11:
            execute(blocks['fill_hash'],env)
            assert env['hash_code'].data == [float((1<<k)-1)]
            # Exhaust all bit codes via the actual source fill hash block.
            width = 1 << k
            env['offload_key'] = Tensor([1.0 if code & (1 << bit) else -1.0 for code in range(width) for bit in range(k)],(1,width,k))
            obj.hash_func = Tensor([float(i == j) for i in range(k) for j in range(k)],(k,k))
            env['end'] = width
            execute(blocks['fill_hash'],env)
            assert env['hash_code'].data == list(range(width))
            exact.append(k)
        else:
            packed = TorchAdapter.mv(Tensor([1]*k,(1,k)),obj.binary_pack).data[0]
            assert packed != (1<<k)-1
            losses.append({'K':k,'exact':(1<<k)-1,'binary16':packed,'int16_safe':packed <= 32767})
    result = {'id':'MAGICPIG-SIMHASH-CPU','name':'simhash','status':'passed','source':str(path.relative_to(args.root)),
              'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'selected_blocks':{},
              'hash_ordering_cases':4,'centering_heads':2,'append_batches':2,'attention_max_error':max_error,
              'packing_exact_K':exact,'packing_exact_code_checks':sum(1 << k for k in exact),'packing_loss_examples':losses,'gpu_execution':False,'torch_runtime':False,
              'adapter':'restricted FP64 list tensor; binary16 sum packing; positive int16 only'}
    for name,nodes in blocks.items():
        text = '\n'.join(ast.get_source_segment(source,n) or '' for n in nodes)
        result['selected_blocks'][name] = {'start_line':nodes[0].lineno,'end_line':nodes[-1].end_lineno,'sha256':hashlib.sha256(text.encode()).hexdigest()}
    print(json.dumps(result,sort_keys=True))
    if args.evidence:
        args.evidence.parent.mkdir(parents=True,exist_ok=True)
        args.evidence.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
