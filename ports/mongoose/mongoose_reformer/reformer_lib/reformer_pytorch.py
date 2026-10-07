import math
import torch
import torch.nn as nn
from torch.nn import Identity
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce, wraps
from itertools import chain
from operator import mul
from collections.abc import Sequence
from typing import Callable, Protocol, TypeAlias, TypeVar, cast

from local_attention import LocalAttention
from axial_positional_embedding import AxialPositionalEmbedding
from product_key_memory import PKM
from .reversible import ReversibleSequence
from .scheduler import Scheduler

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_

# constants
TOKEN_SELF_ATTN_VALUE = -5e4  # carefully set for half precision to work

_DefaultT = TypeVar("_DefaultT")
_CacheT = TypeVar("_CacheT")
_TupleT = TypeVar("_TupleT")
AttentionResult: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]


class AttentionChunkFn(Protocol):
    def __call__(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> AttentionResult: ...


class _LocalAttentionCallable(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        input_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...



# helper fns

def sort_key_val(
    t1: torch.Tensor, t2: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values, indices = t1.sort(dim=dim)
    expanded = t2.expand_as(t1)
    return values, expanded.gather(dim, indices), indices


def batched_index_select(
    values: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    last_dim = values.shape[-1]
    return values.gather(
        1, indices[:, :, None].expand(-1, -1, last_dim)
    )


def mine_triplet_examples(
        qk: torch.Tensor,
        attention_probs: torch.Tensor,
        candidate_indices: torch.Tensor,
        sorted_query_indices: torch.Tensor,
        negative_samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select released-code positive/negative examples for learnable LSH.

    Positive examples use the largest post-mask/post-softmax attention
    probability. Negative indices are supplied from the raw pre-mask dot-product
    argmin computed earlier in LSHAttention.forward.

    This is the released Reformer-side mining rule associated with MONGOOSE
    Section 3.3; it is an implementation choice around the paper's
    positive/negative training examples.

    Shape / dtype contract:
        qk is floating (batch*heads, sequence, dim_head). attention_probs and
        candidate_indices describe the same hashed candidate axis; the index
        tensors are integer-valued. Returned positive/negative vectors preserve
        qk's leading/indexed dimensions and dim_head and are detached.

    Traceability:
        MONGOOSE-TRIPLET-MINING.
    """
    positive_samples = attention_probs.argmax(dim=-1)
    positive_indices = torch.gather(
        candidate_indices, -1, positive_samples
    ).view_as(sorted_query_indices)
    negative_indices = torch.gather(
        candidate_indices, -1, negative_samples
    ).view_as(sorted_query_indices)
    positive_vectors = batched_index_select(qk, positive_indices).detach()
    negative_vectors = batched_index_select(qk, negative_indices).detach()
    return positive_vectors, negative_vectors


def process_inputs_chunk(
    fn: AttentionChunkFn, chunks: int = 1, dim: int = 0
) -> AttentionChunkFn:
    """Chunk tensor arguments, execute one attention callable, and concatenate.

    The callable must return the six-tensor/optional-tensor AttentionResult
    contract shared by LSHAttention and FullQKAttention.
    """

    def inner_fn(
        *args: torch.Tensor, **kwargs: torch.Tensor
    ) -> AttentionResult:
        keys = tuple(kwargs.keys())
        values = tuple(kwargs.values())
        len_args = len(args)
        chunked_args = list(
            zip(
                *(
                    tensor.chunk(chunks, dim=dim)
                    for tensor in (*args, *values)
                )
            )
        )
        call_args = [
            (
                chunk[:len_args],
                dict(zip(keys, chunk[len_args:])),
            )
            for chunk in chunked_args
        ]
        outputs = [
            fn(*positional, **keyword)
            for positional, keyword in call_args
        ]

        def cat_fn(
            pieces: tuple[torch.Tensor | None, ...],
        ) -> torch.Tensor | None:
            if pieces[0] is None:
                return None
            tensors = [piece for piece in pieces if piece is not None]
            return torch.cat(tensors, dim=dim)

        columns = list(zip(*outputs))
        merged = tuple(cat_fn(tuple(column)) for column in columns)
        return cast(AttentionResult, merged)

    return inner_fn


def chunked_sum(
    tensor: torch.Tensor, chunks: int = 1
) -> torch.Tensor:
    *orig_size, last_dim = tensor.shape
    flattened = tensor.reshape(-1, last_dim)
    summed_tensors = [
        chunk.sum(dim=-1)
        for chunk in flattened.chunk(chunks, dim=0)
    ]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)


def default(
    val: _DefaultT | None, default_val: _DefaultT
) -> _DefaultT:
    return default_val if val is None else val


def cast_tuple(
    x: _TupleT | tuple[_TupleT, ...],
) -> tuple[_TupleT, ...]:
    return x if isinstance(x, tuple) else (x,)


def max_neg_value(tensor: torch.Tensor) -> float:
    return -torch.finfo(tensor.dtype).max


def cache_fn(
    factory: Callable[[], _CacheT],
) -> Callable[[], _CacheT]:
    cache: _CacheT | None = None

    @wraps(factory)
    def cached_fn() -> _CacheT:
        nonlocal cache
        if cache is None:
            cache = factory()
        return cache

    return cached_fn


def cosine_similarity(x1, x2, dim=1, eps=1e-6):
    r"""Returns cosine similarity between x1 and x2, computed along dim.

    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero. Default: 1e-8

    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.
    """
    w1 = torch.norm(x1 + eps, 2, dim, keepdim=True)
    w2 = torch.norm(x2 + eps, 2, dim, keepdim=True)
    x1 /= w1.clamp(min=eps)
    x2 /= w2.clamp(min=eps)
    w12 = torch.sum(x1 * x2, dim)
    return w12.squeeze()


def cache_method_decorator(cache_attr, cache_namespace, reexecute=False):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, key_namespace=None, fetch=False, set_cache=True, **kwargs):
            namespace_str = str(default(key_namespace, ''))
            _cache = getattr(self, cache_attr)
            _keyname = f'{cache_namespace}:{namespace_str}'

            if fetch:
                val = _cache[_keyname]
                if reexecute:
                    fn(self, *args, **kwargs)
            else:
                val = fn(self, *args, **kwargs)
                if set_cache:
                    setattr(self, cache_attr, {**_cache, **{_keyname: val}})
            return val

        return wrapper

    return inner_fn


def expand_dim(
    dim: int, k: int, tensor: torch.Tensor
) -> torch.Tensor:
    expanded = tensor.unsqueeze(dim)
    expand_shape = [-1] * len(expanded.shape)
    expand_shape[dim] = k
    return expanded.expand(*expand_shape)


def merge_dims(
    ind_from: int, ind_to: int, tensor: torch.Tensor
) -> torch.Tensor:
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def split_at_index(
    dim: int, index: int, tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pre_slices = (slice(None),) * dim
    left = (*pre_slices, slice(None, index))
    right = (*pre_slices, slice(index, None))
    return tensor[left], tensor[right]


# helper classes

class MatrixMultiply(nn.Module):
    def __init__(
        self,
        tensor: torch.Tensor,
        transpose: bool = False,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.tensor = tensor
        self.transpose = transpose
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor = self.tensor
        if self.normalize:
            tensor = F.normalize(tensor, dim=-1)
        if self.transpose:
            tensor = tensor.t()
        return x @ tensor


class ReZero(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.fn(x, **kwargs)) * self.g


class ScaleNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = torch.norm(
            x, dim=-1, keepdim=True
        ).clamp(min=self.eps)
        return x / n * self.g


class PreNorm(nn.Module):
    def __init__(
        self,
        norm_class: Callable[[int], nn.Module],
        dim: int,
        fn: nn.Module,
    ) -> None:
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        normalized = cast(torch.Tensor, self.norm(x))
        return cast(torch.Tensor, self.fn(normalized, **kwargs))


class Chunk(nn.Module):
    def __init__(
        self, chunks: int, fn: nn.Module, along_dim: int = -1
    ) -> None:
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        if self.chunks == 1:
            return cast(torch.Tensor, self.fn(x, **kwargs))
        chunks = x.chunk(self.chunks, dim=self.dim)
        outputs = [
            cast(torch.Tensor, self.fn(chunk, **kwargs))
            for chunk in chunks
        ]
        return torch.cat(outputs, dim=self.dim)


# LSH attention as described in https://openreview.net/pdf?id=rkgNKkHtvB
# adapted from trax, stripped to what paper said needed to work
# namely that buckets need to be at least 64 with 8 rounds of hashing
# https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py#L442

# +
class LSHAttention(nn.Module):
    """Base Reformer locality-sensitive-hash attention.

    Shape contract:
        qk, v: (batch*heads, sequence, dim_head)
        hash_vectors result: (batch*heads, n_hashes*sequence)
        forward result: output plus attention/bucket/example tensors.

    This is a Reformer prerequisite used by MONGOOSE; learnable rotations are
    supplied by TripletLSHAttention.
    """

    def __init__(
        self,
        dropout: float = 0.0,
        bucket_size: int = 64,
        n_hashes: int = 8,
        causal: bool = False,
        allow_duplicate_attention: bool = True,
        attend_across_buckets: bool = True,
        rehash_each_round: bool = True,
        drop_for_hash_rate: float = 0.0,
        random_rotations_per_head: bool = False,
        return_attn: bool = False,
        store_stats: bool = False,
    ) -> None:
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        #         self.rotations = nn.Linear(64, 128, bias=False)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache: dict[str, torch.Tensor] = {}

        self.store_stats = store_stats
        self.mean_dp = 0.0
        self.stat_count = 0

    # @cache_method_decorator('_cache', 'buckets', reexecute=True)
    def hash_vectors(
        self,
        n_buckets: int,
        vecs: torch.Tensor,
        rotations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hash vectors into round-offset bucket ids.

        vecs has shape (batch*heads, sequence, dim_head). rotations, when
        supplied by MONGOOSE, has shape
        (batch*heads, dim_head, n_hashes, n_buckets/2 before sign expansion).
        The returned int64 tensor has shape
        (batch*heads, n_hashes*sequence). vecs/rotations are floating tensors.

        TRACE_TEST_ID: MONGOOSE-REFORMER-HASH-SHAPES.
        """
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        # add rotations
        # random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        if rotations is None:
            random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1,
                                                                                                    -1)
        else:
            if rotations.size(-1) == rotations_shape[-1]:
                random_rotations = rotations
            elif rotations.size(-1) < rotations_shape[-1]:
                complement_shape = (
                    batch_size if self._random_rotations_per_head else 1,
                    vecs.shape[-1],
                    self.n_hashes if self._rehash_each_round else 1,
                    rot_size // 2 - rotations.size(-1))

                tmp = torch.randn(complement_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)
                random_rotations = torch.cat([rotations, tmp], dim=-1)
            else:
                random_rotations = rotations[:, :, :, torch.randperm(rotations.size(-1))[:rotations_shape[-1]]]

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = torch.arange(self.n_hashes, device=device)
            offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
            buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 0)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)

            _, buckets, _ = sort_key_val(
                rotated_vecs, bucket_range, dim=-1
            )
            buckets = buckets[..., -self.n_hashes:]
            buckets = torch.reshape(buckets, (batch_size, -1))

        return buckets

    def forward(
        self,
        qk: torch.Tensor,
        v: torch.Tensor,
        query_len: int | None = None,
        input_mask: torch.Tensor | None = None,
        input_attn_mask: torch.Tensor | None = None,
        rotations: torch.Tensor | None = None,
        triplet_examples: bool = False,
        **kwargs: object,
    ) -> AttentionResult:
        """Run LSH attention on merged batch/head tensors.

        qk and v are floating tensors shaped
        (batch*heads, sequence, dim_head). input_mask and input_attn_mask are
        boolean masks. Boolean masks are broadcast over the hashed attention
        blocks. When triplet_examples is true, returned positive/negative
        tensors are detached mining examples; otherwise those two return slots
        are None.
        """
        batch_size, seqlen, dim, device = *qk.shape, qk.device

        query_len = default(query_len, seqlen)
        kwargs.pop('_reverse', False)
        kwargs.pop('_depth', None)

        assert seqlen % (
                self.bucket_size * 2) == 0, f'Sequence length ({seqlen}) needs to be divisible by target bucket size  x 2 - {self.bucket_size * 2}'

        n_buckets = seqlen // self.bucket_size
        # buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training)

        # add customized rotations
        # buckets = self.hash_vectors(n_buckets, qk, key_namespace=depth, fetch=is_reverse, set_cache=self.training, rotations=rotations)
        buckets = self.hash_vectors(n_buckets, qk, rotations=rotations)

        max_idxs = buckets.reshape(batch_size, -1, seqlen)

        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        total_hashes = self.n_hashes

        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach()

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker, indices = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x: torch.Tensor) -> torch.Tensor:
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        if triplet_examples:
            min_samples = dots.argmin(dim=-1).detach()
        masked_value = max_neg_value(dots)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask,
                                    (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]),
                                    value=True)
            dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > query_len:
                mask = mask & (bkv_t[:, :, None, :] < query_len)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        with torch.no_grad():
            if self.store_stats:
                computed_mean = dots.detach()
                mean = torch.mean(computed_mean[computed_mean > TOKEN_SELF_ATTN_VALUE + 1])
                self.mean_dp = mean.item()

                self.stat_count += 1

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)

        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        # compute pos/neg examples
        if triplet_examples:
            with torch.no_grad():
                pos_vectors, neg_vectors = mine_triplet_examples(
                    qk, dots, bkv_t, st, min_samples
                )
        else:
            pos_vectors = None
            neg_vectors = None

        # unsort logits
        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)

        o = torch.reshape(o, (batch_size, total_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, total_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * total_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * total_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, total_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        # return output, attention matrix, and bucket distribution
        return out, attn, buckets, sqk.detach(), pos_vectors, neg_vectors


# customized training for hash functions

class TripletLSHAttention(LSHAttention):
    """Reformer attention with trainable LSH rotations.

    Paper mapping:
        MONGOOSE (ICLR 2021), Section 3.3 and Section 3.3.1 "Learnable LSH",
        especially Equation 3 and Algorithm 2.

    The base LSH attention supplies current query/key neighborhoods. This class
    parameterizes the hash rotations so positive/negative examples mined from
    those neighborhoods can update the hash function during training.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        dim: int = 512,
        seq_len: int = 1024,
        heads: int = 8,
        dropout: float = 0.0,
        bucket_size: int = 64,
        n_hashes: int = 8,
        causal: bool = False,
        allow_duplicate_attention: bool = True,
        attend_across_buckets: bool = True,
        rehash_each_round: bool = True,
        drop_for_hash_rate: float = 0.0,
        random_rotations_per_head: bool = False,
        return_attn: bool = False,
        triplet_chunks: int | None = None,
        store_stats: bool = False,
    ) -> None:
        super().__init__(dropout=dropout,
                         bucket_size=bucket_size,
                         n_hashes=n_hashes,
                         causal=causal,
                         allow_duplicate_attention=allow_duplicate_attention,
                         attend_across_buckets=attend_across_buckets,
                         rehash_each_round=rehash_each_round,
                         drop_for_hash_rate=drop_for_hash_rate,
                         random_rotations_per_head=random_rotations_per_head,
                         return_attn=return_attn,
                         store_stats=store_stats)
        self.alpha = alpha
        self.seq_len = seq_len
        self.heads = heads
        n_buckets = self.seq_len // bucket_size
        # buckets_dim = n_buckets // 2
        buckets_dim = n_buckets
        if self._rehash_each_round:
            buckets_dim *= n_hashes
        self.rotations = nn.Linear(dim // self.heads, buckets_dim, bias=False)

        # number of chunks to split up computation of pos/neg examples for triplet loss
        self.triplet_chunks = default(triplet_chunks, dim)

    def reset_rotations(self) -> None:
        self.rotations.reset_parameters()

    def extract_rotations(self, batch_size: int) -> torch.Tensor:
        """Return detached learned rotations for Reformer hashing.

        Shape: (batch*heads, dim_head, n_hashes, n_buckets).
        TRACE_TEST_ID: MONGOOSE-REFORMER-HASH-SHAPES.
        """
        n_buckets = self.seq_len // self.bucket_size
        rotations = self.rotations.weight.t().detach()  # dim x (buckets * n_hashes / 2)
        # rotations = rotations[:, torch.randperm(rotations.size(-1))[:rotations.size(-1) // 2]]
        rotations = torch.reshape(rotations, (-1, self.n_hashes, n_buckets))
        rotations = rotations.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return rotations

    def triplet_forward(
            self,
            x: torch.Tensor,
            p: torch.Tensor,
            n: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the MONGOOSE learnable-LSH triplet objective.

        Paper mapping:
            Section 3.3.1, Equation 3. Positive examples should have greater
            similarity under the learned hash projection than negative
            examples by the configured margin alpha.

        Traceability test:\n            MONGOOSE-TRIPLET-LOSS.\n\n        Implementation note:\n            x, p, and n are detached from the main model so this loss updates
            the rotation/hash parameters rather than backpropagating through
            the example-mining path.
        """
        x = x.detach()
        p = p.detach()
        n = n.detach()
        emb_x = self.rotations(x)
        emb_p = self.rotations(p)
        emb_n = self.rotations(n)

        # cosine similarity
        sim_xp = F.cosine_similarity(emb_x, emb_p, dim=-1, eps=1e-6)
        sim_xn = F.cosine_similarity(emb_x, emb_n, dim=-1, eps=1e-6)

        # distance in radians
        # dis_xp = 1 - torch.acos(sim_xp)/pi
        # dis_xn = 1 - torch.acos(sim_xn)/pi
        dis_xp = 1 - sim_xp
        dis_xn = 1 - sim_xn
        triplet_loss = dis_xp - dis_xn + self.alpha

        triplet_loss = torch.mean(torch.max(triplet_loss,
                                            torch.zeros(triplet_loss.size()).to(x.device)))
        if torch.isnan(triplet_loss):
            print("nan!")

        return triplet_loss

    def forward(
        self,
        qk: torch.Tensor,
        v: torch.Tensor,
        query_len: int | None = None,
        input_mask: torch.Tensor | None = None,
        input_attn_mask: torch.Tensor | None = None,
        rotations: torch.Tensor | None = None,
        triplet_examples: bool = False,
        **kwargs: object,
    ) -> AttentionResult:
        batch_size, _seqlen, _dim = qk.shape
        del rotations
        learned_rotations = self.extract_rotations(batch_size)
        out, attn, buckets, emb_x, pos, neg = super().forward(
            qk,
            v,
            query_len=query_len,
            input_mask=input_mask,
            input_attn_mask=input_attn_mask,
            rotations=learned_rotations,
            triplet_examples=triplet_examples,
            **kwargs,
        )
        return out, attn, buckets, emb_x, pos, neg


# simple full attention
class FullQKAttention(nn.Module):
    """Dense attention fallback sharing the AttentionResult contract."""

    def __init__(self, causal: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.attn: torch.Tensor | None = None

    def forward(
        self,
        qk: torch.Tensor,
        v: torch.Tensor,
        query_len: int | None = None,
        input_mask: torch.Tensor | None = None,
        input_attn_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> AttentionResult:
        """Run dense attention on (batch*heads, sequence, dim_head) tensors.

        qk/v are floating tensors; input masks are boolean tensors. The return
        tuple uses the same six-slot AttentionResult contract as LSHAttention.
        """
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type_as(q)

        dot = torch.einsum('bie,bje->bij', q, qk) * (dim ** -0.5)

        # qk attention requires tokens not attend to self
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, 0:query_len, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), value=True)
            dot.masked_fill_(~mask, masked_value)

        # Mask for post qk attention logits of the input sequence
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seq_len - input_attn_mask.shape[-1]), value=True)
            dot.masked_fill_(~input_attn_mask, masked_value)

        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, i, j] = masked_value

        dot = dot.softmax(dim=-1)
        self.attn = dot.detach()
        dot = self.dropout(dot)

        out = torch.einsum('bij,bje->bie', dot, v)

        return out, dot, torch.empty(0, device=qk.device), None, None, None


class LSHSelfAttention(nn.Module):
    """LSH attention wrapper that couples MONGOOSE scheduling and hash learning.

    Paper mapping:
        MONGOOSE Section 3.2 (adaptive update scheduling) and Section 3.3
        (learnable parameterized LSH).

    When attn_type == 'triplet', the module owns both a Scheduler and a
    TripletLSHAttention instance. The forward path asks the scheduler whether
    the model has changed enough before mining examples and accumulating a
    triplet loss.

    Implementation distinction:
        The released Scheduler used here is a compact packed-code
        absolute-difference trigger. It realizes the paper's "avoid expensive
        updates while parameters change slowly" control-flow goal, but it is
        not the full dynamic-maintenance data structure of Algorithm 1.

    Traceability:
        MONGOOSE-SCHEDULER-CHANGE, MONGOOSE-SCHEDULER-GATE.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        bucket_size: int = 64,
        n_hashes: int = 8,
        causal: bool = False,
        dim_head: int | None = None,
        attn_chunks: int | None = 1,
        random_rotations_per_head: bool = False,
        attend_across_buckets: bool = True,
        allow_duplicate_attention: bool = True,
        num_mem_kv: int = 0,
        one_value_head: bool = False,
        use_full_attn: bool = False,
        full_attn_thres: int | None = None,
        return_attn: bool = False,
        post_attn_dropout: float = 0.0,
        dropout: float = 0.0,
        n_local_attn_heads: int = 0,
        attn_type: str = "lsh",
        max_seq_len: int | None = None,
        alpha: float = 1.0,
        triplet_chunks: int | None = None,
        scheduler_hashes: int = 10,
        thresh: float = 0.01,
        store_stats: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'dimensions must be divisible by number of heads'
        assert n_local_attn_heads < heads, 'local attention heads must be less than number of heads'

        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported LSHSelfAttention options: {unknown}")
        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.attn_chunks: int = default(attn_chunks, 1)

        self.v_head_repeats = (heads if one_value_head else 1)
        v_dim = dim_heads // self.v_head_repeats

        self.toqk = nn.Linear(dim, dim_heads, bias=False)
        self.tov = nn.Linear(dim, v_dim, bias=False)
        self.to_out = nn.Linear(dim_heads, dim)

        self.bucket_size = bucket_size

        self.attn_type = attn_type
        self.scheduler: Scheduler | None = None
        self.lsh_attn: LSHAttention
        if self.attn_type == 'triplet':
            if max_seq_len is None:
                raise ValueError("triplet attention requires max_seq_len")
            self.lsh_attn = TripletLSHAttention(alpha=alpha, dim=self.dim, seq_len=max_seq_len, heads=self.heads,
                                                bucket_size=bucket_size, n_hashes=n_hashes, causal=causal,
                                                random_rotations_per_head=random_rotations_per_head,
                                                attend_across_buckets=attend_across_buckets,
                                                allow_duplicate_attention=allow_duplicate_attention,
                                                return_attn=return_attn, triplet_chunks=triplet_chunks,
                                                store_stats=store_stats)
            # init scheduler
            self.scheduler = Scheduler(self.toqk.weight, dim, scheduler_hashes, 1, thresh)

        else:
            self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal,
                                         random_rotations_per_head=random_rotations_per_head,
                                         attend_across_buckets=attend_across_buckets,
                                         allow_duplicate_attention=allow_duplicate_attention, return_attn=return_attn,
                                         dropout=dropout, store_stats=store_stats)

        self.full_attn = FullQKAttention(causal=causal, dropout=dropout)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)

        self.use_full_attn = use_full_attn
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True)) if num_mem_kv > 0 else None

        self.n_local_attn_heads = n_local_attn_heads
        self.local_attn = cast(
            _LocalAttentionCallable,
            LocalAttention(
                window_size=bucket_size * 2,
                causal=causal,
                dropout=dropout,
                shared_qk=True,
                look_forward=(1 if not causal else 0),
            ),
        )

        self.callback: Callable[[torch.Tensor, torch.Tensor], None] | None = None
        self.attn: torch.Tensor | None = None
        self.triplet_loss: float | torch.Tensor | None = 0.0
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # if self._qkv_same_embed_dim:
        xavier_uniform_(self.toqk.weight)
        xavier_uniform_(self.tov.weight)
        xavier_uniform_(self.to_out.weight)

    def forward(
        self,
        x: torch.Tensor,
        keys: torch.Tensor | None = None,
        input_mask: torch.Tensor | None = None,
        input_attn_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        calc_triplet: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run attention and optionally refresh the learnable-LSH training signal.

        x is a floating tensor shaped (batch, sequence, dim). keys, when
        supplied, is (batch, context, dim). input/context masks are boolean;
        input_attn_mask is a boolean attention matrix.

        If calc_triplet is requested, Scheduler.detect_change() first applies
        the inexpensive change test. Only a positive trigger allows triplet
        example mining and accumulation of the learned-hash loss, matching the
        Section 3.2 -> Section 3.3 control flow described by MONGOOSE.
        """
        del kwargs
        device, dtype = x.device, x.dtype
        b, t, e, h, _dh, m, l_h = (
            *x.shape,
            self.heads,
            self.dim_head,
            self.num_mem_kv,
            self.n_local_attn_heads,
        )

        mem_kv = default(self.mem_kv, torch.empty(b, 0, e, dtype=dtype, device=device))
        mem = mem_kv.expand(-1, m, -1)

        keys = default(keys, torch.empty(b, 0, e, dtype=dtype, device=device))
        c = keys.shape[1]

        kv_len = t + m + c
        use_full_attn = self.use_full_attn or kv_len <= self.full_attn_thres

        x = torch.cat((x, mem, keys), dim=1)
        qk = self.toqk(x)
        v = self.tov(x)
        v = v.repeat(1, 1, self.v_head_repeats)

        def merge_heads(v: torch.Tensor) -> torch.Tensor:
            return v.view(b, kv_len, h, -1).transpose(1, 2)

        def split_heads(v: torch.Tensor) -> torch.Tensor:
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        merge_batch_and_heads = partial(merge_dims, 0, 1)

        qk, v = map(merge_heads, (qk, v))

        has_local = l_h > 0
        lsh_h = h - l_h

        split_index_fn = partial(split_at_index, 1, l_h)
        (lqk, qk), (lv, v) = map(split_index_fn, (qk, v))
        lqk, qk, lv, v = map(merge_batch_and_heads, (lqk, qk, lv, v))

        masks: dict[str, torch.Tensor] = {}
        if input_mask is not None or context_mask is not None:
            default_mask = torch.tensor([True], device=device)
            i_mask = default(input_mask, default_mask.expand(b, t))
            m_mask = default_mask.expand(b, m)
            c_mask = default(context_mask, default_mask.expand(b, c))
            mask = torch.cat((i_mask, m_mask, c_mask), dim=1)
            mask = merge_batch_and_heads(expand_dim(1, lsh_h, mask))
            masks['input_mask'] = mask

        if input_attn_mask is not None:
            input_attn_mask = merge_batch_and_heads(expand_dim(1, lsh_h, input_attn_mask))
            masks['input_attn_mask'] = input_attn_mask

        if calc_triplet:
            calc_triplet = (
                self.scheduler is not None
                and self.scheduler.detect_change(self.toqk.weight)
            )
        return_triplet_examples = (
            self.attn_type == "triplet"
            and calc_triplet
            and not use_full_attn
        )
        if use_full_attn:
            partial_attn_fn = cast(
                AttentionChunkFn,
                partial(
                    self.full_attn.forward,
                    query_len=t,
                    input_mask=input_mask,
                    triplet_examples=return_triplet_examples,
                ),
            )
        else:
            partial_attn_fn = cast(
                AttentionChunkFn,
                partial(
                    self.lsh_attn.forward,
                    query_len=t,
                    input_mask=input_mask,
                    triplet_examples=return_triplet_examples,
                ),
            )

        attn_fn_in_chunks = process_inputs_chunk(
            partial_attn_fn, chunks=self.attn_chunks
        )
        out, attn, buckets, emb_x, pos, neg = attn_fn_in_chunks(qk, v, **masks)

        if self.callback is not None:
            self.callback(attn.reshape(b, lsh_h, t, -1), buckets.reshape(b, lsh_h, -1))

        if return_triplet_examples:
            triplet_attention = cast(TripletLSHAttention, self.lsh_attn)
            assert emb_x is not None and pos is not None and neg is not None

            def chunked_loss(
                fn: Callable[
                    [torch.Tensor, torch.Tensor, torch.Tensor],
                    torch.Tensor,
                ],
                x_arg: torch.Tensor,
                p_arg: torch.Tensor,
                n_arg: torch.Tensor,
                chunks: int = 1,
                dim: int = 0,
            ) -> torch.Tensor:
                chunked_inputs = [
                    arg.chunk(chunks, dim=dim)
                    for arg in (x_arg, p_arg, n_arg)
                ]
                outputs = [
                    fn(*inputs) for inputs in zip(*chunked_inputs)
                ]
                result = outputs[0]
                for output in outputs[1:]:
                    result = result + output
                return result

            triplet_loss = chunked_loss(
                triplet_attention.triplet_forward,
                emb_x,
                pos,
                neg,
                chunks=self.attn_chunks,
                dim=1,
            )

            if self.triplet_loss is None or isinstance(
                self.triplet_loss, float
            ):
                self.triplet_loss = triplet_loss
            else:
                self.triplet_loss = self.triplet_loss + triplet_loss

        if has_local:
            lqk, lv = lqk[:, :t], lv[:, :t]
            local_out = self.local_attn(lqk, lqk, lv, input_mask=input_mask)
            local_out = local_out.reshape(b, l_h, t, -1)
            out = out.reshape(b, lsh_h, t, -1)
            out = torch.cat((local_out, out), dim=1)

        out = split_heads(out).view(b, t, -1)

        self.attn = out.detach()

        out = self.to_out(out)
        return cast(torch.Tensor, self.post_attn_dropout(out))


# feed forward
class GELU_(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1
            + torch.tanh(
                math.sqrt(2 / math.pi)
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


GELU: type[nn.Module] = nn.GELU if hasattr(nn, "GELU") else GELU_


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
        activation: type[nn.Module] | None = None,
        glu: bool = False,
    ) -> None:
        super().__init__()
        activation_type = default(activation, GELU)
        self.glu = glu
        self.w1 = nn.Linear(
            dim, dim * mult * (2 if glu else 1)
        )
        self.act = activation_type()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        del kwargs
        if not self.glu:
            x = self.w1(x)
            x = cast(torch.Tensor, self.act(x))
        else:
            x, value = self.w1(x).chunk(2, dim=-1)
            x = cast(torch.Tensor, self.act(x)) * value
        x = self.dropout(x)
        return self.w2(x)


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            x.shape[1], device=x.device
        )
        return self.emb(positions)


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_freq = cast(
            torch.Tensor, getattr(self, "inv_freq")
        )
        positions = torch.arange(
            x.shape[1], device=x.device
        ).type_as(inv_freq)
        sinusoid = torch.einsum(
            "i,j->ij", positions, inv_freq
        )
        emb = torch.cat(
            (sinusoid.sin(), sinusoid.cos()), dim=-1
        )
        return emb[None, :, :]


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            "pe", pe.unsqueeze(0).transpose(0, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = cast(torch.Tensor, getattr(self, "pe"))
        return cast(
            torch.Tensor,
            self.dropout(x + pe[: x.size(0), :]),
        )


def _wrap_residual(
    module: nn.Module,
    *,
    use_rezero: bool,
    norm_type: Callable[[int], nn.Module],
    dim: int,
) -> nn.Module:
    return (
        ReZero(module)
        if use_rezero
        else PreNorm(norm_type, dim, module)
    )


class Reformer_tune(nn.Module):
    """Reformer stack with per-layer hash/attention configuration."""

    def __init__(
        self,
        dim: int,
        depth: int,
        max_seq_len: int,
        heads: int = 8,
        dim_head: int | None = None,
        bucket_size_list: Sequence[int] | None = None,
        n_hashes_list: Sequence[int] | None = None,
        ff_chunks: int = 100,
        attn_chunks: int | None = None,
        causal: bool = False,
        weight_tie: bool = False,
        lsh_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_activation: type[nn.Module] | None = None,
        ff_mult: int = 4,
        ff_glu: bool = False,
        post_attn_dropout: float = 0.0,
        layer_dropout: float = 0.0,
        lsh_attend_across_buckets: bool = True,
        lsh_allow_duplicate_attention: bool = True,
        random_rotations_per_head: bool = False,
        twin_attention: bool = False,
        use_scale_norm: bool = False,
        use_rezero: bool = False,
        use_full_attn: bool = False,
        full_attn_thres: int = 0,
        reverse_thres: int = 0,
        num_mem_kv: int = 0,
        one_value_head: bool = False,
        n_local_attn_heads: int = 0,
        pkm_layers: Sequence[int] = (),
        pkm_num_keys: int = 128,
        attn_type_list: Sequence[str] | None = None,
        store_stats: bool = False,
        scheduler_hashes: int = 10,
        thresh: float = 0.01,
    ) -> None:
        super().__init__()
        bucket_sizes = (
            [] if bucket_size_list is None
            else list(bucket_size_list)
        )
        hash_counts = (
            [] if n_hashes_list is None
            else list(n_hashes_list)
        )
        attention_types = (
            [] if attn_type_list is None
            else list(attn_type_list)
        )
        if len(bucket_sizes) != depth:
            raise ValueError("bucket_size_list must match depth")
        if len(hash_counts) != depth:
            raise ValueError("n_hashes_list must match depth")
        if len(attention_types) != depth:
            raise ValueError("attn_type_list must match depth")

        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.bucket_size_list = bucket_sizes
        self.num_mem_kv = num_mem_kv
        self.twin_attention = twin_attention
        self.full_attn_thres = full_attn_thres

        def make_ff() -> nn.Module:
            return Chunk(
                ff_chunks,
                FeedForward(
                    dim,
                    dropout=ff_dropout,
                    activation=ff_activation,
                    mult=ff_mult,
                    glu=ff_glu,
                ),
                along_dim=-2,
            )

        def make_pkm() -> nn.Module:
            return cast(
                nn.Module, PKM(dim, num_keys=pkm_num_keys)
            )

        def make_attn(index: int) -> LSHSelfAttention:
            return LSHSelfAttention(
                dim,
                heads,
                bucket_sizes[index],
                hash_counts[index],
                causal=causal,
                dim_head=dim_head,
                dropout=lsh_dropout,
                post_attn_dropout=post_attn_dropout,
                attn_chunks=attn_chunks,
                allow_duplicate_attention=(
                    lsh_allow_duplicate_attention
                ),
                attend_across_buckets=(
                    lsh_attend_across_buckets
                ),
                random_rotations_per_head=(
                    random_rotations_per_head
                ),
                num_mem_kv=num_mem_kv,
                use_full_attn=use_full_attn,
                full_attn_thres=full_attn_thres,
                one_value_head=one_value_head,
                n_local_attn_heads=n_local_attn_heads,
                max_seq_len=max_seq_len,
                attn_type=attention_types[index],
                store_stats=store_stats,
                scheduler_hashes=scheduler_hashes,
                thresh=thresh,
            )

        shared_attn = (
            cache_fn(lambda: make_attn(0))
            if weight_tie else None
        )
        ff_factory = (
            cache_fn(make_ff) if weight_tie else make_ff
        )
        pkm_factory = (
            cache_fn(make_pkm) if weight_tie else make_pkm
        )
        norm_type: Callable[[int], nn.Module] = (
            ScaleNorm if use_scale_norm else nn.LayerNorm
        )

        blocks: list[list[nn.Module]] = []
        for index in range(depth):
            attn = (
                shared_attn()
                if shared_attn is not None
                else make_attn(index)
            )
            layer_num = index + 1
            if layer_num in pkm_layers:
                parallel = pkm_factory()
            elif twin_attention:
                parallel = (
                    shared_attn()
                    if shared_attn is not None
                    else make_attn(index)
                )
            else:
                parallel = ff_factory()

            blocks.append(
                [
                    _wrap_residual(
                        attn,
                        use_rezero=use_rezero,
                        norm_type=norm_type,
                        dim=dim,
                    ),
                    _wrap_residual(
                        parallel,
                        use_rezero=use_rezero,
                        norm_type=norm_type,
                        dim=dim,
                    ),
                ]
            )

        self.layers = ReversibleSequence(
            blocks,
            layer_dropout=layer_dropout,
            reverse_thres=reverse_thres,
            send_signal=True,
        )
        self.layer_modules: list[nn.Module] = list(
            chain.from_iterable(blocks)
        )

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        doubled = torch.cat([x, x], dim=-1)
        routed = self.layers(
            doubled,
            arg_route=(True, self.twin_attention),
            **kwargs,
        )
        return torch.stack(
            routed.chunk(2, dim=-1)
        ).mean(dim=0)


class Reformer(nn.Module):
    """Reformer stack with one hash/attention configuration."""

    def __init__(
        self,
        dim: int,
        depth: int,
        max_seq_len: int,
        heads: int = 8,
        dim_head: int | None = None,
        bucket_size: int = 64,
        n_hashes: int = 8,
        ff_chunks: int = 100,
        attn_chunks: int | None = None,
        causal: bool = False,
        weight_tie: bool = False,
        lsh_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_activation: type[nn.Module] | None = None,
        ff_mult: int = 4,
        ff_glu: bool = False,
        post_attn_dropout: float = 0.0,
        layer_dropout: float = 0.0,
        lsh_attend_across_buckets: bool = True,
        lsh_allow_duplicate_attention: bool = True,
        random_rotations_per_head: bool = False,
        twin_attention: bool = False,
        use_scale_norm: bool = False,
        use_rezero: bool = False,
        use_full_attn: bool = False,
        full_attn_thres: int = 0,
        reverse_thres: int = 0,
        num_mem_kv: int = 0,
        one_value_head: bool = False,
        n_local_attn_heads: int = 0,
        pkm_layers: Sequence[int] = (),
        pkm_num_keys: int = 128,
        attn_type: str = "lsh",
        store_stats: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.bucket_size = bucket_size
        self.num_mem_kv = num_mem_kv
        self.twin_attention = twin_attention
        self.full_attn_thres = full_attn_thres

        def make_attn() -> LSHSelfAttention:
            return LSHSelfAttention(
                dim,
                heads,
                bucket_size,
                n_hashes,
                causal=causal,
                dim_head=dim_head,
                dropout=lsh_dropout,
                post_attn_dropout=post_attn_dropout,
                attn_chunks=attn_chunks,
                allow_duplicate_attention=(
                    lsh_allow_duplicate_attention
                ),
                attend_across_buckets=(
                    lsh_attend_across_buckets
                ),
                random_rotations_per_head=(
                    random_rotations_per_head
                ),
                num_mem_kv=num_mem_kv,
                use_full_attn=use_full_attn,
                full_attn_thres=full_attn_thres,
                one_value_head=one_value_head,
                n_local_attn_heads=n_local_attn_heads,
                max_seq_len=max_seq_len,
                attn_type=attn_type,
                store_stats=store_stats,
            )

        def make_ff() -> nn.Module:
            return Chunk(
                ff_chunks,
                FeedForward(
                    dim,
                    dropout=ff_dropout,
                    activation=ff_activation,
                    mult=ff_mult,
                    glu=ff_glu,
                ),
                along_dim=-2,
            )

        def make_pkm() -> nn.Module:
            return cast(
                nn.Module, PKM(dim, num_keys=pkm_num_keys)
            )

        attn_factory = (
            cache_fn(make_attn) if weight_tie else make_attn
        )
        ff_factory = (
            cache_fn(make_ff) if weight_tie else make_ff
        )
        pkm_factory = (
            cache_fn(make_pkm) if weight_tie else make_pkm
        )
        norm_type: Callable[[int], nn.Module] = (
            ScaleNorm if use_scale_norm else nn.LayerNorm
        )

        blocks: list[list[nn.Module]] = []
        for index in range(depth):
            attn = attn_factory()
            if (index + 1) in pkm_layers:
                parallel = pkm_factory()
            elif twin_attention:
                parallel = attn_factory()
            else:
                parallel = ff_factory()
            blocks.append(
                [
                    _wrap_residual(
                        attn,
                        use_rezero=use_rezero,
                        norm_type=norm_type,
                        dim=dim,
                    ),
                    _wrap_residual(
                        parallel,
                        use_rezero=use_rezero,
                        norm_type=norm_type,
                        dim=dim,
                    ),
                ]
            )

        self.layers = ReversibleSequence(
            blocks,
            layer_dropout=layer_dropout,
            reverse_thres=reverse_thres,
            send_signal=True,
        )
        self.layer_modules: list[nn.Module] = list(
            chain.from_iterable(blocks)
        )

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        doubled = torch.cat([x, x], dim=-1)
        routed = self.layers(
            doubled,
            arg_route=(True, self.twin_attention),
            **kwargs,
        )
        return torch.stack(
            routed.chunk(2, dim=-1)
        ).mean(dim=0)


class ReformerLM_tune(nn.Module):
    """Token LM wrapper around the per-layer-configurable Reformer."""

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        max_seq_len: int,
        heads: int = 8,
        dim_head: int | None = None,
        bucket_size_list: Sequence[int] | None = None,
        n_hashes_list: Sequence[int] | None = None,
        ff_chunks: int = 100,
        attn_chunks: int = 1,
        causal: bool = False,
        weight_tie: bool = False,
        lsh_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        ff_activation: type[nn.Module] | None = None,
        ff_glu: bool = False,
        post_attn_dropout: float = 0.0,
        layer_dropout: float = 0.0,
        random_rotations_per_head: bool = False,
        twin_attention: bool = False,
        use_scale_norm: bool = False,
        use_rezero: bool = False,
        use_full_attn: bool = False,
        full_attn_thres: int = 0,
        reverse_thres: int = 0,
        num_mem_kv: int = 0,
        one_value_head: bool = False,
        emb_dim: int | None = None,
        return_embeddings: bool = False,
        weight_tie_embedding: bool = False,
        fixed_position_emb: bool = False,
        absolute_position_emb: bool = False,
        axial_position_shape: tuple[int, ...] | None = None,
        n_local_attn_heads: int = 0,
        pkm_layers: Sequence[int] = (),
        pkm_num_keys: int = 128,
        attn_type_list: Sequence[str] | None = None,
        store_stats: bool = False,
        scheduler_hashes: int = 10,
        thresh: float = 0.01,
    ) -> None:
        super().__init__()
        embedding_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, embedding_dim)
        self.to_model_dim: nn.Module = (
            Identity()
            if embedding_dim == dim
            else nn.Linear(embedding_dim, dim)
        )

        bucket_sizes = (
            [] if bucket_size_list is None
            else list(bucket_size_list)
        )
        if absolute_position_emb:
            self.pos_emb: nn.Module = (
                AbsolutePositionalEmbedding(
                    embedding_dim, max_seq_len
                )
            )
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(
                embedding_dim
            )
        else:
            if not bucket_sizes:
                raise ValueError(
                    "bucket_size_list is required for axial embedding"
                )
            axial_shape = default(
                axial_position_shape,
                (
                    max_seq_len // bucket_sizes[0],
                    bucket_sizes[0],
                ),
            )
            self.pos_emb = cast(
                nn.Module,
                AxialPositionalEmbedding(
                    embedding_dim, axial_shape
                ),
            )

        self.reformer = Reformer_tune(
            dim,
            depth,
            max_seq_len,
            heads=heads,
            dim_head=dim_head,
            bucket_size_list=bucket_sizes,
            n_hashes_list=n_hashes_list,
            ff_chunks=ff_chunks,
            attn_chunks=attn_chunks,
            causal=causal,
            weight_tie=weight_tie,
            lsh_dropout=lsh_dropout,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
            ff_glu=ff_glu,
            ff_dropout=ff_dropout,
            post_attn_dropout=post_attn_dropout,
            layer_dropout=layer_dropout,
            random_rotations_per_head=random_rotations_per_head,
            twin_attention=twin_attention,
            use_scale_norm=use_scale_norm,
            use_rezero=use_rezero,
            use_full_attn=use_full_attn,
            full_attn_thres=full_attn_thres,
            reverse_thres=reverse_thres,
            num_mem_kv=num_mem_kv,
            one_value_head=one_value_head,
            n_local_attn_heads=n_local_attn_heads,
            pkm_layers=pkm_layers,
            pkm_num_keys=pkm_num_keys,
            attn_type_list=attn_type_list,
            store_stats=store_stats,
            scheduler_hashes=scheduler_hashes,
            thresh=thresh,
        )

        if return_embeddings:
            self.out: nn.Module = Identity()
        else:
            output_projection: nn.Module = (
                nn.Linear(embedding_dim, num_tokens)
                if not weight_tie_embedding
                else MatrixMultiply(
                    self.token_emb.weight,
                    transpose=True,
                    normalize=True,
                )
            )
            self.out = nn.Sequential(
                nn.Linear(dim, embedding_dim)
                if embedding_dim != dim
                else Identity(),
                output_projection,
            )
            self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.token_emb.weight.data.uniform_(
            -initrange, initrange
        )
        if isinstance(self.out, nn.Sequential):
            final = self.out[-1]
            if isinstance(final, nn.Linear):
                if final.bias is not None:
                    final.bias.data.zero_()
                final.weight.data.uniform_(
                    -initrange, initrange
                )

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        embedded = self.token_emb(x)
        position = cast(
            torch.Tensor, self.pos_emb(embedded)
        ).type_as(embedded)
        hidden = embedded + position
        hidden = cast(
            torch.Tensor, self.to_model_dim(hidden)
        )
        hidden = self.reformer(hidden, **kwargs)
        return cast(torch.Tensor, self.out(hidden))

    def _attention_layers(self) -> list[LSHSelfAttention]:
        layers: list[LSHSelfAttention] = []
        for index in range(
            len(self.reformer.layer_modules) // 2
        ):
            wrapper = self.reformer.layer_modules[
                2 * index
            ]
            fn = getattr(wrapper, "fn", None)
            if not isinstance(fn, LSHSelfAttention):
                raise TypeError(
                    "expected LSHSelfAttention residual wrapper"
                )
            layers.append(fn)
        return layers

    def clear_non_rotation_gradients(self) -> None:
        for index, attention in enumerate(
            self._attention_layers()
        ):
            attention.toqk.zero_grad()
            attention.tov.zero_grad()
            attention.to_out.zero_grad()
            parallel = self.reformer.layer_modules[
                2 * index + 1
            ]
            parallel.zero_grad()

    def get_triplet_loss(self) -> float | torch.Tensor:
        total: float | torch.Tensor = 0.0
        for attention in self._attention_layers():
            loss = attention.triplet_loss
            if loss is None:
                continue
            if isinstance(total, float):
                total = loss
            else:
                total = total + loss
        return total

    def update_simhash(self) -> None:
        for attention in self._attention_layers():
            method = getattr(
                attention.lsh_attn,
                "update_simhash",
                None,
            )
            if callable(method):
                cast(Callable[[], None], method)()

    def reset_triplet(self) -> None:
        for attention in self._attention_layers():
            if isinstance(
                attention.lsh_attn, TripletLSHAttention
            ):
                attention.lsh_attn.reset_rotations()

    def get_statistics(
        self, batch_size: int
    ) -> list[float]:
        del batch_size
        means: list[float] = []
        for attention in self._attention_layers():
            attn = attention.lsh_attn
            means.append(
                attn.mean_dp / attn.stat_count
            )
            attn.mean_dp = 0.0
            attn.stat_count = 0
        return means

    def set_alpha(self, alpha: float) -> None:
        for attention in self._attention_layers():
            if isinstance(
                attention.lsh_attn, TripletLSHAttention
            ):
                attention.lsh_attn.alpha = alpha

    def clear_triplet_loss(self) -> None:
        for attention in self._attention_layers():
            attention.triplet_loss = None

    def save_triplet_params(self, prefix: str) -> None:
        for index, attention in enumerate(
            self._attention_layers()
        ):
            if isinstance(
                attention.lsh_attn, TripletLSHAttention
            ):
                torch.save(
                    attention.lsh_attn.rotations.weight,
                    f"{prefix}{index}.pt",
                )


class ReformerLM(nn.Module):
    """Token LM wrapper around a uniform-config Reformer stack."""

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        depth: int,
        max_seq_len: int,
        heads: int = 8,
        dim_head: int | None = None,
        bucket_size: int = 64,
        n_hashes: int = 4,
        ff_chunks: int = 100,
        attn_chunks: int = 1,
        causal: bool = False,
        weight_tie: bool = False,
        lsh_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        ff_activation: type[nn.Module] | None = None,
        ff_glu: bool = False,
        post_attn_dropout: float = 0.0,
        layer_dropout: float = 0.0,
        random_rotations_per_head: bool = False,
        twin_attention: bool = False,
        use_scale_norm: bool = False,
        use_rezero: bool = False,
        use_full_attn: bool = False,
        full_attn_thres: int = 0,
        reverse_thres: int = 0,
        num_mem_kv: int = 0,
        one_value_head: bool = False,
        emb_dim: int | None = None,
        return_embeddings: bool = False,
        weight_tie_embedding: bool = False,
        fixed_position_emb: bool = False,
        absolute_position_emb: bool = False,
        axial_position_shape: tuple[int, ...] | None = None,
        n_local_attn_heads: int = 0,
        pkm_layers: Sequence[int] = (),
        pkm_num_keys: int = 128,
        attn_type: str = "lsh",
        store_stats: bool = False,
    ) -> None:
        super().__init__()
        embedding_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, embedding_dim)
        self.to_model_dim: nn.Module = (
            Identity()
            if embedding_dim == dim
            else nn.Linear(embedding_dim, dim)
        )

        if absolute_position_emb:
            self.pos_emb: nn.Module = (
                AbsolutePositionalEmbedding(
                    embedding_dim, max_seq_len
                )
            )
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(
                embedding_dim
            )
        else:
            axial_shape = default(
                axial_position_shape,
                (
                    max_seq_len // bucket_size,
                    bucket_size,
                ),
            )
            self.pos_emb = cast(
                nn.Module,
                AxialPositionalEmbedding(
                    embedding_dim, axial_shape
                ),
            )

        self.reformer = Reformer(
            dim,
            depth,
            max_seq_len,
            heads=heads,
            dim_head=dim_head,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            ff_chunks=ff_chunks,
            attn_chunks=attn_chunks,
            causal=causal,
            weight_tie=weight_tie,
            lsh_dropout=lsh_dropout,
            ff_mult=ff_mult,
            ff_activation=ff_activation,
            ff_glu=ff_glu,
            ff_dropout=ff_dropout,
            post_attn_dropout=post_attn_dropout,
            layer_dropout=layer_dropout,
            random_rotations_per_head=random_rotations_per_head,
            twin_attention=twin_attention,
            use_scale_norm=use_scale_norm,
            use_rezero=use_rezero,
            use_full_attn=use_full_attn,
            full_attn_thres=full_attn_thres,
            reverse_thres=reverse_thres,
            num_mem_kv=num_mem_kv,
            one_value_head=one_value_head,
            n_local_attn_heads=n_local_attn_heads,
            pkm_layers=pkm_layers,
            pkm_num_keys=pkm_num_keys,
            attn_type=attn_type,
            store_stats=store_stats,
        )

        if return_embeddings:
            self.out: nn.Module = Identity()
        else:
            output_projection: nn.Module = (
                nn.Linear(embedding_dim, num_tokens)
                if not weight_tie_embedding
                else MatrixMultiply(
                    self.token_emb.weight,
                    transpose=True,
                    normalize=True,
                )
            )
            self.out = nn.Sequential(
                nn.Linear(dim, embedding_dim)
                if embedding_dim != dim
                else Identity(),
                output_projection,
            )
            self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.token_emb.weight.data.uniform_(
            -initrange, initrange
        )
        if isinstance(self.out, nn.Sequential):
            final = self.out[-1]
            if isinstance(final, nn.Linear):
                if final.bias is not None:
                    final.bias.data.zero_()
                final.weight.data.uniform_(
                    -initrange, initrange
                )

    def forward(
        self, x: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        embedded = self.token_emb(x)
        position = cast(
            torch.Tensor, self.pos_emb(embedded)
        ).type_as(embedded)
        hidden = embedded + position
        hidden = cast(
            torch.Tensor, self.to_model_dim(hidden)
        )
        hidden = self.reformer(hidden, **kwargs)
        return cast(torch.Tensor, self.out(hidden))

    def _attention_layers(self) -> list[LSHSelfAttention]:
        layers: list[LSHSelfAttention] = []
        for index in range(
            len(self.reformer.layer_modules) // 2
        ):
            wrapper = self.reformer.layer_modules[
                2 * index
            ]
            fn = getattr(wrapper, "fn", None)
            if not isinstance(fn, LSHSelfAttention):
                raise TypeError(
                    "expected LSHSelfAttention residual wrapper"
                )
            layers.append(fn)
        return layers

    def clear_non_rotation_gradients(self) -> None:
        for index, attention in enumerate(
            self._attention_layers()
        ):
            attention.toqk.zero_grad()
            attention.tov.zero_grad()
            attention.to_out.zero_grad()
            self.reformer.layer_modules[
                2 * index + 1
            ].zero_grad()

    def get_triplet_loss(self) -> float | torch.Tensor:
        total: float | torch.Tensor = 0.0
        for attention in self._attention_layers():
            loss = attention.triplet_loss
            if loss is None:
                continue
            if isinstance(total, float):
                total = loss
            else:
                total = total + loss
        return total

    def update_simhash(self) -> None:
        for attention in self._attention_layers():
            method = getattr(
                attention.lsh_attn,
                "update_simhash",
                None,
            )
            if callable(method):
                cast(Callable[[], None], method)()

    def reset_triplet(self) -> None:
        for attention in self._attention_layers():
            if isinstance(
                attention.lsh_attn, TripletLSHAttention
            ):
                attention.lsh_attn.reset_rotations()

    def set_alpha(self, alpha: float) -> None:
        for attention in self._attention_layers():
            if isinstance(
                attention.lsh_attn, TripletLSHAttention
            ):
                attention.lsh_attn.alpha = alpha

    def get_statistics(
        self, batch_size: int
    ) -> list[float]:
        del batch_size
        means: list[float] = []
        for attention in self._attention_layers():
            attn = attention.lsh_attn
            means.append(
                attn.mean_dp / attn.stat_count
            )
            attn.mean_dp = 0.0
            attn.stat_count = 0
        return means

    def clear_triplet_loss(self) -> None:
        for attention in self._attention_layers():
            attention.triplet_loss = None

    def save_triplet_params(self, prefix: str) -> None:
        for index, attention in enumerate(
            self._attention_layers()
        ):
            if isinstance(
                attention.lsh_attn, TripletLSHAttention
            ):
                torch.save(
                    attention.lsh_attn.rotations.weight,
                    f"{prefix}{index}.pt",
                )
