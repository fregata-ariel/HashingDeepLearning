from __future__ import annotations

# Modified selected later CPU component; upstream GPL-derived attribution: ../NOTICE.
# Adapted from https://github.com/giannisdaras/smyrf/blob/master/smyrf/torch/attn.py
import math
from typing import TypeAlias, cast
import torch
import torch.nn as nn

from einops import rearrange, repeat

from .padding import pad_to_multiple
from .permutation import invert_permutation
from .hash_utils import XBOXPLUS, lsh_clustering, cpu_factory
from .batching import batched_index_select
from .reformer_utils import max_neg_value, chunked_sum
from src.models.modules.masking import LengthMask, FullMask
from .masks import pad_mask
from src.models.modules.attention.feature_maps_sb import SBPerformerFeatures
from src.models.modules.attention.scatterbrain_utils import linear_attention_normalization
from src.models.modules.attention.scatterbrain_utils import linear_attention


ReturnedWeights: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None


class SBSmyrfAttention(nn.Module):
    """Selected actual hybrid forward, restricted to validated noncausal CPU use."""
    def __init__(self, n_hashes: int, q_cluster_size: int, k_cluster_size: int, d_head: int,
                 r: float = 1, nb_features: int | None = None, ortho_scaling: int = 0,
                 softmax_eps: float = 0.0, causal: bool = False, softmax_temp: float | None = None,
                 attention_dropout: float = 0.0, device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        allocation_dtype = torch.get_default_dtype() if dtype is None else dtype
        allocation_device = torch.device('cpu') if device is None else torch.device(device)
        cpu_factory(allocation_device, allocation_dtype)
        if any(type(n) is not int or n <= 0 for n in (n_hashes, q_cluster_size, k_cluster_size, d_head)):
            raise ValueError('counts and dimensions must be positive integers')
        if causal or attention_dropout != 0.0 or softmax_eps != 0.0:
            raise ValueError('only noncausal dropout=0 eps=0 is supported')
        if (not math.isfinite(r) or r <= 0 or ortho_scaling not in (0, 1)
                or (softmax_temp is not None and (not math.isfinite(softmax_temp) or softmax_temp <= 0))):
            raise ValueError('invalid finite positive radius/temperature or projection scaling')
        effective_features = int(d_head * math.log(d_head)) if nb_features is None else nb_features
        if type(effective_features) is not int or effective_features < 2:
            raise ValueError('cosh features require at least two requested features')
        self.feature_map = SBPerformerFeatures(d_head, nb_features, ortho_scaling=ortho_scaling,
                                               softmax_temp=softmax_temp, eps=softmax_eps,
                                               device=allocation_device, dtype=allocation_dtype)
        self.n_hashes = n_hashes
        self.q_cluster_size = q_cluster_size
        self.k_cluster_size = k_cluster_size
        self.d_head = d_head
        self.causal = causal
        self.softmax_temp = softmax_temp
        self.dropout = nn.Dropout(attention_dropout)
        self.hash_fn = XBOXPLUS()
        self.r = r

    def _validate(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  key_padding_mask: LengthMask | None) -> None:
        if self.causal or self.dropout.p != 0 or self.feature_map.eps != 0:
            raise ValueError('only noncausal dropout=0 eps=0 is supported')
        for tensor in (query, key, value):
            cpu_factory(tensor.device, tensor.dtype)
            if tensor.ndim != 4 or min(tensor.shape) <= 0 or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Q/K/V must be finite nonempty rank-four CPU tensors')
        if (query.dtype != key.dtype or query.dtype != value.dtype or query.device != key.device
                or query.device != value.device or query.shape[0] != key.shape[0]
                or query.shape[0] != value.shape[0] or query.shape[2] != key.shape[2]
                or query.shape[2] != value.shape[2] or query.shape[-1] != self.d_head
                or key.shape[-1] != self.d_head or key.shape[1] != value.shape[1]):
            raise ValueError('Q/K/V dtype/device/batch/head/dimension mismatch')
        if (self.feature_map.projection_matrix.dtype != query.dtype
                or self.feature_map.projection_matrix.device != query.device):
            raise ValueError('module feature dtype/device must match Q/K/V; construct or move the module coherently')
        temperature = self.softmax_temp or 1 / math.sqrt(self.d_head)
        if any(not bool(torch.isfinite(t.square().sum(dim=-1) * temperature).all()) for t in (query, key)):
            raise ValueError('scaled query/key squared norms exceed computation dtype')
        q_padded = (query.shape[1] + self.q_cluster_size - 1) // self.q_cluster_size
        k_padded = (key.shape[1] + self.k_cluster_size - 1) // self.k_cluster_size
        if q_padded != k_padded:
            raise ValueError('query/key must have equal padded cluster counts')
        if key_padding_mask is not None:
            if not isinstance(key_padding_mask, LengthMask):
                raise ValueError('only prefix LengthMask is supported')
            mask = key_padding_mask.bool_matrix
            lengths = key_padding_mask.lengths
            if (mask.shape != (key.shape[0], key.shape[1]) or mask.device != key.device
                    or lengths.dtype != torch.long or lengths.shape != (key.shape[0],)
                    or bool((lengths < 0).any()) or bool((lengths > key.shape[1]).any())):
                raise ValueError('invalid prefix lengths/mask shape/device/dtype')

    def hash_vectors(self, query: torch.Tensor, key: torch.Tensor,
                     key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # XBOX+ transform
        self.hash_fn.set_norms(query, key)
        query_t = self.hash_fn.Q(query)
        key_t = self.hash_fn.K(key)

        num_clusters = query_t.shape[-2] // self.q_cluster_size
        assert num_clusters == (key_t.shape[-2] // self.k_cluster_size), 'Unequal number of clusters for query and key.'
        q_positions, k_positions = lsh_clustering(query_t, key_t, n_hashes=self.n_hashes, r=self.r,
                                                  key_padding_mask=key_padding_mask)
        return q_positions, k_positions

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: FullMask | None = None, key_padding_mask: LengthMask | None = None,
                need_weights: bool = False, return_attn_unnormalized: bool = False) -> tuple[torch.Tensor, ReturnedWeights]:
        """Execute selected sparse residual plus low-rank CPU attention.

        The selected support replaces low-rank entries with exact exponent
        entries before a shared normalizer; it does not make normalized weights
        on that support equal dense softmax weights. Tests derive H, HV and H1
        independently from raw inputs and actual rank permutations. Allocation
        and count-log repairs preserve dtype without a global-default change.
        TRACE_TEST_ID: SCATTERBRAIN-SBSMYRF-HYBRID-CPU.
        TRACE_TEST_ID: SCATTERBRAIN-SBSMYRF-DTYPE-CPU.
        """
        if attn_mask is not None:
            raise ValueError('arbitrary attention masks are unsupported')
        self._validate(query, key, value, key_padding_mask)
        if return_attn_unnormalized and not need_weights:
            raise ValueError('unnormalized weights require need_weights=True')
        _, q_seqlen_og, _, _ = query.shape
        _, k_seqlen_og, _, _ = key.shape
        query = pad_to_multiple(query, self.q_cluster_size, dims=1)
        key = pad_to_multiple(key, self.k_cluster_size, dims=1)
        value = pad_to_multiple(value, self.k_cluster_size, dims=1)

        # Extract some shapes and compute the temperature
        B, T, H, E = query.shape
        _, S, _, D = value.shape
        softmax_temp = self.softmax_temp or 1 / math.sqrt(E)

        # pad the masks
        if S > k_seqlen_og:
            if key_padding_mask is None:
                key_padding_mask = LengthMask(key.new_full((key.shape[0],), k_seqlen_og,
                                                           dtype=torch.long), max_len=S)
            else:
                key_padding_mask = pad_mask(key_padding_mask, pad_length=S - k_seqlen_og,
                                            left=False, value=False)

        query = rearrange(query, 'b t h e -> (b h) t e')
        key = rearrange(key, 'b t h e -> (b h) t e')
        value = rearrange(value, 'b s h d -> (b h) s d')
        bs = query.shape[0]

        if key_padding_mask is not None and not key_padding_mask.all_ones:
            # Repeat for all heads
            key_padding_mask_bool = repeat(key_padding_mask.bool_matrix, 'b s -> (b h) s', h=H)
        else:
            key_padding_mask_bool = None

        with torch.no_grad():
            q_positions, k_positions = self.hash_vectors(query, key,
                                                         rearrange(key_padding_mask_bool,
                                                                   'b s -> 1 b s')
                                                         if key_padding_mask_bool is not None else None)


        self.feature_map.new_feature_map(query.device)
        q_prime, q_prime_log_scale = self.feature_map.forward_queries(query)
        k_prime, k_prime_log_scale = self.feature_map.forward_keys(key)

        prime_log_scale = q_prime_log_scale + k_prime_log_scale
        m = q_prime.shape[-1]
        if key_padding_mask_bool is not None:
            k_prime.masked_fill_(~rearrange(key_padding_mask_bool, 'b s -> b s 1'), 0.0)
        q_prime_k_prime_1 = linear_attention_normalization(q_prime, k_prime, causal=self.causal)
        q_prime_k_prime_v, attn_prime = linear_attention(q_prime, k_prime, value, need_weights=need_weights)

        # sort queries, keys, values
        def sort_to_buckets(x: torch.Tensor, perm: torch.Tensor, bucketsz: int) -> torch.Tensor:
            return rearrange(batched_index_select(rearrange(x, 'b s d -> 1 b s d'), perm),
                             'h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d',
                             bucketsz=bucketsz)

        s_query = sort_to_buckets(query, q_positions, self.q_cluster_size)
        s_key = sort_to_buckets(key, k_positions, self.k_cluster_size)
        s_value = sort_to_buckets(value, k_positions, self.k_cluster_size)
        sq_prime = sort_to_buckets(q_prime, q_positions, self.q_cluster_size)
        sk_prime = sort_to_buckets(k_prime, k_positions, self.k_cluster_size)
        # sq_prime, sq_prime_log_scale = kernel_fn(s_queries, is_query=True)
        # sk_prime, sk_prime_log_scale = kernel_fn(s_keys, is_query=False)
        # k_prime_log_scale doesn't depend on the index of the token
        sprime_log_scale = sort_to_buckets(prime_log_scale, q_positions, self.q_cluster_size)
        # sprime_log_scale = sq_prime_log_scale + sk_prime_log_scale

        inner = torch.einsum('...id,...jd->...ij', s_query, s_key) * softmax_temp
        dots_prime = torch.einsum('...im,...jm->...ij', sq_prime, sk_prime)

        masked_value = max_neg_value(inner)
        # mask out attention to padded tokens
        if key_padding_mask is not None and not key_padding_mask.all_ones:
            assert key_padding_mask_bool is not None
            s_key_padding_mask = sort_to_buckets(rearrange(key_padding_mask_bool,
                                                           'b s -> b s 1'),
                                                 k_positions, self.k_cluster_size)
            s_key_padding_mask = rearrange(s_key_padding_mask,
                                           '... bucketsz 1 -> ... 1 bucketsz')
            inner.masked_fill_(~s_key_padding_mask, masked_value)
            dots_prime.masked_fill_(~s_key_padding_mask, 0.0)

        # Causal masking
        if self.causal:
            s_q_positions = rearrange(q_positions,
                                      'h b (nbuckets bucketsz) -> h b nbuckets bucketsz 1',
                                      bucketsz=self.q_cluster_size)
            s_k_positions = rearrange(k_positions,
                                      'h b (nbuckets bucketsz) -> h b nbuckets 1 bucketsz',
                                      bucketsz=self.k_cluster_size)
            causal_mask = s_q_positions < s_k_positions
            inner.masked_fill_(causal_mask, masked_value)
            dots_prime.masked_fill_(causal_mask, 0.0)
            del causal_mask

        q_rev_positions = invert_permutation(q_positions)

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # Count how many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition.
        if self.n_hashes > 1:
            k_rev_positions = invert_permutation(k_positions)
            q_bucket_idx = rearrange(q_rev_positions // self.q_cluster_size,
                                     'h b seqlen -> b seqlen h')
            k_bucket_idx = rearrange(k_rev_positions // self.k_cluster_size,
                                     'h b seqlen -> b seqlen h')
            s_q_bucket_idx = sort_to_buckets(q_bucket_idx, q_positions, self.q_cluster_size)
            s_k_bucket_idx = sort_to_buckets(k_bucket_idx, k_positions, self.k_cluster_size)
            dup_counts = (rearrange(s_q_bucket_idx, '... bk_size h -> ... bk_size 1 h') ==
                          rearrange(s_k_bucket_idx, '... bk_size h -> ... 1 bk_size h'))
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(self.n_hashes * bs))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == inner.shape
            inner = inner - torch.log(dup_counts.to(dtype=inner.dtype))
            dots_prime = dots_prime / dup_counts

        # free memory

        # softmax denominator
        # TD: Even though we call this dots_logsumexp, it can be of arbitrary value and the
        # computation would still be correct (assuming infinite precision), since it's just an
        # arbitrary scaling of @dots.
        # Here we choose it for numerical stability: we want torch.exp(inner - dots_logsumexp) <= 1.0
        # and torch.exp(spring_log_scale - dots_logsumexp) <= 1.0
        # dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        dots_logsumexp = torch.maximum(torch.amax(inner, dim=-1, keepdim=True), sprime_log_scale)
        # TD: dots and dots_sum has log scale dots_logsumexp
        # TD: No longer need this because we pick dots_logsumexp to not be -inf
        # dots_prime_scale = torch.exp(sprime_log_scale - dots_logsumexp)
        # nan_q_indices = dots_prime_scale.isinf()
        # # dots_logsumexp[nan_q_indices] = 0.0
        # dots_logsumexp = torch.where(nan_q_indices, torch.tensor(0.0, device=dots_logsumexp.device),
        #                              dots_logsumexp)
        dots_prime_scale = torch.exp(sprime_log_scale - dots_logsumexp)
        dots = torch.exp(inner - dots_logsumexp) - dots_prime * dots_prime_scale
        # TD: No longer need this because we pick dots_logsumexp to not be -inf
        # If the whole row within this bucket is masked out, then inner is the uniform distribution.
        # We actually want it to be zero.
        # if key_padding_mask is not None and not key_padding_mask.all_ones:
        #     full_row_mask = (inner <= masked_value).all(dim=-1, keepdim=True)
        #     dots = dots.masked_fill(full_row_mask, 0.0)
        dots_sum = dots.sum(dim=-1, keepdim=True)

        # dropout
        dropped_dots = cast(torch.Tensor, self.dropout(dots))

        # n_hashes outs
        so = torch.einsum('...ij,...jd->...id', dropped_dots, s_value)

        # undo sort
        def unsort_from_buckets(s_x: torch.Tensor, perm_inverse: torch.Tensor) -> torch.Tensor:
            b_x = rearrange(s_x, 'h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d')
            return batched_index_select(b_x, perm_inverse)

        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(dots_logsumexp, q_rev_positions)
        dots_sum_unsort = unsort_from_buckets(dots_sum, q_rev_positions)

        # free memory
        del q_rev_positions

        normalization_log_scale = torch.logsumexp(logits, dim=0)
        probs = torch.exp(logits - rearrange(normalization_log_scale, '... -> 1 ...'))
        out_lsh = torch.sum(o * probs, dim=0)

        prime_scale = torch.exp(prime_log_scale - normalization_log_scale)
        out = out_lsh + q_prime_k_prime_v * prime_scale

        normalization = (dots_sum_unsort * probs).sum(dim=0) + q_prime_k_prime_1.unsqueeze(-1) * prime_scale
        out_normalized = out / normalization.clamp_min(1e-6)
        out_normalized = (rearrange(out_normalized, '(b h) t d -> b t h d', h=H))[:, :q_seqlen_og]

        attn: ReturnedWeights = None
        if need_weights:
            assert attn_prime is not None
            q_pos_2d = rearrange(q_positions, 'h b (nbuckets bucketsz) -> h b nbuckets bucketsz 1',
                                 bucketsz=self.q_cluster_size)
            k_pos_2d = rearrange(k_positions, 'h b (nbuckets bucketsz) -> h b nbuckets 1 bucketsz',
                                 bucketsz=self.k_cluster_size)
            pos_2d = rearrange(q_pos_2d * S + k_pos_2d,
                               'h b nbuckets qbucketsz kbucketsz -> h b (nbuckets qbucketsz kbucketsz)')
            unsorted_dots = dots.new_zeros(self.n_hashes, bs, T * S)
            unsorted_dots.scatter_(-1, pos_2d, dots.view_as(pos_2d))
            del pos_2d
            unsorted_dots = rearrange(unsorted_dots,
                                      'h b (q_seqlen k_seqlen) -> h b q_seqlen k_seqlen',
                                      q_seqlen=T)
            attn_lsh = torch.sum(unsorted_dots * probs, dim=0)
            attn_unnormalized = attn_lsh + attn_prime * prime_scale
            attn = attn_unnormalized / normalization.clamp_min(1e-6)
            attn = rearrange(attn, '(b h) t s -> b h t s', h=H)[:, :, :q_seqlen_og, :k_seqlen_og]
            if return_attn_unnormalized:  # For testing purpose
                attn_unnormalized = rearrange(
                    attn_unnormalized, '(b h) t s -> b h t s', h=H
                )[:, :, :q_seqlen_og, :k_seqlen_og]
                normalization_log_scale = rearrange(normalization_log_scale,
                                                    '(b h) s 1 -> b h s 1', h=H)[:, :, :q_seqlen_og]
                attn_prime = rearrange(attn_prime,
                                       '(b h) s d -> b h s d', h=H)[:, :, :q_seqlen_og, :k_seqlen_og]
                prime_log_scale = rearrange(prime_log_scale,
                                            '(b h) s 1 -> b h s 1', h=H)[:, :, :q_seqlen_og]
                smyrf_mask = rearrange(attn_lsh != 0.0,
                                       '(b h) t s -> b h t s', h=H)[:, :, :q_seqlen_og, :k_seqlen_og]
                attn = (attn, attn_unnormalized * torch.exp(normalization_log_scale),
                        attn_prime * torch.exp(prime_log_scale), smyrf_mask)

        if not bool(torch.isfinite(out_normalized).all()):
            raise RuntimeError('nonfinite maintained attention output')
        if attn is not None:
            returned = (attn,) if isinstance(attn, torch.Tensor) else attn
            if any(not bool(torch.isfinite(t).all()) for t in returned):
                raise RuntimeError('nonfinite maintained attention weights')
        return out_normalized, attn
