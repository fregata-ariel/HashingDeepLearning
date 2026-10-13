#include "sparse_attention.h"
#include <limits>
#include <stdexcept>

const __m512 LOG2E_VEC = _mm512_set1_ps(1.44269504089f); // log2(e)
const __m512 MAGIC_FLOAT_BIAS = _mm512_set1_ps(12582912.0f); // Bias for integer part
const __m512 ONE_VEC = _mm512_set1_ps(1.0f);
const __m512 LN2_PART_VEC = _mm512_set1_ps(0.69314718056f); // ln(2) for scaling

// Degree-six Taylor coefficients for 2^f with |f| <= 1/2.
const __m512 EXP_POLY_COEFFS[7] = {
    _mm512_set1_ps(1.0000000000f),
    _mm512_set1_ps(0.6931471806f),
    _mm512_set1_ps(0.2402265069f),
    _mm512_set1_ps(0.05550410866f),
    _mm512_set1_ps(0.009618129108f),
    _mm512_set1_ps(0.001333355815f),
    _mm512_set1_ps(0.000154035304f),
};

/**
 * AVX512 exp policy (#74): finite x in [-87,88] uses a degree-six
 * centered polynomial (|fraction| <= 1/2) and a normal binary exponent.
 * Mathematical polynomial relative remainder <= 2.5e-7; the independent
 * native FP64 gate allows 2e-5 including FP32 scaling/FMA/reconstruction.
 * All other lanes use the platform expf: subnormal results/underflow, overflow,
 * +Inf -> +Inf, -Inf -> 0, NaN -> NaN. IEEE round-to-nearest and gradual
 * underflow are assumed; no fast-math/FTZ is enabled by these checks.
 * Special/large lanes are replaced with zero BEFORE integer conversion and
 * bit construction. Portable host-exp lanes do not validate this function.
 * Support: TRACE_TEST_ID: MAGICPIG-EXP-NATIVE-CPU.
 */
__m512 avx512_exp_ps(__m512 x) {
    const __mmask16 ordinary =
        _mm512_cmp_ps_mask(x, _mm512_set1_ps(-87.0f), _CMP_GE_OQ) &
        _mm512_cmp_ps_mask(x, _mm512_set1_ps(88.0f), _CMP_LE_OQ);
    const __m512 bounded = _mm512_maskz_mov_ps(ordinary, x);
    const __m512 scaled = _mm512_mul_ps(bounded, LOG2E_VEC);
    const __m512 rounded = _mm512_roundscale_ps(
        scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512i int_part = _mm512_cvttps_epi32(rounded);
    const __m512 frac_part = _mm512_sub_ps(scaled, rounded);
    const __m512 int_exp = _mm512_castsi512_ps(_mm512_slli_epi32(
        _mm512_add_epi32(int_part, _mm512_set1_epi32(127)), 23));

    __m512 poly = EXP_POLY_COEFFS[6];
    for (int i = 5; i >= 0; --i)
        poly = _mm512_fmadd_ps(poly, frac_part, EXP_POLY_COEFFS[i]);
    __m512 result = _mm512_mul_ps(int_exp, poly);
    if (ordinary != 0xFFFF) {
        float inputs[16], outputs[16];
        _mm512_storeu_ps(inputs, x);
        _mm512_storeu_ps(outputs, result);
        for (int i = 0; i < 16; ++i)
            if (!(ordinary & (1u << i))) outputs[i] = expf(inputs[i]);
        result = _mm512_loadu_ps(outputs);
    }
    return result;
}

/**
 * @par Paper mapping
 * MagicPIG arXiv v4, Sections 4.3 and 4.4: selected CPU attention support.
 * @par Implementation note
 * Borrowed BF16 keys/FP32 query; dim divisible by16. Supplied IDs form a
 * multiset. Repair #75 reads/initializes/writes only the logical nnz; exact
 * candidate/score allocations are supported, and nnz0 accesses no buffers.
 * Repair #76 rejects unsupported dimensions before writes. FP64 dot oracle.
 * Support: TRACE_TEST_ID: MAGICPIG-ATTENTION-BOUNDARIES-CPU.
 */
void qk_kernel(
const bfloat16 *key, 
const int *ind,
const float *query, 
float *attn_weight,
const int HEAD_DIM,
const int nnz) 
{
    if (HEAD_DIM <= 0 || HEAD_DIM % 16 != 0) throw std::invalid_argument("FP32 QK head_dim must be a positive multiple of 16");

  if (nnz < 0) throw std::invalid_argument("QK nnz must be nonnegative");
  if (nnz == 0) return;
  memset(attn_weight, 0, sizeof(float) * nnz);
  float w_f32[16];
  for (int io = 0; io < nnz; io += 16) {
    
    for (int jo = 0; jo < (HEAD_DIM / 16); ++jo) {
      __m512 x_tile = _mm512_loadu_ps(query + jo * 16);
      for (int ii = 0; ii < std::min(16, nnz - io); ++ii) {
        const int i = io + ii;
        const int row = ind[i];

        Bfloat16ToFloat_avx512(key + row * HEAD_DIM + jo * 16, w_f32, 16);
        __m512 w_tile = _mm512_loadu_ps(w_f32);

        __m512 wx = _mm512_mul_ps(w_tile, x_tile);
        float sum = _mm512_reduce_add_ps(wx);
        attn_weight[io + ii] += sum;
      }
    }

  }
}
#ifdef __AVX512BF16__
/**
 * @par Paper mapping
 * MagicPIG arXiv v4, Sections 4.3 and 4.4: BF16 dot-product support.
 * @par Implementation note
 * Borrowed keys/query use uint16_t BF16 storage; caller owns all buffers.
 * Preserves bootstrap nnz=16/dim32 and covers dim32/64/128, nnz0/16/32.
 * Repair #75 partial tiles use exact logical candidate/score lengths. Empty
 * calls access no buffers; repair #76 rejects nonmultiple32 dimensions.
 * Native execution requires AVX512F/BW/BF16 and GCC >=11, checked before the
 * SIMD binary starts. Storage alias replaces the FBGEMM header only; no
 * Torch/FBGEMM extension or conversion/dispatch/ownership claim is made.
 * @par Traceability relation
 * Support: actual restricted native body; TRACE_TEST_ID: MAGICPIG-BF16-SELECTED-CPU.
 * TRACE_TEST_ID: MAGICPIG-BF16-BOUNDARY-CPU.
 */
void qk_kernel_bf16_impl(
const bfloat16 *key, 
const int *ind,
const bfloat16 *query, 
float *attn_weight,
const int HEAD_DIM,
const int nnz) 
{
    if (HEAD_DIM <= 0 || HEAD_DIM % 32 != 0) throw std::invalid_argument("BF16 QK head_dim must be a positive multiple of 32");

    if (nnz < 0) throw std::invalid_argument("BF16 QK nnz must be nonnegative");
    if (nnz == 0) return;
    memset(attn_weight, 0, sizeof(float) * nnz);
    for (int io = 0; io < nnz; io += 16) {
        for (int jo = 0; jo < (HEAD_DIM / 32); ++jo) {
            // Load 32 bfloat16 values from query into x_tile
            __m512bh x_tile = __m512bh(_mm512_loadu_epi16(reinterpret_cast<const void*>(query + jo * 32)));

            for (int ii = 0; ii < std::min(16, nnz - io); ++ii) {
                const int i = io + ii;
                const int row = ind[i];

                // Load 32 bfloat16 values from key into w_tile
                __m512bh w_tile = __m512bh(_mm512_loadu_epi16(reinterpret_cast<const void*>(key + row * HEAD_DIM + jo * 32)));

                // Compute dot product of x_tile and w_tile
                __m512 result = _mm512_dpbf16_ps(_mm512_setzero_ps(), w_tile, x_tile);

                // Reduce and sum the result into a scalar
                float sum = _mm512_reduce_add_ps(result);

                // Update attn_weight
                attn_weight[io + ii] += sum;
            }
        }
    }
}
#endif

void qk_kernel_full(
    const bfloat16 *key,      // k 的输入，形状为 max_length x 128
    const float *query,       // q 的输入，形状为  num_attention_groups x 128
    float *attn_weight,       // 输出，形状为 num_attention_groups x max_length
    const int num_attention_groups,
    const int HEAD_DIM,       // 每个向量的维度，这里是 128
    const int nnz,           // 只计算k的前nnz <= max_length
    const int max_length
) {
    if (HEAD_DIM <= 0 || HEAD_DIM % 16 != 0 || num_attention_groups <= 0) throw std::invalid_argument("full QK requires positive groups and head_dim multiple of 16");

    const int HEAD_DIM_16 = HEAD_DIM / 16; // 每次处理 16 个维度
    const int TILE_SIZE = 64; // 块大小，处理 TILE_SIZE 行 key 一次
    
    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int ko_start = 0; ko_start < nnz; ko_start += TILE_SIZE) {
        float key_tile[TILE_SIZE][HEAD_DIM]; // 局部缓存 key 块，存为 float 格式
        int tile_rows = (ko_start + TILE_SIZE <= nnz) ? TILE_SIZE : (nnz - ko_start);
        
        // 加载当前块的 key 并转换为 float
        for (int i = 0; i < tile_rows; ++i) {
            for (int j = 0; j < HEAD_DIM_16; ++j) {
                Bfloat16ToFloat_avx512(
                    key + (ko_start + i) * HEAD_DIM + j * 16,
                    key_tile[i] + j * 16,
                    16
                );
            }
        }
        
        // 遍历 query 的每一行
        for (int qi = 0; qi < num_attention_groups; ++qi) {
            const float* query_vec = query + qi * HEAD_DIM; // 当前查询向量
            
            for (int j = 0; j < HEAD_DIM_16; ++j) {
    // 预加载 query 的一部分到局部变量
    __m512 q_tile = _mm512_loadu_ps(query_vec + j * 16);

    // 遍历当前块的所有 key 行
    for (int ki = 0; ki < tile_rows; ++ki) {
        if (j == 0) {
            // 初始化累加器，仅在每行的第一轮计算时清零
            attn_weight[qi * max_length + ko_start + ki] = 0.0f;
        }

        // 加载 key 块
        __m512 k_tile = _mm512_loadu_ps(key_tile[ki] + j * 16);

        // 逐元素乘法并累加到结果中
        __m512 prod = _mm512_mul_ps(q_tile, k_tile);
        attn_weight[qi * max_length + ko_start + ki] += _mm512_reduce_add_ps(prod);
    }
}

        }
    }
}



/**
 * Stable log of the >=2-hit binomial tail (Issue #68). No FP32 complement
 * subtraction: small expected hit counts use a positive tail recurrence,
 * larger counts use log1p/expm1. The logarithm stays finite for positive
 * inclusion even when its probability is below floating-point range.
 * TRACE_TEST_ID: MAGICPIG-PROBABILITY-STABILITY-CPU.
 */
void magicpig_inclusion_log_probability(
const double cosine, const int k, const int l, double *log_inclusion) {
  if (!log_inclusion || !std::isfinite(cosine) || cosine < -1 || cosine > 1 ||
      k < 1 || l < 2)
    throw std::invalid_argument("invalid inclusion probability parameters");
  if (cosine == -1) {
    *log_inclusion = -std::numeric_limits<double>::infinity();
    return;
  }
  if (cosine == 1) {
    *log_inclusion = 0;
    return;
  }
  // acos(-c)/pi equals 1-acos(c)/pi without cancelling near c=-1.
  const long double collision = std::acos(-static_cast<long double>(cosine)) /
                                std::acos(-1.0L);
  const long double log_table = k * std::log(collision);
  const long double p = std::exp(log_table);
  long double result;
  if (static_cast<long double>(l) * p <= 0.5L) {
    // P2 = C(L,2) p^2 (1-p)^(L-2); all later terms are positive.
    // For L*p<=0.5, successive ratios are <0.25 and the sum converges
    // quickly even for INT_MAX tables. If p underflows, later terms are
    // below machine precision while 2*log_table remains representable.
    const long double log_p2 = std::log(static_cast<long double>(l)) +
        std::log(static_cast<long double>(l - 1)) - std::log(2.0L) +
        2 * log_table + (l - 2) * std::log1p(-p);
    long double sum = 1, term = 1;
    for (int hits = 2; hits < l; ++hits) {
      term *= (static_cast<long double>(l - hits) / (hits + 1)) * p / (1 - p);
      sum += term;
      if (term <= std::numeric_limits<long double>::epsilon() * sum) break;
    }
    result = log_p2 + std::log(sum);
  } else {
    const long double excluded_log = (l - 1) * std::log1p(-p) +
                                    std::log1p((l - 1) * p);
    result = std::log(-std::expm1(excluded_log));
  }
  if (!std::isfinite(result) || result > 0)
    throw std::overflow_error("inclusion log probability outside finite range");
  *log_inclusion = static_cast<double>(result);
}

/**
 * @par Paper mapping
 * MagicPIG arXiv v4, Section 4.3 Equations 9-11: probability correction.
 * @par Implementation note
 * score and indices contain nnz valid entries; norms must be positive and
 * finite, cosine in [-1,1], sqrt_dim positive, K positive and L at least two.
 * Mutates borrowed scores; the released +1e-4 regularizer is a variant.
 * Selected-body CPU bootstrap calls this arithmetic before Softmax;
 * Independent exhaustive bit/table events and FP64 binomial mass characterize
 * +1e-4 smoothing and finite self-normalized bias, not an unbiased attention
 * estimator. Issue #67 rejects nonfinite scores, nonpositive/nonfinite norms
 * or scaling, negative nnz, K<1 and L<2 before modifying any score. Cosine
 * excursions of at most 8*FLT_EPSILON are rounded to the nearest endpoint;
 * larger excursions are rejected. Empty valid input is a no-op. Caller must
 * provide valid borrowed buffers and indices; this signature has no capacity.
 * Issue #68 computes stable log inclusion and retains explicit +1e-4
 * smoothing; finite tiny positive probabilities no longer cancel to zero.
 * @par Traceability relation
 * Support: probability/domain characterization;
 * TRACE_TEST_ID: MAGICPIG-PROBABILITY-CPU.
 * TRACE_TEST_ID: MAGICPIG-PROBABILITY-DOMAIN-CPU.
 */
void transform_kernel(
float *score, 
const int nnz,
float q_norm,
float *k_norm,
const int k,
const int l,
const float sqrt_dim,
const int *indices) { 
  if (nnz < 0 || k < 1 || l < 2 || !std::isfinite(q_norm) || q_norm <= 0 ||
      !std::isfinite(sqrt_dim) || sqrt_dim <= 0)
    throw std::invalid_argument("invalid probability parameters");
  if (nnz == 0) return;
  if (!score || !k_norm || !indices)
    throw std::invalid_argument("null probability buffer");
  const double cosine_tolerance = 8.0 * std::numeric_limits<float>::epsilon();
  std::vector<float> corrected(nnz);
  for(int j = 0; j < nnz; ++j){
        const int ind = indices[j];
        if (ind < 0 || !std::isfinite(score[j]) || !std::isfinite(k_norm[ind]) ||
            k_norm[ind] <= 0)
          throw std::invalid_argument("invalid probability score or norm");
        // Double intermediates avoid overflow/underflow of a finite FP32
        // norm product. Index capacity remains a caller precondition.
        double cosine = double(score[j]) / (double(q_norm) * k_norm[ind]);
        if (cosine < -1.0 - cosine_tolerance || cosine > 1.0 + cosine_tolerance)
          throw std::invalid_argument("probability cosine outside tolerance");
        cosine = std::max(-1.0, std::min(1.0, cosine));
        double log_inclusion;
        magicpig_inclusion_log_probability(cosine, k, l, &log_inclusion);
        // Preserve the released estimator variant explicitly: +1e-4. The
        // log-add formulation also works when exp(log_inclusion) underflows.
        const double log_regularizer = std::log(1e-4);
        const double correction = log_regularizer +
            std::log1p(std::exp(log_inclusion - log_regularizer));
        corrected[j] = static_cast<float>(double(score[j]) / sqrt_dim - correction);
        if (!std::isfinite(corrected[j]))
          throw std::overflow_error("corrected probability score outside finite range");
    }
  // Rejection of a later entry never leaves earlier scores partially corrected.
  std::copy(corrected.begin(), corrected.end(), score);
}

/**
 * @par Paper mapping
 * MagicPIG arXiv v4, Sections 4.3 and 4.4: normalized attention support.
 * @par Implementation note
 * Borrowed score buffer has nnz finite logits, nnz >= 0. At zero, score may
 * be null: no writes, maximum and base-2 LSE are both -Inf (zero mass).
 * Mutates scores to probabilities; max_value[0] is the base-2 maximum and
 * expsum[0] is base-2 log-sum-exp, not a raw sum. Buffers remain caller-owned.
 * Bootstrap uses host QK/WV; #58 also composes actual selected QK/WV bodies.
 * Nonempty candidates require finite logits. Portable lanes replace exp
 * with host exp. Native AVX512F/FMA runs the actual centered polynomial with
 * explicit extreme/special fallback and a separate approximation tolerance. No Torch extension,
 * LSH sampling, OpenMP scheduling or GPU execution is inferred.
 * @par Traceability relation
 * Support: selected arithmetic bootstrap; TRACE_TEST_ID: MAGICPIG-BASELINE-CPU.
 * TRACE_TEST_ID: MAGICPIG-SOFTMAX-BOUNDARY-CPU.
 * TRACE_TEST_ID: MAGICPIG-SOFTMAX-EMPTY-CPU.
 */
void softmax_kernel(
float *score, 
const int nnz,
float *max_value,
float *expsum) {
    // Empty support is the zero-mass domain: no score access, no probabilities.
    // Keep the ABI; -Inf base-2 metadata tells the caller this row has no mass.
    if (nnz == 0) {
        max_value[0] = -INFINITY;
        expsum[0] = -INFINITY;
        return;
    }

  
  const int nnz_16 = (nnz % 16 == 0) ? nnz : nnz + 16 - nnz % 16;

    // Step 1: Find the maximum value (reduction max)
    float max_val = *std::max_element(score, score + nnz);
    __m512 max_vec = _mm512_set1_ps(max_val);

    // Step 2: Compute exp(x - max) and sum of exponents
    __m512 sum_vec = _mm512_setzero_ps();
    for (int i = 0; i < nnz / 16; ++i) {
        __m512 scores = _mm512_loadu_ps(score + i * 16);
        __m512 shifted_scores = _mm512_sub_ps(scores, max_vec); // x - max
        __m512 exp_scores = avx512_exp_ps(shifted_scores); // exp(x - max)
        _mm512_storeu_ps(score + i * 16, exp_scores); // Store back
        sum_vec = _mm512_add_ps(sum_vec, exp_scores); // Accumulate sum
    }

    // Step 3: Horizontal sum of `sum_vec`
    float exp_sum = 0.0f;
    float temp[16];
    _mm512_storeu_ps(temp, sum_vec);
    for (int i = 0; i < 16; ++i) {
        exp_sum += temp[i];
    }

    // Process remaining elements (tail loop)
    for (int i = (nnz / 16) * 16; i < nnz; ++i) {
        float shifted_score = score[i] - max_val;
        float exp_score = expf(shifted_score);
        score[i] = exp_score;
        exp_sum += exp_score;
    }

    // Step 4: Normalize scores and store
    __m512 sum = _mm512_set1_ps(exp_sum);
    for (int i = 0; i < nnz / 16; ++i) {
        __m512 scores = _mm512_loadu_ps(score + i * 16);
        __m512 normalized_scores = _mm512_div_ps(scores, sum);
        _mm512_storeu_ps(score + i * 16, normalized_scores);
    }

    // Normalize tail elements
    for (int i = (nnz / 16) * 16; i < nnz; ++i) {
        score[i] /= exp_sum;
    }

    // Step 5: Compute final max_value and expsum
    max_value[0] = max_val * M_LOG2E;
    expsum[0] = log2f(exp_sum) + max_value[0];
}

/**
 * @par Paper mapping
 * MagicPIG arXiv v4, Sections 4.3 and 4.4: selected CPU attention support.
 * @par Implementation note
 * Borrowed raw scores require exactly nnz floats: complete SIMD tiles and a
 * scalar tail use only logical candidates in maximum, sum and writes.
 * Positive finite head_dim_scale is sqrt(head_dim); metadata uses base-2.
 * Empty support performs no score access and reports -Inf metadata.
 * Native exp handles extreme shifts safely; finite input logits are required.
 * Exact-capacity sanitizer regressions never rely on allocated padding.
 * Support: TRACE_TEST_ID: MAGICPIG-SOFTMAX-BOUNDARY-CPU.
 * TRACE_TEST_ID: MAGICPIG-SOFTMAX-EXTREME-CPU.
 */
void softmax_kernel_optimized(
    float *score,
    const int nnz,
    const float head_dim_scale, // sqrt(head_dim)
    float *max_value,
    float *expsum) {
    // Empty support is the zero-mass domain: no score access, no probabilities.
    // Keep the ABI; -Inf base-2 metadata tells the caller this row has no mass.
    if (nnz == 0) {
        max_value[0] = -INFINITY;
        expsum[0] = -INFINITY;
        return;
    }


    const int full_lanes = (nnz / 16) * 16;
    const float inverse_scale = 1.0f / head_dim_scale;
    __m512 scale_vec = _mm512_set1_ps(inverse_scale);

    // Only complete tiles enter SIMD; the logical tail never reads padding.
    __m512 max_vec = _mm512_set1_ps(-INFINITY);
    for (int i = 0; i < full_lanes; i += 16) {
        __m512 s_vec = _mm512_mul_ps(_mm512_loadu_ps(score + i), scale_vec);
        max_vec = _mm512_max_ps(max_vec, s_vec);
    }
    float max_scalar = _mm512_reduce_max_ps(max_vec);
    for (int i = full_lanes; i < nnz; ++i)
        max_scalar = std::max(max_scalar, score[i] * inverse_scale);

    __m512 exp_sum_vec = _mm512_setzero_ps();
    __m512 max_scalar_vec = _mm512_set1_ps(max_scalar);
    for (int i = 0; i < full_lanes; i += 16) {
        __m512 s_vec = _mm512_mul_ps(_mm512_loadu_ps(score + i), scale_vec);
        __m512 exp_vec = avx512_exp_ps(_mm512_sub_ps(s_vec, max_scalar_vec));
        _mm512_storeu_ps(score + i, exp_vec);
        exp_sum_vec = _mm512_add_ps(exp_sum_vec, exp_vec);
    }
    float exp_sum_scalar = _mm512_reduce_add_ps(exp_sum_vec);
    for (int i = full_lanes; i < nnz; ++i) {
        score[i] = expf(score[i] * inverse_scale - max_scalar);
        exp_sum_scalar += score[i];
    }

    __m512 sum_vec = _mm512_set1_ps(exp_sum_scalar);
    for (int i = 0; i < full_lanes; i += 16) {
        __m512 normalized = _mm512_div_ps(_mm512_loadu_ps(score + i), sum_vec);
        _mm512_storeu_ps(score + i, normalized);
    }
    for (int i = full_lanes; i < nnz; ++i) score[i] /= exp_sum_scalar;

    // Save max_value and expsum
    max_value[0] = max_scalar * M_LOG2E;
    expsum[0] = log2f(exp_sum_scalar) + max_value[0];
}


/**
 * @par Paper mapping
 * MagicPIG arXiv v4, Sections 4.3 and 4.4: selected CPU attention support.
 * @par Implementation note
 * Nonnegative nnz, finite moderate logits, dimension scale positive. Borrowed
 * scores require exactly nnz floats; scalar tails do not read/write padding.
 * Empty support never touches scores and reports -Inf metadata. Returned
 * max and LSE use base2; no raw exponential sum or natural-log claim.
 * Support: see MAGICPIG-SOFTMAX-BOUNDARY-CPU in the sparse Softmax contract.
 */
void softmax_kernel_full(
float *score, 
const int nnz,
float sqrt_dim,
float *max_value,
float *expsum
) {
    // Empty support is the zero-mass domain: no score access, no probabilities.
    // Keep the ABI; -Inf base-2 metadata tells the caller this row has no mass.
    if (nnz == 0) {
        max_value[0] = -INFINITY;
        expsum[0] = -INFINITY;
        return;
    }

  const int full_tiles = nnz / 16;
  __m512 hdiv = _mm512_set1_ps(sqrt_dim);
  for (int i = 0; i < full_tiles; ++i){
        __m512 s_i = _mm512_loadu_ps(score + 16 * i);
        s_i = _mm512_div_ps(s_i, hdiv);
        _mm512_storeu_ps(score + 16 * i, s_i);
    }
  for (int i = full_tiles * 16; i < nnz; ++i) score[i] /= sqrt_dim;
  float m = *std::max_element(score,score+nnz);
  float exp_sum = 0.f;
  for (int i = 0; i < nnz; ++i){
      score[i] = expf32(score[i] - m);
      exp_sum += score[i];
    }

  __m512 sum = _mm512_set1_ps(exp_sum);
  for (int i = 0; i < full_tiles; ++i){
        __m512 s_i = _mm512_loadu_ps(score + 16 * i);
        s_i = _mm512_div_ps(s_i, sum);
        _mm512_storeu_ps(score + 16 * i, s_i);
    }
  
  for (int i = full_tiles * 16; i < nnz; ++i) score[i] /= exp_sum;

  max_value[0] = m * M_LOG2E;
  expsum[0] = log2f32(exp_sum) + max_value[0];
}

/**
 * @par Paper mapping
 * MagicPIG arXiv v4, Sections 4.3 and 4.4: selected CPU attention support.
 * @par Implementation note
 * Borrowed BF16 values, supplied candidate multiset and FP32 probabilities;
 * output dimension divisible by16. FP32 FMA accumulation precedes BF16
 * conversion. Dependency model adds raw FP32 bits0x8000 before shifting16;
 * it is not ties-even, nor evidence of real FBGEMM linkage/dispatch.
 * Independent FP64 oracle separates quantization, accumulation and rounding.
 * Support: TRACE_TEST_ID: MAGICPIG-BF16-MODEL-CPU.
 */
void wv_kernel(
const bfloat16 *value, 
const int *ind,
const float *attn_weight, 
bfloat16 *output, 
const int HEAD_DIM,
const int nnz) {
    if (HEAD_DIM <= 0 || HEAD_DIM % 16 != 0) throw std::invalid_argument("WV head_dim must be a positive multiple of 16");

    float y_f32[16];
    float w_f32[16];
    for (int io = 0; io < (HEAD_DIM / 16); ++io) {
        memset(y_f32, 0, sizeof(float) * 16);
        __m512 y_acc = _mm512_loadu_ps(y_f32);
        for(int jo = 0; jo < nnz; ++jo)
          {
            const int row = ind[jo];
            __m512 x_j = _mm512_set1_ps(attn_weight[jo]);

            Bfloat16ToFloat_avx512(value + row * HEAD_DIM + io * 16, w_f32, 16);
            __m512 w_tile = _mm512_loadu_ps(w_f32);
            //__m512 w_tile = _mm512_loadu_ps(value + row * HEAD_DIM + io * 16);
            y_acc = _mm512_fmadd_ps(w_tile, x_j, y_acc);
          }
        
        _mm512_storeu_ps(y_f32, y_acc);
        FloatToBfloat16_avx512(y_f32, output + io * 16, 16);
    }
}


void wv_kernel_dim128(
const bfloat16 *value, 
const int *ind,
const float *attn_weight, 
bfloat16 *output, 
const int HEAD_DIM,
const int nnz) {
    if (HEAD_DIM != 128) throw std::invalid_argument("fixed WV head_dim must be 128");


    __m512 y[8];
    float v_f32[16];
    float y_f32[16];
    memset(v_f32, 0, 16 * sizeof(float));
    memset(y_f32, 0, 16 * sizeof(float));
    for (int i = 0; i < 8; ++i){
        y[i] = _mm512_setzero_ps();
    }
    for (int i = 0; i < nnz; ++i){
        __m512 w_i = _mm512_set1_ps(attn_weight[i]);
        const int row = ind[i];

        for (int j = 0; j < 8; ++j){
            Bfloat16ToFloat_avx512(value + row * HEAD_DIM + j * 16, v_f32, 16);
            __m512 v_tile = _mm512_loadu_ps(v_f32);
             y[j] = _mm512_fmadd_ps(v_tile, w_i, y[j]);

        }

    }
    for (int i = 0; i < 8; ++i){
         _mm512_storeu_ps(y_f32, y[i]);
          FloatToBfloat16_avx512(y_f32, output + i * 16, 16);

    }
}


void wv_kernel_dim128_full(
    const bfloat16 *value,        // value 的输入，形状为 max_length x 128
    const float *attn_weight,     // attn_weight 的输入，形状为 num_q x max_length
    bfloat16 *output,             // 输出，形状为 num_q x 128
    const int HEAD_DIM,           // 每个向量的维度，这里为 128
    const int max_length,         // value 的行数
    const int nnz,                // 每行前 nnz 的非零部分
    const int num_q               // 查询向量的数量，必须是 1, 4 或 8
) {
    if (HEAD_DIM != 128 || (num_q != 1 && num_q != 4 && num_q != 8)) throw std::invalid_argument("full WV requires head_dim 128 and groups 1, 4 or 8");


    // 全局结果数组
    __m512 y[8][8]; // 每个查询向量最多分成 8 个块（128维）
    for (int q = 0; q < num_q; ++q) {
        for (int i = 0; i < 8; ++i) {
            y[q][i] = _mm512_setzero_ps();
        }
    }

    // 临时数组分配放在线程中，避免 false sharing
    #pragma omp parallel num_threads(ATTENTION_THREADS)
    {
        __m512 y_local[8][8]; // 每个线程的局部累积结果
        float v_f32[16];      // bfloat16 转换到 float
        for (int q = 0; q < num_q; ++q) {
            for (int i = 0; i < 8; ++i) {
                y_local[q][i] = _mm512_setzero_ps(); // 初始化局部向量
            }
        }

        // 使用 OpenMP 并行化非零值主循环
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < nnz; ++i) {
            for (int j = 0; j < 8; ++j) {
                Bfloat16ToFloat_avx512(value + i * HEAD_DIM + j * 16, v_f32, 16);
                __m512 v_tile = _mm512_loadu_ps(v_f32);

                for (int q = 0; q < num_q; ++q) {
                    __m512 w_i = _mm512_set1_ps(attn_weight[q * max_length + i]);
                    y_local[q][j] = _mm512_fmadd_ps(v_tile, w_i, y_local[q][j]);
                }
            }
        }

        // 合并结果到全局变量
        #pragma omp critical
        {
            for (int q = 0; q < num_q; ++q) {
                for (int j = 0; j < 8; ++j) {
                    y[q][j] = _mm512_add_ps(y[q][j], y_local[q][j]);
                }
            }
        }
    }

    // 将结果转换回 bfloat16 并存储
    float y_f32[16]; // 临时存储 float 类型结果
    for (int q = 0; q < num_q; ++q) {
        for (int i = 0; i < 8; ++i) {
            _mm512_storeu_ps(y_f32, y[q][i]);
            FloatToBfloat16_avx512(y_f32, output + q * HEAD_DIM + i * 16, 16);
        }
    }
}




void wv_kernel_dim128_dim_parallel(
const bfloat16 *value, 
const int *ind,
const float *attn_weight, 
bfloat16 *output, 
const int offset,
const int COMPUTE_HEAD_DIM,
const int HEAD_DIM,
const int nnz) {
    if (HEAD_DIM != 128 || COMPUTE_HEAD_DIM != 64 || (offset != 0 && offset != 64)) throw std::invalid_argument("parallel fixed WV requires dim128 and aligned 64-coordinate halves");


    __m512 y[4];
    float v_f32[16];
    float y_f32[16];
    memset(v_f32, 0, 16 * sizeof(float));
    memset(y_f32, 0, 16 * sizeof(float));
    for (int i = 0; i < 4; ++i){
        y[i] = _mm512_setzero_ps();
    }
    for (int i = 0; i < nnz; ++i){
        __m512 w_i = _mm512_set1_ps(attn_weight[i]);
        const int row = ind[i];
        for (int j = 0; j < 4; ++j){
            Bfloat16ToFloat_avx512(value + row * HEAD_DIM + j * 16 + offset, v_f32, 16);
            __m512 v_tile = _mm512_loadu_ps(v_f32);
             y[j] = _mm512_fmadd_ps(v_tile, w_i, y[j]);

        }

    }
    for (int i = 0; i < 4; ++i){
         _mm512_storeu_ps(y_f32, y[i]);
          FloatToBfloat16_avx512(y_f32, output + i * 16 + offset, 16);

    }
}

void wv_kernel_dim_parallel(
const bfloat16 *value, 
const int *ind,
const float *attn_weight, 
bfloat16 *output, 
const int offset,
const int COMPUTE_HEAD_DIM,
const int HEAD_DIM,
const int nnz) {
    if (HEAD_DIM <= 0 || HEAD_DIM % 16 != 0 || COMPUTE_HEAD_DIM <= 0 || COMPUTE_HEAD_DIM % 16 != 0 || offset < 0 || offset % 16 != 0 || COMPUTE_HEAD_DIM > HEAD_DIM || offset > HEAD_DIM - COMPUTE_HEAD_DIM) throw std::invalid_argument("parallel WV requires aligned in-range dimensions");

    float y_f32[16];
    float w_f32[16];
    for (int io = 0; io < (COMPUTE_HEAD_DIM / 16); ++io) {
        memset(y_f32, 0, sizeof(float) * 16);
        __m512 y_acc = _mm512_loadu_ps(y_f32);
        for(int jo = 0; jo < nnz; ++jo)
          {
            const int row = ind[jo];
            __m512 x_j = _mm512_set1_ps(attn_weight[jo]);
            Bfloat16ToFloat_avx512(value + row * HEAD_DIM + io * 16 + offset, w_f32, 16);
            __m512 w_tile = _mm512_loadu_ps(w_f32);
            y_acc = _mm512_fmadd_ps(w_tile, x_j, y_acc);
          }
        
        _mm512_storeu_ps(y_f32, y_acc);
        FloatToBfloat16_avx512(y_f32, output + io * 16 + offset, 16);
    }
}

SparseAttentionServer::SparseAttentionServer(){
    this->allocated = false;
    #ifdef __AVX512BF16__
    std::cout << "USING AVX512 BFLOAT16" << std::endl;
    #else
    std::cout << "NOT USING AVX512 BFLOAT16" << std::endl;
    #endif
}

SparseAttentionServer::~SparseAttentionServer(){
    if(this->allocated){
        for (int i = 0; i < this->num_layers; ++i)
            {
                delete [] this->key_cache[i];
                delete [] this->value_cache[i];
                delete [] this->key_norm[i];
            }
        delete [] this->attention_score;
        delete [] this->query_buffer;
        delete [] this->output_buffer;
        this->allocated = false;
      
    }

}

void SparseAttentionServer::alloc(
int num_layers, 
int num_attention_heads, 
int num_key_value_heads, 
int head_dim, 
int batch_size, 
int max_length)
{
    if (this->allocated) throw std::invalid_argument("SparseAttentionServer alloc may only be called once");
    if (num_layers <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0 || num_attention_heads % num_key_value_heads != 0 || head_dim <= 0 || head_dim % 16 != 0 || batch_size <= 0 || max_length <= 0) throw std::invalid_argument("invalid sparse attention allocation dimensions");

    this->num_layers = num_layers;
    this->num_attention_heads = num_attention_heads;
    this->num_key_value_heads = num_key_value_heads;
    this->num_attention_groups = static_cast<int>(this->num_attention_heads/this->num_key_value_heads);
    this->head_dim = head_dim;
    this->batch_size = batch_size;
    this->max_length = max_length;
    this->require_transform = true;

    
    for(int i = 0; i < this->num_layers; ++i){
        bfloat16 * k = new bfloat16[this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim];
        bfloat16 * v = new bfloat16[this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim];

        memset(k, 0, this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim * sizeof(bfloat16));
        memset(v, 0, this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim * sizeof(bfloat16));

        float * kn = new float[this->batch_size * this->num_key_value_heads * this->max_length];
        memset(kn, 0, this->batch_size * this->num_key_value_heads * this->max_length * sizeof(float));
        this->key_cache.push_back(k);
        this->value_cache.push_back(v);
        this->key_norm.push_back(kn);
    }
    this->attention_score = new float[this->batch_size * this->num_attention_heads * this->max_length];
    memset(this->attention_score, 0, this->batch_size * this->num_attention_heads * this->max_length * sizeof(float));
    this->allocated = true;

    this->query_buffer = new float[ATTENTION_THREADS * this->head_dim];
    this->output_buffer = new bfloat16[ATTENTION_THREADS * this->head_dim];
}


void SparseAttentionServer::clear()
{

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for(int i = 0; i < this->num_layers; ++i){
        
        memset(this->key_cache[i], 0, this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim * sizeof(bfloat16));
        memset(this->value_cache[i], 0, this->batch_size * this->num_key_value_heads * this->max_length * this->head_dim * sizeof(bfloat16));
        memset(this->key_norm[i], 0, this->batch_size * this->num_key_value_heads * this->max_length * sizeof(float));
    
    }
    memset(this->attention_score, 0, this->batch_size * this->num_attention_heads * this->max_length * sizeof(float));
}


void SparseAttentionServer::fill(
int layer_id, 
int request_id, 
torch::Tensor k, 
torch::Tensor v, 
torch::Tensor kn)
{
    int stride = this->num_key_value_heads * this->max_length * this->head_dim;
    bfloat16 * key = this->key_cache[layer_id] + request_id * stride;
    bfloat16 * value = this->value_cache[layer_id] + request_id * stride;

    //float * key = this->key_cache[layer_id] + request_id * stride;
    //float * value = this->value_cache[layer_id] + request_id * stride;
    float * k_norm = this->key_norm[layer_id] + request_id * this->num_key_value_heads * this->max_length;

    int seq_len = k.size(1);

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->num_key_value_heads; ++i){
        memcpy(key + i * this->max_length * this->head_dim, static_cast<bfloat16 *>(k.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(bfloat16));
        memcpy(value + i * this->max_length * this->head_dim, static_cast<bfloat16 *>(v.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(bfloat16));

        //memcpy(key + i * this->max_length * this->head_dim, static_cast<float *>(k.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(float));
        //memcpy(value + i * this->max_length * this->head_dim, static_cast<float *>(v.data_ptr()) + i * seq_len * this->head_dim, seq_len * this->head_dim * sizeof(float));
        memcpy(k_norm + i * this->max_length, static_cast<float *>(kn.data_ptr()) + i * seq_len, seq_len * sizeof(float));
    }
}

void SparseAttentionServer::attention_wrapper(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{

#ifdef __AVX512BF16__
this->attention_wrapper_bf16(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
#else
if (this->batch_size * this->num_attention_heads == 32 && ATTENTION_THREADS == 64)
{
    this->scheduled_attention(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
}
else if (this->batch_size * this->num_attention_heads > ATTENTION_THREADS){
    this->dynamic_attention(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
} else
{
    this->attention(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
}
#endif

};
#ifdef __AVX512BF16__
void SparseAttentionServer::attention_wrapper_bf16(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{

if (this->batch_size * this->num_attention_heads == 32 && ATTENTION_THREADS == 64)
{
    this->scheduled_attention_bf16(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
}
else if (this->batch_size * this->num_attention_heads > ATTENTION_THREADS){
    this->dynamic_attention_bf16(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
} else
{
    this->attention_bf16(
    layer_id, 
    K,
    L,
    output_pt,
    max_value_expsum_pt,
    query_pt, 
    query_norm_pt,
    ind_pt, 
    nnz_pt);
}


};
#endif

void SparseAttentionServer::dynamic_attention(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    auto q_f32 = query_pt.to(torch::kFloat32);
    float * query = static_cast<float *>(q_f32.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(dynamic,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        qk_kernel(
          key + (i/this->num_attention_groups) * this->max_length * this->head_dim,
          ind + i * this->max_length,
          query + i * this->head_dim,
          this->attention_score + i * this->max_length,
          this->head_dim,
          nnz[i]
        );
        transform_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          query_norm[i],
          this->key_norm[layer_id] + (i/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + i * this->max_length
        );
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
        wv_kernel(
            value + (i/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + i * this->max_length,
            this->attention_score + i * this->max_length,
            output + i * this->head_dim,
            this->head_dim,
            nnz[i]
        );
      }

}
#ifdef __AVX512BF16__
void SparseAttentionServer::dynamic_attention_bf16(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    if (this->head_dim <= 0 || this->head_dim % 32 != 0) throw std::invalid_argument("BF16 attention head_dim must be a positive multiple of 32");

    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    bfloat16 * query = static_cast<bfloat16 *>(query_pt.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(dynamic,1) num_threads(64)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        qk_kernel_bf16_impl(
          key + (i/this->num_attention_groups) * this->max_length * this->head_dim,
          ind + i * this->max_length,
          query + i * this->head_dim,
          this->attention_score + i * this->max_length,
          this->head_dim,
          nnz[i]
        );
        transform_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          query_norm[i],
          this->key_norm[layer_id] + (i/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + i * this->max_length
        );
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
        wv_kernel(
            value + (i/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + i * this->max_length,
            this->attention_score + i * this->max_length,
            output + i * this->head_dim,
            this->head_dim,
            nnz[i]
        );
      }

}
#endif
/**
 * @par Paper mapping
 * MagicPIG arXiv v4, Sections 4.3 and 4.4: selected CPU attention support.
 * @par Implementation note
 * CPU selected-body composition only: alloc once, fill within capacity, valid
 * layer/head/request IDs, rounded candidate capacity, server outlives borrowed
 * getter views. Static/dynamic queries are already FP32 in the Tensor shim;
 * no Torch conversion/ABI or OpenMP scheduling is established. Supported
 * sparse dimensions16/32/128 and candidate multisets are compared with an
 * independent dense FP64 oracle on the same supplied subset. Full path uses
 * dim128/groups1/4/8/uniform nnz; varying GQA lengths reproduce repair #77.
 * All-candidate corrected sparse differs from uncorrected dense by design.
 * Support: TRACE_TEST_ID: MAGICPIG-ATTENTION-CPU.
 */
void SparseAttentionServer::attention(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    auto q_f32 = query_pt.to(torch::kFloat32);
    float * query = static_cast<float *>(q_f32.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        qk_kernel(
          key + (i/this->num_attention_groups) * this->max_length * this->head_dim,
          ind + i * this->max_length,
          query + i * this->head_dim,
          this->attention_score + i * this->max_length,
          this->head_dim,
          nnz[i]
        );
        transform_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          query_norm[i],
          this->key_norm[layer_id] + (i/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + i * this->max_length
        );
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
        wv_kernel(
            value + (i/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + i * this->max_length,
            this->attention_score + i * this->max_length,
            output + i * this->head_dim,
            this->head_dim,
            nnz[i]
        );
      }

}

#ifdef __AVX512BF16__
void SparseAttentionServer::attention_bf16(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    if (this->head_dim <= 0 || this->head_dim % 32 != 0) throw std::invalid_argument("BF16 attention head_dim must be a positive multiple of 32");

    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    
    bfloat16 * query = static_cast<bfloat16 *>(query_pt.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(static,1) num_threads(64)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        qk_kernel_bf16_impl(
          key + (i/this->num_attention_groups) * this->max_length * this->head_dim,
          ind + i * this->max_length,
          query + i * this->head_dim,
          this->attention_score + i * this->max_length,
          this->head_dim,
          nnz[i]
        );
        transform_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          query_norm[i],
          this->key_norm[layer_id] + (i/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + i * this->max_length
        );
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
        wv_kernel(
            value + (i/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + i * this->max_length,
            this->attention_score + i * this->max_length,
            output + i * this->head_dim,
            this->head_dim,
            nnz[i]
        );
      }

}
#endif
void SparseAttentionServer::full_attention(
int layer_id, 
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt,
torch::Tensor nnz_pt)
{
    if (!this->allocated || this->head_dim != 128 || (this->num_attention_groups != 1 && this->num_attention_groups != 4 && this->num_attention_groups != 8)) throw std::invalid_argument("full_attention requires dim128 and query groups 1, 4 or 8");

    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    auto q_f32 = query_pt.to(torch::kFloat32);
    float * query = static_cast<float *>(q_f32.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;
    for (int i = 0; i < this->batch_size * this->num_key_value_heads; ++i){
        qk_kernel_full(
          key + i * this->max_length * this->head_dim,
          query + i * this->head_dim * this->num_attention_groups,
          this->attention_score + i * this->max_length * this->num_attention_groups,
          this->num_attention_groups,
          this->head_dim,
          nnz[i],
          this->max_length
        );
    }
    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        softmax_kernel_optimized(
          this->attention_score + i * this->max_length,
          nnz[i],
          sqrtf32(this->head_dim),
          max_value + i,
          expsum + i
        );
      }

    for (int i = 0; i < this->batch_size * this->num_key_value_heads; ++i){
        wv_kernel_dim128_full(
            value + i * this->max_length * this->head_dim,
            this->attention_score + i * this->max_length * this->num_attention_groups,
            output + i * this->head_dim * this->num_attention_groups,
            this->head_dim,
            this->max_length,
            nnz[i],
            this->num_attention_groups
        );
      }

}

void SparseAttentionServer::scheduled_attention(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    if (this->head_dim != 128) throw std::invalid_argument("scheduled_attention requires head_dim 128");

    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    auto q_f32 = query_pt.to(torch::kFloat32);
    float * query = static_cast<float *>(q_f32.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < ATTENTION_THREADS; ++i){
        int p = int (i / 2);
        int offset = int(nnz[p] / 2);
        int offset_16 = (offset % 16 == 0) ? (offset): (offset + 16 - offset % 16);
        int ith_offset = (i % 2 == 0) ? (0) : (offset_16);
        int workload = (i % 2 == 0) ? (offset_16) : (nnz[p] - offset_16);
        if (workload > 0){
          qk_kernel(
            key + (p/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + p * this->max_length + ith_offset,
            query + p * this->head_dim,
            this->attention_score + p * this->max_length + ith_offset,
            this->head_dim,
            workload
          );

          transform_kernel(
          this->attention_score + p * this->max_length + ith_offset,
          workload,
          query_norm[p],
          this->key_norm[layer_id] + (p/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + p * this->max_length + ith_offset
        );
        }
      }
    

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
    }

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < ATTENTION_THREADS; ++i){

      int p = int (i / 2);
      int ith_offset = (i % 2 == 0) ? (0) : int(this->head_dim / 2);



      wv_kernel_dim128_dim_parallel(
            value + (p/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + p * this->max_length,
            this->attention_score + p * this->max_length,
            output + p * this->head_dim,
            ith_offset,
            int(this->head_dim / 2),
            this->head_dim,
            nnz[p]
        );
      
    }



}
#ifdef __AVX512BF16__
void SparseAttentionServer::scheduled_attention_bf16(
int layer_id, 
int K,
int L,
torch::Tensor output_pt,
torch::Tensor max_value_expsum_pt,
torch::Tensor query_pt, 
torch::Tensor query_norm_pt,
torch::Tensor ind_pt, 
torch::Tensor nnz_pt)
{
    if (this->head_dim != 128) throw std::invalid_argument("scheduled_attention requires head_dim 128");

    if (this->head_dim <= 0 || this->head_dim % 32 != 0) throw std::invalid_argument("BF16 attention head_dim must be a positive multiple of 32");

    bfloat16 * key = this->key_cache[layer_id];
    bfloat16 * value = this->value_cache[layer_id];
    bfloat16 * query = static_cast<bfloat16 *>(query_pt.data_ptr());
    bfloat16 * output = static_cast<bfloat16 *>(output_pt.data_ptr());
    int * ind = static_cast<int *>(ind_pt.data_ptr());
    int * nnz = static_cast<int *>(nnz_pt.data_ptr());
    float * query_norm = static_cast<float *>(query_norm_pt.data_ptr());
    float * max_value =  static_cast<float *>(max_value_expsum_pt.data_ptr());
    float * expsum = max_value + this->batch_size * this->num_attention_heads;

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < ATTENTION_THREADS; ++i){
        int p = int (i / 2);
        int offset = int(nnz[p] / 2);
        int offset_16 = (offset % 16 == 0) ? (offset): (offset + 16 - offset % 16);
        int ith_offset = (i % 2 == 0) ? (0) : (offset_16);
        int workload = (i % 2 == 0) ? (offset_16) : (nnz[p] - offset_16);

        if (workload > 0){
          qk_kernel_bf16_impl(
            key + (p/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + p * this->max_length + ith_offset,
            query + p * this->head_dim,
            this->attention_score + p * this->max_length + ith_offset,
            this->head_dim,
            workload
          );

          transform_kernel(
          this->attention_score + p * this->max_length + ith_offset,
          workload,
          query_norm[p],
          this->key_norm[layer_id] + (p/this->num_attention_groups) * this->max_length, 
          K,
          L,
          sqrtf(this->head_dim),
          ind + p * this->max_length + ith_offset
        );
        }
      }
    

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < this->batch_size * this->num_attention_heads; ++i){
        softmax_kernel(
          this->attention_score + i * this->max_length,
          nnz[i],
          max_value + i,
          expsum + i
        );
    }

    #pragma omp parallel for schedule(static,1) num_threads(ATTENTION_THREADS)
    for (int i = 0; i < ATTENTION_THREADS; ++i){

      int p = int (i / 2);
      int ith_offset = (i % 2 == 0) ? (0) : int(this->head_dim / 2);
     
      wv_kernel_dim128_dim_parallel(
            value + (p/this->num_attention_groups) * this->max_length * this->head_dim,
            ind + p * this->max_length,
            this->attention_score + p * this->max_length,
            output + p * this->head_dim,
            ith_offset,
            int(this->head_dim/2),
            this->head_dim,
            nnz[p]
        );
      
    }



}
#endif
torch::Tensor SparseAttentionServer::get_key_cache(int layer_id)
{
auto options = torch::TensorOptions().dtype(torch::kBFloat16);
//auto options = torch::TensorOptions().dtype(torch::kFloat32);
torch::Tensor tensor = torch::from_blob(this->key_cache[layer_id], {this->batch_size, this->num_key_value_heads, this->max_length, this->head_dim}, options);
return tensor;
}

torch::Tensor SparseAttentionServer::get_value_cache(int layer_id)
{
auto options = torch::TensorOptions().dtype(torch::kBFloat16);
//auto options = torch::TensorOptions().dtype(torch::kFloat32);
torch::Tensor tensor = torch::from_blob(this->value_cache[layer_id], {this->batch_size, this->num_key_value_heads, this->max_length, this->head_dim}, options);
return tensor;
}

torch::Tensor SparseAttentionServer::get_key_norm(int layer_id)
{
auto options = torch::TensorOptions().dtype(torch::kFloat32);
torch::Tensor tensor = torch::from_blob(this->key_norm[layer_id], {this->batch_size, this->num_key_value_heads, this->max_length}, options);
return tensor;
}

torch::Tensor SparseAttentionServer::get_score()
{
auto options = torch::TensorOptions().dtype(torch::kFloat32);
torch::Tensor tensor = torch::from_blob(this->attention_score, {this->batch_size, this->num_attention_heads, this->max_length}, options);
return tensor;
}

PYBIND11_MODULE(sparse_attention_cpu, m) {
    py::class_<SparseAttentionServer>(m, "SparseAttentionServer")
        .def(py::init<>())
        .def("alloc", &SparseAttentionServer::alloc)
        .def("fill", &SparseAttentionServer::fill)
        .def("attention", &SparseAttentionServer::attention)
        #ifdef __AVX512BF16__
        .def("attention_bf16", &SparseAttentionServer::attention_bf16)
        #endif
        .def("full_attention", &SparseAttentionServer::full_attention)
        .def("scheduled_attention", &SparseAttentionServer::scheduled_attention)
        .def("attention_wrapper", &SparseAttentionServer::attention_wrapper)
        #ifdef __AVX512BF16__
        .def("attention_wrapper_bf16", &SparseAttentionServer::attention_wrapper_bf16)
        #endif
        .def("get_key_cache", &SparseAttentionServer::get_key_cache)
        .def("get_value_cache", &SparseAttentionServer::get_value_cache)
        .def("get_key_norm", &SparseAttentionServer::get_key_norm)
        .def("get_score", &SparseAttentionServer::get_score)
        .def("clear", &SparseAttentionServer::clear);
}
