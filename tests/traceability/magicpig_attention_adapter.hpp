#pragma once
// Explicit std-only dependency models. No Torch ABI, allocator, dtype conversion,
// FBGEMM dispatch, intrinsic conversion or OpenMP scheduling is exercised here.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#define ATTENTION_THREADS 64
using bfloat16 = std::uint16_t;
namespace magicpig_attention_model {
inline float decode(bfloat16 x) {
    const std::uint32_t bits = std::uint32_t(x) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof result);
    return result;
}
inline bfloat16 encode(float x) {
    std::uint32_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    // Pinned FBGEMM finite-value policy: add 0x8000 then shift, not ties-even.
    return bfloat16((bits + 0x8000u) >> 16);
}
}
inline void Bfloat16ToFloat_avx512(const bfloat16* src, float* dst, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) dst[i] = magicpig_attention_model::decode(src[i]);
}
inline void FloatToBfloat16_avx512(const float* src, bfloat16* dst, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) dst[i] = magicpig_attention_model::encode(src[i]);
}
#ifndef MAGIC_NATIVE_AVX512
inline __m512 _mm512_mul_ps(__m512 a, __m512 b) {
    for (int i = 0; i < 16; ++i) a.lane[i] *= b.lane[i];
    return a;
}
inline __m512 _mm512_fmadd_ps(__m512 a, __m512 b, __m512 c) {
    for (int i = 0; i < 16; ++i) c.lane[i] = std::fma(a.lane[i], b.lane[i], c.lane[i]);
    return c;
}
inline float _mm512_reduce_add_ps(__m512 a) {
    float result = 0;
    for (float x : a.lane) result += x;
    return result;
}
inline __m512 _mm512_maskz_loadu_ps(unsigned mask, const float* src) {
    auto result = _mm512_setzero_ps();
    for (int i = 0; i < 16; ++i) if (mask & (1u << i)) result.lane[i] = src[i];
    return result;
}
inline void _mm512_mask_storeu_ps(float* dst, unsigned mask, __m512 value) {
    for (int i = 0; i < 16; ++i) if (mask & (1u << i)) dst[i] = value.lane[i];
}
inline __m512 _mm512_max_ps(__m512 a, __m512 b) {
    for (int i = 0; i < 16; ++i) a.lane[i] = std::max(a.lane[i], b.lane[i]);
    return a;
}
inline float _mm512_reduce_max_ps(__m512 a) {
    return *std::max_element(a.lane.begin(), a.lane.end());
}
#endif
namespace torch {
constexpr int kBFloat16 = 1;
constexpr int kFloat32 = 2;
struct TensorOptions { TensorOptions dtype(int) const { return *this; } };
class Tensor {
    void* data_;
    std::vector<int64_t> shape_;
public:
    Tensor(void* data, std::initializer_list<int64_t> shape) : data_(data), shape_(shape) {
        if (!data) throw std::invalid_argument("null model tensor");
        for (auto dim : shape_) if (dim < 0) throw std::invalid_argument("negative model dimension");
    }
    int64_t size(std::size_t i) const { return shape_.at(i); }
    void* data_ptr() const { return data_; }
    Tensor to(int dtype) const {
        if (dtype != kFloat32) throw std::invalid_argument("unsupported model conversion");
        return *this; // Fixtures already provide contiguous float32 queries.
    }
};
inline Tensor from_blob(void* data, std::initializer_list<int64_t> shape, TensorOptions) {
    return Tensor(data, shape); // Non-owning view; fixture never outlives server.
}
}
// Declaration/layout mirrors maintained header, with only dependency includes replaced.
class SparseAttentionServer {
public:
    SparseAttentionServer();
    ~SparseAttentionServer();
    void alloc(int, int, int, int, int, int);
    void fill(int, int, torch::Tensor, torch::Tensor, torch::Tensor);
    void clear();
    void attention(int, int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
    void dynamic_attention(int, int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
    void full_attention(int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
    torch::Tensor get_key_cache(int);
    torch::Tensor get_value_cache(int);
    torch::Tensor get_key_norm(int);
    torch::Tensor get_score();
private:
    int num_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int num_attention_groups;
    int batch_size;
    int max_length;
    bool require_transform;
    bool allocated;
    std::vector<bfloat16*> key_cache;
    std::vector<bfloat16*> value_cache;
    std::vector<float*> key_norm;
    float* attention_score;
    float* query_buffer;
    bfloat16* output_buffer;
};
