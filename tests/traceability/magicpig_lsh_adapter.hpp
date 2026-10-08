#pragma once
// Explicit dependency substitute: contiguous, caller-owned buffers only. This is
// not Torch's allocator, dtype validation, dispatcher, ABI, device or threading.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#define LSH_THREADS 64
namespace torch {
constexpr int kInt8 = 1;
struct TensorOptions { TensorOptions dtype(int) const { return *this; } };
class Tensor {
    void* data_;
    std::vector<int64_t> dimensions_;
public:
    Tensor(void* data, std::initializer_list<int64_t> dimensions)
        : data_(data), dimensions_(dimensions) {
        if (!data_) throw std::invalid_argument("null tensor storage");
        for (auto d : dimensions_) if (d < 0) throw std::invalid_argument("negative tensor dimension");
    }
    int64_t size(size_t dim) const { return dimensions_.at(dim); }
    void* data_ptr() const { return data_; }
};
inline Tensor from_blob(void* p, std::initializer_list<int64_t> shape, TensorOptions) {
    return Tensor(p, shape); // Non-owning alias: caller must outlive view.
}
}
#ifndef MAGICPIG_NATIVE_LSH_COPY
inline void fast_memcpy_avx512(int* dest, const int* src, size_t count) {
    std::copy_n(src, count, dest); // Explicit portable replacement, never native claim.
}
#endif
// The maintained lsh.h declaration, with dependency includes replaced and
// retrieve exposed for selected-body testing. No maintained function body changes.
class LSH {
public:
    LSH(); ~LSH();
    void alloc(int K, int L, int num_layers, int num_attention_heads, int num_key_value_heads, int batch_size, int max_length);
    void fill(int layer_id, int request_id, torch::Tensor sorted_hash_code_pt, torch::Tensor sorted_indices_pt);
    void fastfill(int layer_id, int request_id, torch::Tensor hash_code_pt);
    void clear();
    void batch_retrieve(int layer_id, torch::Tensor query_pt, torch::Tensor results_pt, torch::Tensor nnz_pt);
    torch::Tensor get_mask();
    int retrieve(int layer_id, int head_id, const int* __restrict query, int* __restrict results);
private:
    int K, L, num_buckets, num_layers, num_attention_heads, num_key_value_heads;
    int num_attention_groups, batch_size, max_length;
    bool allocated;
    uint8_t* mask;
    std::vector<int*> table_start, table_end, table;
};
