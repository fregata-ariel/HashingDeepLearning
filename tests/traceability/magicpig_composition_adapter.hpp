#pragma once
// Combine the existing explicit host dependency substitutes, with one Tensor ABI.
#include "magicpig_attention_adapter.hpp"
#include <cassert>
#ifndef TORCH_CHECK
#define TORCH_CHECK(condition, message) do { if (!(condition)) throw std::invalid_argument(message); } while (false)
#endif
#define LSH_THREADS 64
inline void fast_memcpy_avx512(int* dst, const int* src, size_t n) { std::copy_n(src,n,dst); }
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
