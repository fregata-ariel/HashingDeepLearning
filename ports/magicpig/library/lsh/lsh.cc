#include "lsh.h"

static inline void fast_memcpy_avx512(int* dest, const int* src, size_t count) {
    const size_t avx512_width = 16; // Number of 32-bit integers in 512 bits
    size_t i = 0;

    // Calculate the number of full AVX-512 vectors to process
    size_t vec_count = count / avx512_width;

    // Process data in AVX-512 chunks
    for (; i < vec_count * avx512_width; i += avx512_width) {
        // Load 16 integers from src (unaligned)
        __m512i data = _mm512_loadu_si512(reinterpret_cast<const void*>(src + i));

        // Store 16 integers to dest (unaligned)
        _mm512_storeu_si512(reinterpret_cast<void*>(dest + i), data);
    }

    // Handle remaining elements that don't fit into a full AVX-512 vector
    for (; i < count; ++i) {
        dest[i] = src[i];
    }
}

LSH::LSH(){
    this->allocated = false;
}

LSH::~LSH(){

    if (this->allocated) {
        for (int i = 0; i < this->num_layers; ++i){
            delete [] this->table_start[i];
            delete [] this->table_end[i];
            delete [] this->table[i];
        }
        delete [] this->mask;
        //cudaFreeHost(this->query_buffer);
        //cudaStreamDestroy(this->stream);
    }
    this->allocated = false;
}

/**
 * @par Implementation note
 * Allocate once per owner, with valid positive/divisible head dimensions and
 * bounded K/length products. Reallocation and unchecked integer overflow are
 * outside the selected CPU contract. Torch ABI and OpenMP remain unverified.
 * Accepted K is 1..11: all 2^K codes are exactly representable by the archived
 * FP16 key/query packing. Reject K outside that domain before changing owner
 * state or computing bucket counts (Issue #65 maintained rejection policy).
 * @par Traceability relation
 * Variant; TRACE_TEST_ID: MAGICPIG-HASH-DOMAIN-CPU.
 */
void LSH::alloc(
int K, 
int L, 
int num_layers, 
int num_attention_heads, 
int num_key_value_heads, 
int batch_size,
int max_length)
{
    TORCH_CHECK(K >= 1 && K <= 11, "LSH K must be in [1,11] for exact FP16 hash packing");
    this->K = K;
    this->L = L;
    this->num_buckets = static_cast<int>(pow(2, this->K));
    this->num_layers = num_layers;
    this->num_attention_heads = num_attention_heads;
    this->num_key_value_heads = num_key_value_heads;
    this->batch_size = batch_size;
    this->max_length = max_length;
    this->num_attention_groups = static_cast<int>(this->num_attention_heads / this->num_key_value_heads);
    for (int i = 0; i < this->num_layers; ++i){

        int * start = new int [this->batch_size * this->num_key_value_heads * this->L * this->num_buckets]; 
        memset(start, 0, this->batch_size * this->num_key_value_heads * this->L * this->num_buckets * sizeof(int));
        this->table_start.push_back(start);

        int * end = new int [this->batch_size * this->num_key_value_heads * this->L * this->num_buckets]; 
        memset(end, 0, this->batch_size * this->num_key_value_heads * this->L * this->num_buckets * sizeof(int));
        this->table_end.push_back(end);

        int * content = new int [this->batch_size * this->num_key_value_heads * this->L * this->max_length]; 
        memset(content, 0, this->batch_size * this->num_key_value_heads * this->L * this->max_length * sizeof(int));
        this->table.push_back(content);

        // std::vector<std::vector<std::vector<std::vector<int>>>> ht_li(this->batch_size * this->num_key_value_heads, 
        // std::vector<std::vector<std::vector<int>>>(this->L, 
        // std::vector<std::vector<int>>(this->num_buckets, std::vector<int>(0))));
        // this->ht.push_back(ht_li);

    }

    
    this->mask = new uint8_t [this->batch_size * this->num_attention_heads * this->max_length];
    memset(mask, 0, this->batch_size * this->num_attention_heads * this->max_length * sizeof(uint8_t));
    //cudaHostAlloc((void**)&query_buffer, this->batch_size * this->num_attention_heads * this->L * sizeof(int), cudaHostAllocDefault);
    //cudaStreamCreate(&this->stream);
    this->allocated = true;

}

/**
 * @par Paper mapping
 * MagicPIG v4 Section 4.3: alternate candidate-table construction support.
 * @par Implementation note
 * Require an allocated owner, valid layer/request, and contiguous CPU int32
 * codes of shape [KV heads, L, sequence length], with length <= max_length and
 * every code in [0, 2^K). Validate the whole input before changing owner state.
 * Replace only the selected layer/request's tables, including empty refill;
 * stable counting placement copies token IDs in increasing order per bucket.
 * Temporary bucket cursors have scoped vector ownership (Issue #66 repair).
 * CPU selected-body tests do not establish real Torch ABI or GPU behavior.
 * @par Traceability relation
 * Variant; TRACE_TEST_ID: MAGICPIG-FASTFILL-REPAIR-CPU.
 * Support regression; TRACE_TEST_ID: MAGICPIG-FASTFILL-REGRESSION-CPU.
 */
void LSH::fastfill(
int layer_id, 
int request_id, 
torch::Tensor hash_code_pt
)
{
TORCH_CHECK(this->allocated, "fastfill requires an allocated LSH owner");
TORCH_CHECK(layer_id >= 0 && layer_id < this->num_layers, "invalid fastfill layer");
TORCH_CHECK(request_id >= 0 && request_id < this->batch_size, "invalid fastfill request");
TORCH_CHECK(hash_code_pt.dim() == 3, "fastfill codes must have rank 3");
TORCH_CHECK(hash_code_pt.is_cpu() && hash_code_pt.is_contiguous(),
            "fastfill codes must be contiguous CPU tensors");
TORCH_CHECK(hash_code_pt.scalar_type() == torch::kInt32, "fastfill codes must be int32");
TORCH_CHECK(hash_code_pt.size(0) == this->num_key_value_heads && hash_code_pt.size(1) == this->L,
            "fastfill code head/table shape mismatch");
TORCH_CHECK(hash_code_pt.size(2) <= this->max_length, "fastfill sequence exceeds capacity");
int seq_len = static_cast<int>(hash_code_pt.size(2));
const int * hash_code = static_cast<const int *>(hash_code_pt.data_ptr());
for (int h = 0; h < this->num_key_value_heads; ++h)
    for (int j = 0; j < this->L; ++j)
        for (int token = 0; token < seq_len; ++token) {
            const int code = hash_code[(h * this->L + j) * seq_len + token];
            TORCH_CHECK(code >= 0 && code < this->num_buckets, "fastfill hash code outside bucket range");
        }
int stride = this->num_key_value_heads * this->L * this->num_buckets;
int * start = this->table_start[layer_id] + request_id * stride;
int * end = this->table_end[layer_id] + request_id * stride;
int * content = this->table[layer_id] + request_id * this->num_key_value_heads * this->L * this->max_length;

for(int i = 0;  i < this->num_key_value_heads; ++i){
        const int * v_i = hash_code + i * (this->L * seq_len);
        int *ms_i = start + i * (this->L * num_buckets);
        int *me_i = end+ i * (this->L * num_buckets);
        for(int j=0; j < this->L; ++j)
            {
            const int * v_ij = v_i + j * seq_len;
            int *ms_ij = ms_i + j * num_buckets;
            int *me_ij = me_i + j * num_buckets;
            std::fill_n(ms_ij, num_buckets, 0);
            std::fill_n(me_ij, num_buckets, 0);
            for(int k=0; k<seq_len; ++k){
                const int v = v_ij[k];
                me_ij[v] += 1;
            }

            for(int b=1; b<this->num_buckets; ++b){
                ms_ij[b] =  me_ij[b-1];
                me_ij[b] += ms_ij[b]; 
            }
            std::vector<int> next(ms_ij, ms_ij + num_buckets);
            int * table_ij = content + (i * this->L + j) * this->max_length;
            for (int token = 0; token < seq_len; ++token)
                table_ij[next[v_ij[token]]++] = token;
            }
}


}
/**
 * @par Implementation note
 * CPU contiguous int16 sorted bucket codes and int32 per-table permutations
 * of valid token IDs are required. Indices are copied into owner storage;
 * clear before refill. Portable tests replace the AVX copy explicitly; native
 * copy is independently capability-gated. No Torch boundary validation added.
 */
void LSH::fill(
int layer_id, 
int request_id, 
torch::Tensor sorted_hash_code_pt,
torch::Tensor sorted_indices_pt
)
{

assert(layer_id >= 0);
assert(layer_id < this->num_layers);

assert(request_id >= 0);
assert(request_id < this->batch_size);

assert(sorted_hash_code_pt.size(0) == this->num_key_value_heads);
assert(sorted_hash_code_pt.size(1) == this->L);
assert(sorted_hash_code_pt.size(2) <= this->max_length);

int seq_len = sorted_hash_code_pt.size(2);
int stride = this->num_key_value_heads * this->L * this->num_buckets;
int16_t * sorted_hash_code = static_cast<int16_t *>(sorted_hash_code_pt.data_ptr());

int * start = this->table_start[layer_id] + request_id * stride;
int * end = this->table_end[layer_id] + request_id * stride;

#pragma omp parallel for collapse(2) schedule(static)
for(int j=0; j < this->L; ++j){
    for(int i = 0;  i < this->num_key_value_heads; ++i){
        const int16_t * v_i = sorted_hash_code + i * (this->L * seq_len);
        int *ms_i = start + i * (this->L * num_buckets);
        int *me_i = end+ i * (this->L * num_buckets);
        const int16_t * v_ij = v_i + j * seq_len;
        int *ms_ij = ms_i + j * num_buckets;
        int *me_ij = me_i + j * num_buckets;
        for(int k=0; k<seq_len; ++k){
            const int v = static_cast<int>(v_ij[k]);
            if(me_ij[v]==0) {
                ms_ij[v] = k;
                me_ij[v] = k+1;
            } else {
                me_ij[v] = me_ij[v] + 1;
            }
        }
            
    }
 }

assert(sorted_indices_pt.size(0) == this->num_key_value_heads);
assert(sorted_indices_pt.size(1) == this->L);
assert(sorted_indices_pt.size(2) == seq_len);

int * sorted_indices = static_cast<int *>(sorted_indices_pt.data_ptr());
int * content = this->table[layer_id] + request_id * this->num_key_value_heads * this->L * this->max_length;

#pragma omp parallel for schedule(static,1) num_threads(LSH_THREADS)
for (int i = 0; i < this->num_key_value_heads * this->L; ++i){
    fast_memcpy_avx512(content + i * max_length, sorted_indices + i * seq_len, seq_len);
}
}

void LSH::copy( 
torch::Tensor query_pt)
{
    //cudaMemcpyAsync(this->query_buffer, query_pt.data_ptr(), this->batch_size * this->num_attention_heads * this->L * sizeof(int), cudaMemcpyDeviceToHost, this->stream);
}


/**
 * @par CPU composition support
 * Borrowed contiguous int32 query codes [batch*query_heads,L], output IDs
 * [batch*query_heads,max_length], and counts [batch*query_heads]. The caller
 * owns buffers; each query head borrows its grouped KV tables. The connected
 * CPU fixture passes these actual IDs/counts into selected attention bodies
 * across append/query steps and clear/refill, with an independent union oracle.
 * Serial copy/Tensor/BF16 models and a host merge replace dependencies; this
 * does not verify Torch ABI, OpenMP scheduling, FlashInfer, or GPU transfers.
 * Support: TRACE_TEST_ID: MAGICPIG-COMPOSITION-BRIDGE-CPU.
 */
void LSH::batch_retrieve(
int layer_id, 
torch::Tensor query_pt, 
torch::Tensor results_pt, 
torch::Tensor nnz_pt)
{
    
int * query = static_cast<int *>(query_pt.data_ptr());

//int * query = this->query_buffer;
int * results = static_cast<int *>(results_pt.data_ptr());
int * nnz = static_cast<int *>(nnz_pt.data_ptr());

assert(query_pt.size(0) == this->batch_size * this->num_attention_heads);
assert(query_pt.size(1) == this->L);
assert(results_pt.size(0) == this->batch_size * this->num_attention_heads);
assert(results_pt.size(1) == this->max_length);
assert(nnz_pt.size(0) == this->batch_size * this->num_attention_heads);
//cudaStreamSynchronize(this->stream);
#pragma omp parallel for schedule(static,1) num_threads(LSH_THREADS)
for (int head_id = 0; head_id < this->batch_size * this->num_attention_heads; ++head_id)
{
    nnz[head_id] = this->retrieve(
        layer_id,
        head_id,
        query,
        results
    );
}


}

/**
 * @par Paper mapping
 * MagicPIG v4 Section 4.3 Equation 10: accept at least two table collisions.
 * @par Implementation note
 * Direct under valid sorted-fill/query inputs and unique IDs per table; the
 * saturating mask emits a token once and is reset between queries. Repeated
 * IDs within one table violate the distinct-table interpretation. Independent
 * CPU tests cover literal collision counts, capacity, clear/refill and GQA,
 * batch/layer isolation. OpenMP scheduling and Torch ABI are unexecuted.
 * @par Traceability relation
 * Direct; TRACE_TEST_ID: MAGICPIG-LSH-RETRIEVAL-CPU.
 */
int LSH::retrieve(
int layer_id, 
int head_id, 
const int* __restrict query, 
int* __restrict results)
{

        // 预计算 group_id 以减少冗余计算
        int group_id = head_id / this->num_attention_groups;
        int* __restrict start = this->table_start[layer_id] + group_id * this->L * this->num_buckets;
        int* __restrict end = this->table_end[layer_id] + group_id * this->L * this->num_buckets;
        int* __restrict content = this->table[layer_id] + group_id * this->L * this->max_length;
        const int* __restrict q = query + head_id * this->L;
        int* __restrict result = results + head_id * this->max_length;
        uint8_t* __restrict tmask = reinterpret_cast<uint8_t*>(mask + head_id * this->max_length);
        
        // 初始化 tmask
        memset(tmask, 0, this->max_length);
        
        int offset = 0;
        int* result_ptr = result;
        
        for (int i = 0; i < this->L; ++i){
            int q_i = q[i];
            int* __restrict m_start = start + i * this->num_buckets;
            int* __restrict m_end = end + i * this->num_buckets;
            int* __restrict m_content = content + i * this->max_length;
            int start_pos = m_start[q_i];
            int end_pos = m_end[q_i];
            for (int j = start_pos; j < end_pos; ++j){
                int idx = m_content[j];
                uint8_t mask_val = tmask[idx];
                // 提示编译器 mask_val == 0 是更可能的情况
                if (__builtin_expect(mask_val == 0, 1)){
                    tmask[idx] = 1;
                }
                else if (__builtin_expect(mask_val == 1, 0)){
                    tmask[idx] = 2;
                    *result_ptr++ = idx;
                }
            }
        }
        
        offset = result_ptr - result;
        return offset;
}




void LSH::clear()
{
#pragma omp parallel for schedule(static,1) num_threads(LSH_THREADS)
for (int i = 0; i < this->num_layers; ++i){

        memset(this->table_start[i], 0, this->batch_size * this->num_key_value_heads * this->L * this->num_buckets * sizeof(int));
        
        memset(this->table_end[i], 0, this->batch_size * this->num_key_value_heads * this->L * this->num_buckets * sizeof(int));
        memset(this->table[i], 0, this->batch_size * this->num_key_value_heads * this->L * this->max_length * sizeof(int));
    }

    memset(this->mask, 0, this->batch_size * this->num_attention_heads * this->max_length * sizeof(uint8_t));

}

/**
 * @par Implementation note
 * Borrowed non-owning CPU int8 view: mutations/clear are visible; never retain
 * or dereference after owner destruction. Tests use an explicit Tensor shim,
 * not real Torch lifetime protection. Other declared getters lack definitions.
 */
torch::Tensor LSH::get_mask()
{

auto options = torch::TensorOptions().dtype(torch::kInt8);
torch::Tensor tensor = torch::from_blob(this->mask, {this->batch_size, this->num_attention_heads, this->max_length}, options);
return tensor;
}

PYBIND11_MODULE(lsh, m) {
    py::class_<LSH>(m, "LSH")
        .def(py::init<>())
        .def("alloc", &LSH::alloc)
        .def("fill", &LSH::fill)
        .def("clear", &LSH::clear)
        .def("copy", &LSH::copy)
        .def("fastfill", &LSH::fastfill)
        .def("batch_retrieve", &LSH::batch_retrieve)
        .def("get_mask", &LSH::get_mask);
}
