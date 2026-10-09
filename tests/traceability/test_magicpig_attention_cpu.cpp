// TRACE_TEST_ID: MAGICPIG-ATTENTION-CPU
// TRACE_TEST_ID: MAGICPIG-ATTENTION-BOUNDARIES-CPU
// Actual selected QK/correction/Softmax/WV/server bodies; FP64 expectations use
// supplied candidates, not LSH sampling or model-level attention validation.
#include <iomanip>
#include <numeric>
#include <sstream>
namespace magicpig_attention_fixture {
struct Metrics {
    int compositions = 0, probability_checks = 0, output_checks = 0;
    int ownership_checks = 0, boundary_cases = 0, corrected_score_checks = 0;
    double probability_error = 0, output_error = 0, metadata_error = 0, corrected_score_error = 0;
};
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double value, double expected, double bound, double& maximum, const std::string& name) {
    require(std::isfinite(value) && std::isfinite(expected), name + " nonfinite");
    const double error = std::abs(value - expected);
    maximum = std::max(maximum, error);
    if (error > bound) {
        std::ostringstream out;
        out << name << " actual=" << value << " expected=" << expected << " error=" << error << " bound=" << bound;
        throw std::runtime_error(out.str());
    }
}
// Independent arithmetic BF16 decoder: finite normal fixtures only. Does not
// call the dependency model's bit-cast decoder or converter.
double decode(bfloat16 x) {
    const int exponent = (x >> 7) & 255;
    const int mantissa = x & 127;
    require(exponent != 255, "oracle BF16 finite");
    const double significand = exponent ? 1.0 + mantissa / 128.0 : mantissa / 128.0;
    const double magnitude = std::ldexp(significand, exponent ? exponent - 127 : -126);
    return x & 0x8000 ? -magnitude : magnitude;
}
double norm(const float* x, int dim) {
    double sum = 0;
    for (int i = 0; i < dim; ++i) sum += double(x[i]) * x[i];
    return std::sqrt(sum);
}
double norm(const bfloat16* x, int dim) {
    double sum = 0;
    for (int i = 0; i < dim; ++i) sum += decode(x[i]) * decode(x[i]);
    return std::sqrt(sum);
}
struct Reference { std::vector<double> probabilities, output; double maximum, lse; std::vector<double> corrected_logits; };
Reference oracle(const bfloat16* key, const bfloat16* value, const float* query,
                 const std::vector<int>& candidates, int dim, bool corrected) {
    std::vector<double> logits;
    const double qnorm = norm(query, dim);
    for (int row : candidates) {
        double dot = 0;
        for (int d = 0; d < dim; ++d) dot += double(query[d]) * decode(key[row * dim + d]);
        double logit = dot / std::sqrt(double(dim));
        if (corrected) {
            const double cosine = dot / (qnorm * norm(key + row * dim, dim));
            require(cosine > -1 && cosine < 1, "oracle valid cosine");
            const double collision = 1 - std::acos(cosine) / std::acos(-1.0);
            const double p = collision * collision * collision; // K=3
            // Independent enumeration of at-least-two hits in four tables.
            const double probability = 6*p*p*(1-p)*(1-p) + 4*p*p*p*(1-p) + p*p*p*p;
            logit -= std::log(probability + 1e-4);
        }
        logits.push_back(logit);
    }
    const double maximum = *std::max_element(logits.begin(), logits.end());
    const auto corrected_logits = logits;
    double sum = 0;
    for (double& logit : logits) { logit = std::exp(logit - maximum); sum += logit; }
    Reference result{logits, std::vector<double>(dim, 0), maximum / std::log(2.0), (maximum + std::log(sum)) / std::log(2.0), corrected_logits};
    for (std::size_t j = 0; j < candidates.size(); ++j) {
        result.probabilities[j] /= sum;
        for (int d = 0; d < dim; ++d) result.output[d] += result.probabilities[j] * decode(value[candidates[j] * dim + d]);
    }
    return result;
}
#ifdef MAGIC_NATIVE_AVX512
constexpr double probability_bound = 0.002;
constexpr double output_bound = 0.008;
// Native polynomial metadata budget matches the existing selected-body baseline.
constexpr double metadata_bound = 0.03;
#else
constexpr double probability_bound = 0.00003;
constexpr double output_bound = 0.004;
constexpr double metadata_bound = 0.00005;
#endif
void check_result(const float* score, const bfloat16* output, const float* metadata,
                  int head, int heads, const Reference& expected, Metrics& metrics) {
    for (std::size_t j = 0; j < expected.probabilities.size(); ++j) {
        near(score[j], expected.probabilities[j], probability_bound, metrics.probability_error, "probability");
        ++metrics.probability_checks;
    }
    for (std::size_t d = 0; d < expected.output.size(); ++d) {
        near(decode(output[d]), expected.output[d], output_bound, metrics.output_error, "BF16 output");
        ++metrics.output_checks;
    }
    near(metadata[head], expected.maximum, metadata_bound, metrics.metadata_error, "base2 maximum");
    near(metadata[heads + head], expected.lse, metadata_bound, metrics.metadata_error, "base2 LSE");
    ++metrics.compositions;
}
void sparse_server(int dim, int count, bool duplicate, bool dynamic, Metrics& metrics) {
    constexpr int layers = 2, batches = 2, kvheads = 2, groups = 2, capacity = 32;
    constexpr int heads = batches * kvheads * groups;
    SparseAttentionServer server;
    server.alloc(layers, kvheads * groups, kvheads, dim, batches, capacity);
    std::vector<std::vector<bfloat16>> keys(layers, std::vector<bfloat16>(batches * kvheads * capacity * dim));
    auto values = keys;
    std::vector<std::vector<float>> norms(layers, std::vector<float>(batches * kvheads * capacity));
    for (int layer = 0; layer < layers; ++layer) {
        for (int row = 0; row < batches * kvheads * capacity; ++row) {
            for (int d = 0; d < dim; ++d) {
                keys[layer][row * dim + d] = magicpig_attention_model::encode(float(((row * 3 + d * 7 + layer * 5) % 29) - 14) / 32);
                values[layer][row * dim + d] = magicpig_attention_model::encode(float(((row * 11 + d * 3 + layer) % 37) - 18) / 32);
            }
            norms[layer][row] = float(norm(keys[layer].data() + row * dim, dim));
        }
        for (int batch = 0; batch < batches; ++batch) {
            const int start = batch * kvheads * capacity;
            server.fill(layer, batch, torch::Tensor(keys[layer].data() + start * dim, {kvheads, capacity, dim}),
                        torch::Tensor(values[layer].data() + start * dim, {kvheads, capacity, dim}),
                        torch::Tensor(norms[layer].data() + start, {kvheads, capacity}));
        }
    }
    std::vector<float> query(heads * dim), qnorm(heads), metadata(2 * heads + 2, 77);
    std::vector<bfloat16> output(heads * dim + 2, 0x3f80);
    std::vector<int> indices(heads * capacity), counts(heads, count);
    for (int head = 0; head < heads; ++head) {
        for (int d = 0; d < dim; ++d) query[head * dim + d] = float(((head * 5 + d * 11) % 31) - 15) / 32;
        qnorm[head] = float(norm(query.data() + head * dim, dim));
        for (int j = 0; j < capacity; ++j) indices[head * capacity + j] = ((duplicate ? j / 2 : j) * 5 + head) % capacity;
    }
    // Getters are actual selected from_blob call sites; the model supplies
    // non-owning aliases. fill copies bytes; an input mutation cannot affect cache.
    auto cache_view = server.get_key_cache(0);
    auto value_view = server.get_value_cache(1);
    auto norm_view = server.get_key_norm(0);
    require(cache_view.size(0) == batches && cache_view.size(3) == dim, "key view dimensions");
    auto* cached_key = static_cast<bfloat16*>(cache_view.data_ptr());
    const bfloat16 saved = keys[0][0];
    keys[0][0] = 0;
    require(cached_key[0] == saved, "fill owns copied cache storage");
    keys[0][0] = saved;
    auto* cached_value = static_cast<bfloat16*>(value_view.data_ptr());
    auto* cached_norm = static_cast<float*>(norm_view.data_ptr());
    const bfloat16 saved_value = cached_value[0];
    cached_value[0] = 0;
    require(static_cast<bfloat16*>(server.get_value_cache(1).data_ptr())[0] == 0, "value view aliases cache");
    cached_value[0] = saved_value;
    const float saved_norm = cached_norm[0];
    cached_norm[0] = saved_norm + 1;
    require(static_cast<float*>(server.get_key_norm(0).data_ptr())[0] == saved_norm + 1, "norm view aliases cache");
    cached_norm[0] = saved_norm;
    metrics.ownership_checks += 4;
    for (int layer = 0; layer < layers; ++layer) {
        auto* scores = static_cast<float*>(server.get_score().data_ptr());
        std::fill_n(scores, heads * capacity, 9.0f); // Valid, initialized padded score storage.
        const auto out = torch::Tensor(output.data(), {heads, dim});
        const auto meta = torch::Tensor(metadata.data(), {2, heads});
        const auto q = torch::Tensor(query.data(), {heads, dim});
        const auto qn = torch::Tensor(qnorm.data(), {heads});
        const auto ind = torch::Tensor(indices.data(), {heads, capacity});
        const auto nnz = torch::Tensor(counts.data(), {heads});
        if (dynamic) server.dynamic_attention(layer, 3, 4, out, meta, q, qn, ind, nnz);
        else server.attention(layer, 3, 4, out, meta, q, qn, ind, nnz);
        for (int head = 0; head < heads; ++head) {
            std::vector<int> selected(indices.begin() + head * capacity, indices.begin() + head * capacity + count);
            const int kv = head / groups;
            const auto expected = oracle(keys[layer].data() + kv * capacity * dim,
                                         values[layer].data() + kv * capacity * dim,
                                         query.data() + head * dim, selected, dim, true);
            check_result(scores + head * capacity, output.data() + head * dim, metadata.data(), head, heads, expected, metrics);
        }
        require(output[heads * dim] == 0x3f80 && output[heads * dim + 1] == 0x3f80, "output capacity canaries");
        require(metadata[2 * heads] == 77 && metadata[2 * heads + 1] == 77, "metadata capacity canaries");
    }
    server.clear();
    for (int layer = 0; layer < layers; ++layer) {
        auto* key = static_cast<bfloat16*>(server.get_key_cache(layer).data_ptr());
        auto* value = static_cast<bfloat16*>(server.get_value_cache(layer).data_ptr());
        auto* kn = static_cast<float*>(server.get_key_norm(layer).data_ptr());
        require(std::all_of(key, key + batches * kvheads * capacity * dim, [](auto x) { return x == 0; }), "clear key");
        require(std::all_of(value, value + batches * kvheads * capacity * dim, [](auto x) { return x == 0; }), "clear value");
        require(std::all_of(kn, kn + batches * kvheads * capacity, [](auto x) { return x == 0; }), "clear norm");
    }
    require(cached_key[0] == 0 && cached_value[0] == 0 && cached_norm[0] == 0, "existing views observe clear");
    auto* scores = static_cast<float*>(server.get_score().data_ptr());
    require(std::all_of(scores, scores + heads * capacity, [](auto x) { return x == 0; }), "clear score");
    metrics.ownership_checks += 8;
    // Server outlives all accesses to non-owning views; destructor is sanitizer checked.
}
void full_server(int groups, int count, Metrics& metrics) {
    constexpr int dim = 128, capacity = 32, batches = 2, kvheads = 2;
    const int heads = batches * kvheads * groups;
    SparseAttentionServer server;
    server.alloc(1, kvheads * groups, kvheads, dim, batches, capacity);
    std::vector<bfloat16> keys(batches * kvheads * capacity * dim), values(keys.size());
    std::vector<float> norms(batches * kvheads * capacity);
    for (int row = 0; row < batches * kvheads * capacity; ++row) {
        for (int d = 0; d < dim; ++d) {
            keys[row * dim + d] = magicpig_attention_model::encode(float(((row * 3 + d * 5) % 23) - 11) / 32);
            values[row * dim + d] = magicpig_attention_model::encode(float(((row * 7 + d * 3) % 31) - 15) / 32);
        }
        norms[row] = float(norm(keys.data() + row * dim, dim));
    }
    for (int batch = 0; batch < batches; ++batch) {
        const int start = batch * kvheads * capacity;
        server.fill(0, batch, torch::Tensor(keys.data() + start * dim, {kvheads, capacity, dim}),
                    torch::Tensor(values.data() + start * dim, {kvheads, capacity, dim}), torch::Tensor(norms.data() + start, {kvheads, capacity}));
    }
    std::vector<float> query(heads * dim), metadata(2 * heads);
    std::vector<bfloat16> output(heads * dim + 1, 0x3f80);
    std::vector<int> counts(heads, count), selected(count);
    std::iota(selected.begin(), selected.end(), 0);
    for (int head = 0; head < heads; ++head) for (int d = 0; d < dim; ++d)
        query[head * dim + d] = float(((head * 5 + d * 7) % 29) - 14) / 32;
    server.full_attention(0, torch::Tensor(output.data(), {heads, dim}), torch::Tensor(metadata.data(), {2, heads}),
                          torch::Tensor(query.data(), {heads, dim}), torch::Tensor(counts.data(), {heads}));
    auto* scores = static_cast<float*>(server.get_score().data_ptr());
    for (int head = 0; head < heads; ++head) {
        const int kv = head / groups;
        const auto expected = oracle(keys.data() + kv * capacity * dim, values.data() + kv * capacity * dim,
                                     query.data() + head * dim, selected, dim, false);
        check_result(scores + head * capacity, output.data() + head * dim, metadata.data(), head, heads, expected, metrics);
    }
    require(output.back() == 0x3f80, "full output capacity canary");
}
void low_level_boundaries(Metrics& metrics) {
    constexpr int dim = 32, capacity = 32;
    std::vector<bfloat16> key(capacity * dim), value(key.size());
    std::vector<float> query(dim), norms(capacity);
    std::vector<int> ids(capacity);
    for (int d = 0; d < dim; ++d) query[d] = float((d % 7) - 3) / 16;
    for (int row = 0; row < capacity; ++row) {
        ids[row] = (row * 5) % capacity;
        for (int d = 0; d < dim; ++d) {
            key[row * dim + d] = magicpig_attention_model::encode(float(((row * 3 + d) % 19) - 9) / 16);
            value[row * dim + d] = magicpig_attention_model::encode(float(((row + d * 3) % 17) - 8) / 16);
        }
        norms[row] = float(norm(key.data() + row * dim, dim));
    }
    for (int count : {1, 15, 16, 17, 31, 32}) {
        std::vector<float> score(capacity + 2, 8), full_score(capacity + 2, 8);
        qk_kernel(key.data(), ids.data(), query.data(), score.data(), dim, count);
        const int padded = (count + 15) / 16 * 16;
        for (int j = 0; j < count; ++j) {
            double expected = 0;
            for (int d = 0; d < dim; ++d) expected += decode(key[ids[j] * dim + d]) * query[d];
            double unused = 0;
            near(score[j], expected, 1e-6, unused, "QK supplied subset");
        }
        if (padded != count) {
            double dot = 0;
            for (int d = 0; d < dim; ++d) dot += decode(key[ids[count] * dim + d]) * query[d];
            double unused = 0;
            near(score[count], 8 + dot, 1e-6, unused, "QK padded slot preserves stale value then accumulates");
            require(score[padded] == 8 && score[capacity + 1] == 8, "QK allocated capacity canaries");
        }
        qk_kernel_full(key.data(), query.data(), full_score.data(), 1, dim, count, capacity);
        std::vector<int> sequential(count);
        std::iota(sequential.begin(), sequential.end(), 0);
        const auto expected = oracle(key.data(), value.data(), query.data(), sequential, dim, false);
        softmax_kernel_full(full_score.data(), count, std::sqrt(float(dim)), score.data(), score.data() + 1);
        for (int j = 0; j < count; ++j) {
            double unused = 0;
            near(full_score[j], expected.probabilities[j], 3e-6, unused, "full softmax valid probabilities");
        }
        require(full_score[padded] == 8 && full_score[capacity + 1] == 8, "full softmax padding capacity canaries");
        if (padded != count) require(full_score[count] != 8, "full softmax mutates padded slots");
        ++metrics.boundary_cases;
    }
    // All supplied rows still form corrected sparse attention, not full dense:
    // released inclusion correction and +1e-4 remain part of its distribution.
    std::vector<int> all(capacity); std::iota(all.begin(), all.end(), 0);
    const auto sparse = oracle(key.data(), value.data(), query.data(), all, dim, true);
    const auto full = oracle(key.data(), value.data(), query.data(), all, dim, false);
    double difference = 0;
    for (int j = 0; j < capacity; ++j) difference = std::max(difference, std::abs(sparse.probabilities[j] - full.probabilities[j]));
    require(difference > 0.001, "all-candidate corrected sparse distribution differs from dense");
    std::vector<float> corrected_score(capacity), dense_score(capacity), corrected_meta(2), dense_meta(2);
    std::vector<bfloat16> corrected_output(dim), dense_output(dim);
    qk_kernel(key.data(), all.data(), query.data(), corrected_score.data(), dim, capacity);
    transform_kernel(corrected_score.data(), capacity, float(norm(query.data(), dim)), norms.data(), 3, 4, std::sqrt(float(dim)), all.data());
    for (int j = 0; j < capacity; ++j) {
        near(corrected_score[j], sparse.corrected_logits[j], 3e-5, metrics.corrected_score_error, "unnormalized corrected score");
        ++metrics.corrected_score_checks;
    }
    softmax_kernel(corrected_score.data(), capacity, corrected_meta.data(), corrected_meta.data() + 1);
    wv_kernel(value.data(), all.data(), corrected_score.data(), corrected_output.data(), dim, capacity);
    qk_kernel_full(key.data(), query.data(), dense_score.data(), 1, dim, capacity, capacity);
    softmax_kernel_full(dense_score.data(), capacity, std::sqrt(float(dim)), dense_meta.data(), dense_meta.data() + 1);
    wv_kernel(value.data(), all.data(), dense_score.data(), dense_output.data(), dim, capacity);
    check_result(corrected_score.data(), corrected_output.data(), corrected_meta.data(), 0, 1, sparse, metrics);
    check_result(dense_score.data(), dense_output.data(), dense_meta.data(), 0, 1, full, metrics);
    double actual_difference = 0;
    for (int j = 0; j < capacity; ++j) actual_difference = std::max(actual_difference, std::abs(double(corrected_score[j]) - dense_score[j]));
    require(actual_difference > 0.001, "actual all-candidate correction differs from actual dense");
    ++metrics.boundary_cases;
    // QK also silently omits a coordinate when HEAD_DIM is not divisible by 16.
    std::vector<bfloat16> partial_key(32 * 17, 0);
    std::vector<float> partial_query(17, 0), partial_score(32, 0);
    partial_key[16] = 0x3f80; // Key/query dot is exactly one in the omitted coordinate.
    partial_query[16] = 1;
    std::vector<int> partial_ids(32, 0);
    qk_kernel(partial_key.data(), partial_ids.data(), partial_query.data(), partial_score.data(), 17, 16);
    require(partial_score[0] == 0, "QK unsupported dimension omits nonzero tail");
    ++metrics.boundary_cases;
    // WV silently drops HEAD_DIM%16; characterize defined memory accesses only.
    std::vector<bfloat16> partial_output(18, 0x3f80);
    std::vector<float> weight(1, 1);
    std::vector<int> first(1, 0);
    wv_kernel(value.data(), first.data(), weight.data(), partial_output.data(), 17, 1);
    require(partial_output[16] == 0x3f80 && partial_output[17] == 0x3f80, "WV unsupported dimension tail untouched");
    ++metrics.boundary_cases;
    // Full server mixes KV-head nnz[i] and query-head nnz[i]. For lengths
    // [16,32] expanded to four queries per KV head, the second head normalizes
    // 32 ones but WV sums only 16 of them: observed output 0.5, dense oracle 1.
    // This is a defined-memory restriction reproducer, not a correctness pass.
    {
        constexpr int full_dim = 128, full_heads = 8, full_capacity = 32;
        SparseAttentionServer server;
        server.alloc(1, full_heads, 2, full_dim, 1, full_capacity);
        std::vector<bfloat16> zero_keys(2 * full_capacity * full_dim, 0);
        std::vector<bfloat16> one_values(zero_keys.size(), 0x3f80);
        std::vector<float> key_norms(2 * full_capacity, 0), zero_query(full_heads * full_dim, 0);
        server.fill(0, 0, torch::Tensor(zero_keys.data(), {2, full_capacity, full_dim}),
                    torch::Tensor(one_values.data(), {2, full_capacity, full_dim}),
                    torch::Tensor(key_norms.data(), {2, full_capacity}));
        std::vector<int> lengths{16,16,16,16,32,32,32,32};
        std::vector<bfloat16> result(full_heads * full_dim, 0);
        std::vector<float> metadata(2 * full_heads);
        server.full_attention(0, torch::Tensor(result.data(), {full_heads, full_dim}),
                              torch::Tensor(metadata.data(), {2, full_heads}),
                              torch::Tensor(zero_query.data(), {full_heads, full_dim}),
                              torch::Tensor(lengths.data(), {full_heads}));
        double unused = 0;
        near(decode(result[4 * full_dim]), 0.5, 1e-6, unused, "varying GQA length observed half output");
        std::vector<int> selected(full_capacity); std::iota(selected.begin(), selected.end(), 0);
        const auto expected = oracle(zero_keys.data() + full_capacity * full_dim,
                                     one_values.data() + full_capacity * full_dim,
                                     zero_query.data() + 4 * full_dim, selected, full_dim, false);
        require(expected.output[0] == 1.0, "varying GQA length independent dense expectation");
        ++metrics.boundary_cases;
    }
    // Full WV's group=2 path returns without writing; it is unsupported.
    std::vector<bfloat16> unsupported_output(2 * 128, 0x3f80);
    wv_kernel_dim128_full(value.data(), weight.data(), unsupported_output.data(), 128, capacity, 1, 2);
    require(std::all_of(unsupported_output.begin(), unsupported_output.end(), [](auto x) { return x == 0x3f80; }), "full WV unsupported group silent return");
    ++metrics.boundary_cases;
}
int run() {
    Metrics metrics;
    for (int dim : {16, 32, 128}) for (int count : {1, 15, 16, 17, 31, 32}) {
        sparse_server(dim, count, false, false, metrics);
        sparse_server(dim, count, true, true, metrics);
    }
    for (int groups : {1, 4, 8}) for (int count : {16, 32}) full_server(groups, count, metrics);
    low_level_boundaries(metrics);
    std::cout << std::setprecision(12)
              << "MAGICPIG-ATTENTION-CPU: PASS compositions=" << metrics.compositions
              << " probability_checks=" << metrics.probability_checks << " output_checks=" << metrics.output_checks
              << " ownership_checks=" << metrics.ownership_checks << " corrected_score_checks=" << metrics.corrected_score_checks
              << " max_corrected_score_error=" << metrics.corrected_score_error << " max_probability_error=" << metrics.probability_error
              << " max_output_error=" << metrics.output_error << " max_metadata_error=" << metrics.metadata_error << '\n'
              << "MAGICPIG-ATTENTION-BOUNDARIES-CPU: PASS cases=" << metrics.boundary_cases
              << " rounded_capacity=explicit padded_mutation=characterized varying_GQA_lengths=characterized_half_output empty=not_called OpenMP=serial model_conversion=FBGEMM_bits_add_0x8000\n";
    return 0;
}
}
int main() {
    try { return magicpig_attention_fixture::run(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
