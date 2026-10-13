// TRACE_TEST_ID: MAGICPIG-ATTENTION-CPU
// TRACE_TEST_ID: MAGICPIG-ATTENTION-BOUNDARIES-CPU
// Actual selected QK/correction/Softmax/WV/server bodies; FP64 expectations use
// supplied candidates, not LSH sampling or model-level attention validation.
#include <iomanip>
#include <numeric>
#include <sstream>
namespace magicpig_attention_fixture {
#ifdef _OPENMP
constexpr const char* threading_mode = "runtime";
#else
constexpr const char* threading_mode = "serial";
#endif
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
void varying_full_server(int groups, Metrics& metrics) {
    constexpr int dim = 128, capacity = 32, batches = 2, kvheads = 2;
    const int heads = batches * kvheads * groups;
    SparseAttentionServer server;
    server.alloc(2, kvheads * groups, kvheads, dim, batches, capacity);
    std::vector<bfloat16> keys(batches * kvheads * capacity * dim), values(keys.size());
    std::vector<float> norms(batches * kvheads * capacity);
    for (int row = 0; row < batches * kvheads * capacity; ++row) {
        for (int d = 0; d < dim; ++d) {
            keys[row * dim + d] = magicpig_attention_model::encode(float(((row * 3 + d * 5) % 23) - 11) / 32);
            values[row * dim + d] = magicpig_attention_model::encode(float(((row * 7 + d * 3) % 31) - 15) / 32);
        }
        norms[row] = float(norm(keys.data() + row * dim, dim));
    }
    for (int layer = 0; layer < 2; ++layer) {
    for (int batch = 0; batch < batches; ++batch) {
        const int start = batch * kvheads * capacity;
        server.fill(layer, batch, torch::Tensor(keys.data() + start * dim, {kvheads, capacity, dim}),
                    torch::Tensor(values.data() + start * dim, {kvheads, capacity, dim}), torch::Tensor(norms.data() + start, {kvheads, capacity}));
    }
    std::vector<float> query(heads * dim), metadata(2 * heads);
    std::vector<bfloat16> output(heads * dim + 1, 0x3f80);
    std::vector<int> counts(heads);
    // Both KV-head lengths and individual queries within a GQA group differ.
    for (int head = 0; head < heads; ++head) counts[head] = ((head / groups + head % groups + layer) % 2) ? 16 : 32;
    for (int head = 0; head < heads; ++head) for (int d = 0; d < dim; ++d)
        query[head * dim + d] = float(((head * 5 + d * 7) % 29) - 14) / 32;
    server.full_attention(layer, torch::Tensor(output.data(), {heads, dim}), torch::Tensor(metadata.data(), {2, heads}),
                          torch::Tensor(query.data(), {heads, dim}), torch::Tensor(counts.data(), {heads}));
    auto* scores = static_cast<float*>(server.get_score().data_ptr());
    for (int head = 0; head < heads; ++head) {
        const int kv = head / groups;
        std::vector<int> selected(counts[head]);
        std::iota(selected.begin(), selected.end(), 0);
        const auto expected = oracle(keys.data() + kv * capacity * dim, values.data() + kv * capacity * dim,
                                     query.data() + head * dim, selected, dim, false);
        check_result(scores + head * capacity, output.data() + head * dim, metadata.data(), head, heads, expected, metrics);
    }
    require(output.back() == 0x3f80, "varying full output capacity canary");
    // Refill a second layer with different data to detect stale cache routing.
    for (std::size_t i = 0; i < keys.size(); ++i) {
        keys[i] = magicpig_attention_model::encode(float((int(i) % 13) - 6) / 32);
        values[i] = magicpig_attention_model::encode(float((int(i) % 17) - 8) / 32);
    }
    }
}
void empty_server(bool full, bool dynamic, int groups, Metrics& metrics) {
    constexpr int dim = 128, capacity = 32, batches = 2, kvheads = 2;
    const int heads = batches * kvheads * groups;
    SparseAttentionServer server;
    server.alloc(2, kvheads * groups, kvheads, dim, batches, capacity);
    std::vector<bfloat16> keys(batches * kvheads * capacity * dim, 0x3d80), values(keys.size(), 0x3f80);
    for (std::size_t i = 0; i < keys.size(); ++i) keys[i] = magicpig_attention_model::encode(float((int(i) % 11) - 5) / 32);
    std::vector<float> norms(batches * kvheads * capacity);
    for (int row = 0; row < batches * kvheads * capacity; ++row) norms[row] = float(norm(keys.data() + row * dim, dim));
    std::vector<float> query(heads * dim, 0.125f), qnorm(heads, float(norm(query.data(), dim))), metadata(2 * heads + 2, 77);
    std::vector<int> counts(heads, 16), ids(heads * capacity);
    for (int head = 0; head < heads; ++head) {
        if (head % 3 != 1) { counts[head] = 0; qnorm[head] = 0; }
        for (int j = 0; j < capacity; ++j) ids[head * capacity + j] = j;
    }
    std::vector<bfloat16> output(heads * dim + 2, 0x3f80);
    for (int layer = 0; layer < 2; ++layer) {
        for (int batch = 0; batch < batches; ++batch) {
            const int start = batch * kvheads * capacity;
            server.fill(layer, batch, torch::Tensor(keys.data() + start * dim, {kvheads, capacity, dim}),
                        torch::Tensor(values.data() + start * dim, {kvheads, capacity, dim}),
                        torch::Tensor(norms.data() + start, {kvheads, capacity}));
        }
        auto* scores = static_cast<float*>(server.get_score().data_ptr());
        std::fill_n(scores, heads * capacity, 91);
        if (full) server.full_attention(layer, torch::Tensor(output.data(), {heads, dim}), torch::Tensor(metadata.data(), {2, heads}),
                                       torch::Tensor(query.data(), {heads, dim}), torch::Tensor(counts.data(), {heads}));
        else if (dynamic) server.dynamic_attention(layer, 3, 4, torch::Tensor(output.data(), {heads, dim}), torch::Tensor(metadata.data(), {2, heads}),
                                       torch::Tensor(query.data(), {heads, dim}), torch::Tensor(qnorm.data(), {heads}),
                                       torch::Tensor(ids.data(), {heads, capacity}), torch::Tensor(counts.data(), {heads}));
        else server.attention(layer, 3, 4, torch::Tensor(output.data(), {heads, dim}), torch::Tensor(metadata.data(), {2, heads}),
                                       torch::Tensor(query.data(), {heads, dim}), torch::Tensor(qnorm.data(), {heads}),
                                       torch::Tensor(ids.data(), {heads, capacity}), torch::Tensor(counts.data(), {heads}));
        for (int head = 0; head < heads; ++head) {
            if (counts[head] == 0) {
                require(std::all_of(output.begin() + head * dim, output.begin() + (head + 1) * dim, [](auto x){return x == 0;}), "empty server output zero");
                require(metadata[head] == -INFINITY && metadata[heads + head] == -INFINITY, "empty server max/LSE negative infinity");
                require(std::all_of(scores + head * capacity, scores + (head + 1) * capacity, [](auto x){return x == 91;}), "empty server leaves scores untouched");
                ++metrics.boundary_cases;
            } else {
                std::vector<int> selected(16); std::iota(selected.begin(), selected.end(), 0);
                const auto expected = oracle(keys.data() + head / groups * capacity * dim, values.data() + head / groups * capacity * dim,
                                             query.data() + head * dim, selected, dim, !full);
                check_result(scores + head * capacity, output.data() + head * dim, metadata.data(), head, heads, expected, metrics);
            }
        }
        require(output[heads * dim] == 0x3f80 && output[heads * dim + 1] == 0x3f80 && metadata[2 * heads] == 77 && metadata[2 * heads + 1] == 77, "empty mixed-head canaries");
    }
}
void caller_atomic_rejection(bool dynamic, Metrics& metrics) {
    constexpr int dim = 32, heads = 4, kvheads = 2, capacity = 32;
    SparseAttentionServer server;
    server.alloc(1, heads, kvheads, dim, 1, capacity);
    std::vector<bfloat16> keys(kvheads * capacity * dim), values(keys.size(), 0x3f80);
    for (std::size_t i = 0; i < keys.size(); ++i) keys[i] = magicpig_attention_model::encode(float((int(i) % 11) - 5) / 32);
    std::vector<float> norms(kvheads * capacity);
    for (int row = 0; row < kvheads * capacity; ++row) norms[row] = float(norm(keys.data() + row * dim, dim));
    server.fill(0, 0, torch::Tensor(keys.data(), {kvheads, capacity, dim}),
                torch::Tensor(values.data(), {kvheads, capacity, dim}), torch::Tensor(norms.data(), {kvheads, capacity}));
    auto* cached_norms = static_cast<float*>(server.get_key_norm(0).data_ptr());
    auto* scores = static_cast<float*>(server.get_score().data_ptr());
    std::vector<float> query(heads * dim, .125f), qnorm(heads, float(norm(query.data(), dim))), metadata(2 * heads + 2, 77);
    std::vector<bfloat16> output(heads * dim + 2, 0x3f80);
    std::vector<int> ids(heads * capacity, 0), counts(heads, 1);
    const float valid_qnorm = qnorm.back(), valid_knorm = cached_norms[capacity];
    for (int fault = 0; fault < 13; ++fault) {
        qnorm.back() = valid_qnorm; cached_norms[capacity] = valid_knorm;
        query.back() = .125f; ids[(heads - 1) * capacity] = 0; counts.back() = 1;
        int K = 3, L = 4;
        switch (fault) {
            case 0: qnorm.back() = 0; break;
            case 1: qnorm.back() = -1; break;
            case 2: qnorm.back() = NAN; break;
            case 3: qnorm.back() = INFINITY; break;
            case 4: cached_norms[capacity] = 0; break;
            case 5: cached_norms[capacity] = NAN; break;
            case 6: qnorm.back() = 1e-20f; break; // Inconsistent cosine outside domain.
            case 7: query.back() = NAN; break;
            case 8: ids[(heads - 1) * capacity] = -1; break;
            case 9: ids[(heads - 1) * capacity] = capacity; break;
            case 10: counts.back() = capacity + 1; break;
            case 11: K = 0; break;
            case 12: L = 1; break;
        }
        std::fill_n(scores, heads * capacity, 91);
        bool rejected = false;
        try {
            if (dynamic) server.dynamic_attention(0, K, L, torch::Tensor(output.data(), {heads, dim}), torch::Tensor(metadata.data(), {2, heads}),
                   torch::Tensor(query.data(), {heads, dim}), torch::Tensor(qnorm.data(), {heads}),
                   torch::Tensor(ids.data(), {heads, capacity}), torch::Tensor(counts.data(), {heads}));
            else server.attention(0, K, L, torch::Tensor(output.data(), {heads, dim}), torch::Tensor(metadata.data(), {2, heads}),
                   torch::Tensor(query.data(), {heads, dim}), torch::Tensor(qnorm.data(), {heads}),
                   torch::Tensor(ids.data(), {heads, capacity}), torch::Tensor(counts.data(), {heads}));
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && std::all_of(scores, scores + heads * capacity, [](auto x){return x == 91;}) &&
                std::all_of(output.begin(), output.end(), [](auto x){return x == 0x3f80;}) &&
                std::all_of(metadata.begin(), metadata.end(), [](auto x){return x == 77;}), "later-head invalid domain preserves all scores/output/metadata");
        ++metrics.boundary_cases;
    }
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
        require(std::all_of(score.begin() + count, score.end(), [](auto x){return x == 8;}),
                "QK logical nnz leaves padding and canaries untouched");
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
    for (int count : {0, 1, 15, 16, 17, 31, 32}) {
        std::vector<int> exact_ids(ids.begin(), ids.begin() + count);
        std::vector<float> exact_scores(count, 83);
        qk_kernel(key.data(), exact_ids.data(), query.data(), exact_scores.data(), dim, count);
        std::vector<float> guarded(count + 2, 83);
        qk_kernel(key.data(), exact_ids.data(), query.data(), guarded.data() + 1, dim, count);
        require(guarded.front() == 83 && guarded.back() == 83, "QK exact logical score canaries");
        for (int j = 0; j < count; ++j) {
            double expected = 0, unused = 0;
            for (int d = 0; d < dim; ++d) expected += decode(key[exact_ids[j] * dim + d]) * query[d];
            near(exact_scores[j], expected, 1e-6, unused, "QK exact allocation FP64 oracle");
            near(guarded[j + 1], expected, 1e-6, unused, "QK guarded exact logical oracle");
        }
        ++metrics.boundary_cases;
    }
    qk_kernel(nullptr, nullptr, nullptr, nullptr, dim, 0);
    std::vector<float> invalid_score(2, 83);
    bool negative_rejected = false;
    try { qk_kernel(nullptr, nullptr, nullptr, invalid_score.data(), dim, -1); }
    catch (const std::invalid_argument&) { negative_rejected = true; }
    require(negative_rejected && invalid_score[0] == 83 && invalid_score[1] == 83, "negative QK nnz rejects before writes");
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
    // Unsupported dimensions reject before any score/output writes or input reads.
    for (int bad_dim : {0, -16, 1, 15, 17, 31, 33, 127, 129}) {
        std::vector<float> score(2, 71);
        std::vector<bfloat16> output(2, 0x3f80);
        bool qk_rejected = false, wv_rejected = false, full_rejected = false;
        try { qk_kernel(nullptr, nullptr, nullptr, score.data(), bad_dim, 1); }
        catch (const std::invalid_argument&) { qk_rejected = true; }
        try { wv_kernel(nullptr, nullptr, nullptr, output.data(), bad_dim, 1); }
        catch (const std::invalid_argument&) { wv_rejected = true; }
        try { qk_kernel_full(nullptr, nullptr, score.data(), 1, bad_dim, 1, 1); }
        catch (const std::invalid_argument&) { full_rejected = true; }
        require(qk_rejected && wv_rejected && full_rejected, "unsupported dimension rejected");
        require(score[0] == 71 && score[1] == 71 && output[0] == 0x3f80 && output[1] == 0x3f80,
                "dimension rejection precedes writes");
        ++metrics.boundary_cases;
    }
    // Regression #77: KV lengths [16,32] expanded over four query heads.
    // All-one V with zero logits must produce one for every query head.
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
        near(decode(result[4 * full_dim]), 1.0, 1e-6, unused, "varying GQA length regression equals dense output");
        std::vector<int> selected(full_capacity); std::iota(selected.begin(), selected.end(), 0);
        const auto expected = oracle(zero_keys.data() + full_capacity * full_dim,
                                     one_values.data() + full_capacity * full_dim,
                                     zero_query.data() + 4 * full_dim, selected, full_dim, false);
        require(expected.output[0] == 1.0, "varying GQA length independent dense expectation");
        ++metrics.boundary_cases;
    }
    {
        constexpr int heads = 8, capacity = 32, dim = 128;
        SparseAttentionServer server;
        server.alloc(1, heads, 2, dim, 1, capacity);
        std::vector<float> query(heads * dim, 0), metadata(2 * heads, 77);
        std::vector<bfloat16> result(heads * dim, 0x3f80);
        std::vector<int> counts(heads, 16);
        auto* scores = static_cast<float*>(server.get_score().data_ptr());
        std::fill_n(scores, heads * capacity, 91);
        for (int invalid : {-1, capacity + 1}) {
            counts.back() = invalid;
            bool rejected = false;
            try { server.full_attention(0, torch::Tensor(result.data(), {heads, dim}),
                      torch::Tensor(metadata.data(), {2, heads}), torch::Tensor(query.data(), {heads, dim}),
                      torch::Tensor(counts.data(), {heads})); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected && std::all_of(result.begin(), result.end(), [](auto x){return x == 0x3f80;}) &&
                    std::all_of(metadata.begin(), metadata.end(), [](auto x){return x == 77;}) &&
                    std::all_of(scores, scores + heads * capacity, [](auto x){return x == 91;}), "invalid late-head nnz rejects before every write");
            ++metrics.boundary_cases;
        }
        counts.back() = 16;
        bool rejected = false;
        try { server.full_attention(0, torch::Tensor(result.data(), {heads, dim}),
                  torch::Tensor(metadata.data(), {2, heads}), torch::Tensor(query.data(), {heads, dim}),
                  torch::Tensor(counts.data(), {heads - 1})); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && scores[0] == 91 && result[0] == 0x3f80 && metadata[0] == 77, "KV-sized nnz vector rejected before writes");
        ++metrics.boundary_cases;
        rejected = false;
        try { server.full_attention(0, torch::Tensor(result.data(), {heads, dim}),
                  torch::Tensor(metadata.data(), {2, heads}), torch::Tensor(query.data(), {heads, dim}),
                  torch::Tensor(counts.data(), {heads, 1})); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && scores[0] == 91 && result[0] == 0x3f80 && metadata[0] == 77, "nnz rank contract rejects before writes");
        ++metrics.boundary_cases;
    }
    // Fixed full WV and full server reject unsupported dimension/group policy.
    for (int dim : {16, 32, 127, 128, 129}) for (int group : {0, 1, 2, 4, 8}) {
        if (dim == 128 && (group == 1 || group == 4 || group == 8)) continue;
        std::vector<bfloat16> result(2, 0x3f80);
        bool rejected = false;
        try { wv_kernel_dim128_full(nullptr, nullptr, result.data(), dim, 32, 1, group); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && result[0] == 0x3f80 && result[1] == 0x3f80, "full WV rejects before input/output access");
        ++metrics.boundary_cases;
    }
    for (int dim : {16, 32, 128}) for (int group : {1, 2, 4, 8}) {
        if (dim == 128 && group != 2) continue;
        SparseAttentionServer server;
        server.alloc(1, group, 1, dim, 1, 1);
        std::vector<float> query(group * dim, 0), metadata(2 * group, 77);
        std::vector<bfloat16> result(group * dim, 0x3f80);
        std::vector<int> counts(group, 1);
        auto* scores = static_cast<float*>(server.get_score().data_ptr());
        std::fill_n(scores, group, 91);
        bool rejected = false;
        try { server.full_attention(0, torch::Tensor(result.data(), {group, dim}),
                  torch::Tensor(metadata.data(), {2, group}), torch::Tensor(query.data(), {group, dim}),
                  torch::Tensor(counts.data(), {group})); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && std::all_of(result.begin(), result.end(), [](auto x){return x == 0x3f80;}) &&
                std::all_of(metadata.begin(), metadata.end(), [](auto x){return x == 77;}) &&
                std::all_of(scores, scores + group, [](auto x){return x == 91;}), "full server rejects before writes");
        ++metrics.boundary_cases;
    }

}
int run() {
    Metrics metrics;
    for (int dim : {16, 32, 128}) for (int count : {1, 15, 16, 17, 31, 32}) {
        sparse_server(dim, count, false, false, metrics);
        sparse_server(dim, count, true, true, metrics);
    }
    for (int groups : {1, 4, 8}) for (int count : {16, 32}) full_server(groups, count, metrics);
    for (int groups : {1, 4, 8}) varying_full_server(groups, metrics);
    empty_server(false, false, 2, metrics);
    empty_server(false, true, 2, metrics);
    for (int groups : {1, 4, 8}) empty_server(true, false, groups, metrics);
    caller_atomic_rejection(false, metrics);
    caller_atomic_rejection(true, metrics);
    low_level_boundaries(metrics);
    std::cout << std::setprecision(12)
              << "MAGICPIG-ATTENTION-CPU: PASS compositions=" << metrics.compositions
              << " probability_checks=" << metrics.probability_checks << " output_checks=" << metrics.output_checks
              << " ownership_checks=" << metrics.ownership_checks << " corrected_score_checks=" << metrics.corrected_score_checks
              << " max_corrected_score_error=" << metrics.corrected_score_error << " max_probability_error=" << metrics.probability_error
              << " max_output_error=" << metrics.output_error << " max_metadata_error=" << metrics.metadata_error << '\n'
              << "MAGICPIG-ATTENTION-BOUNDARIES-CPU: PASS cases=" << metrics.boundary_cases
              << " qk_exact_nnz=verified softmax_padded_mutation=characterized unsupported_dimensions_groups=rejected varying_GQA_lengths=corrected_query_head_contract empty_server=zero_output_negative_infinity_metadata OpenMP=" << threading_mode << " model_conversion=FBGEMM_bits_add_0x8000\n";
    return 0;
}
}
int main() {
    try { return magicpig_attention_fixture::run(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
