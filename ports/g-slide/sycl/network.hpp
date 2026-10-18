#pragma once
#include "../include/TrainingControl.h"
#include "kernels.hpp"
#include <memory>
#include <vector>

namespace gslide::sycl_port {
template <class T> class usm_array {
  sycl::queue *queue;

public:
  T *p;
  int size;
  static T *allocate(sycl::queue &q, int n) {
    require(n >= 0, "allocation extent");
    return sycl::malloc_shared<T>(std::max(n, 1), q);
  }
  usm_array(sycl::queue &q, int n) : queue(&q), p(allocate(q, n)), size(n) {
    if (!p)
      throw std::bad_alloc();
    q.fill(p, T{}, std::max(n, 1)).wait_and_throw();
  }
  ~usm_array() { sycl::free(p, *queue); }
  usm_array(const usm_array &) = delete;
  usm_array &operator=(const usm_array &) = delete;
  T &operator[](int i) { return p[i]; }
  const T &operator[](int i) const { return p[i]; }
  void assign(const std::vector<T> &v) {
    require(v.size() <= unsigned(size), "transfer capacity");
    if (!v.empty())
      queue->memcpy(p, v.data(), v.size() * sizeof(T)).wait_and_throw();
  }
};
struct parameter {
  usm_array<float> value, gradient, moment, velocity;
  parameter(sycl::queue &q, const std::vector<float> &initial)
      : value(q, initial.size()), gradient(q, initial.size()),
        moment(q, initial.size()), velocity(q, initial.size()) {
    value.assign(initial);
  }
};
struct sparse_batch {
  std::vector<int> nodes, offsets, labels, label_offsets;
  std::vector<float> values;
  int size() const { return offsets.size() - 1; }
};
enum class delta_reset { each_step, diagnostic_keep_stale };
// Reference device implementations of the backend library boundary. These are
// intentionally simple deterministic algorithms, not vendor-library speedups.
inline void scan_lengths(sycl::queue &q, const int *lengths, int *offsets,
                         int n) {
  require(n >= 0, "scan extent");
  q.parallel_for(sycl::range<1>(n + 1), [=](sycl::id<1> id) {
     int i = id[0], s = 0;
     for (int j = 0; j < i; ++j)
       s += lengths[j];
     offsets[i] = s;
   }).wait_and_throw();
}
inline void sort_keys(sycl::queue &q, const unsigned *keys, const int *values,
                      unsigned *out_keys, int *out_values, int n) {
  require(n >= 0, "sort extent");
  if (!n)
    return;
  q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
     int i = id[0], rank = 0;
     for (int j = 0; j < n; ++j)
       rank += keys[j] < keys[i] || (keys[j] == keys[i] && j < i);
     out_keys[rank] = keys[i];
     out_values[rank] = values[i];
   }).wait_and_throw();
}
// x [batch,input] row major. w may be row major [output,input] or column
// major [input,output]. y is row major [batch,output], no transpose ambiguity.
inline void dense_product(sycl::queue &q, const float *x, const float *w,
                          const float *bias, float *y, int batch, int inputs,
                          int outputs, bool weights_rowmajor, bool relu) {
  require(batch >= 0 && inputs > 0 && outputs > 0, "dense shape");
  if (!batch)
    return;
  q.parallel_for(sycl::range<1>(batch * outputs), [=](sycl::id<1> id) {
     int b = id[0] / outputs, o = id[0] % outputs;
     float v = bias[o];
     for (int i = 0; i < inputs; ++i)
       v += x[b * inputs + i] *
            w[weights_rowmajor ? o * inputs + i : i * outputs + o];
     y[id[0]] = relu ? sycl::fmax(0.f, v) : v;
   }).wait_and_throw();
}
// TRACE_TEST_ID: GSLIDE-SYCL-PRODUCTION-116
// Support: shared production ordering with a two-layer SYCL numerical backend.
class Network {
  static int validate_configuration(
      int i, int h, int o, int nb, int ic, int lc, int tables, int k, int bin,
      int buckets, int threshold, int minimum, const std::vector<float> &hw,
      const std::vector<float> &hb, const std::vector<float> &ow,
      const std::vector<float> &ob, const std::vector<int> &bins) {
    require(i > 0 && h > 0 && o > 0 && nb > 0 && ic >= 0 && lc >= 0 &&
                tables > 0 && k > 0 && bin > 0 && buckets > 0 &&
                threshold > 0 && minimum >= 0 && minimum <= o,
            "network configuration");
    const auto limit = std::int64_t(std::numeric_limits<int>::max());
    auto check_extent = [&](std::initializer_list<int> dimensions) {
      std::int64_t extent = 1;
      for (int dimension : dimensions) {
        require(dimension > 0 && extent <= (limit - 1) / dimension,
                "network extent overflow");
        extent *= dimension;
      }
    };
    check_extent({nb, i});
    check_extent({nb, h});
    check_extent({nb, o});
    check_extent({i, h});
    check_extent({o, h});
    check_extent({tables, k, bin});
    check_extent({nb, tables, o});
    check_extent({tables, buckets, o});
    require(hw.size() == unsigned(i * h) && hb.size() == unsigned(h) &&
                ow.size() == unsigned(o * h) && ob.size() == unsigned(o),
            "parameter shape");
    for (const auto *vector : {&hw, &hb, &ow, &ob})
      for (float value : *vector)
        require(std::isfinite(value), "finite parameter");
    require(bins.size() == unsigned(tables * k * bin), "bins extent");
    for (int id : bins)
      require(id >= 0 && id < h, "bin ID");
    return i;
  }
  sycl::queue &q;
  int inputs, hidden, outputs, max_batch, input_capacity, label_capacity;
  hash_shape shape;
  usm_array<int> input_nodes, input_offsets, label_nodes, label_offsets,
      hidden_ids, hidden_offsets;
  usm_array<float> input_values, dense_inputs;
  usm_array<int> bins, random_nodes, query_addresses, lengths, gather_offsets,
      gathered, counts, sizes;
  usm_array<unsigned> random_keys_buffer, sorted_keys;
  usm_array<int> sort_values, sorted_values;
  int current_batch = 0;
  void validate_batch(const sparse_batch &b, bool training_labels = true) {
    require(b.offsets.size() >= 1 &&
                b.label_offsets.size() == b.offsets.size() &&
                b.size() <= max_batch,
            "batch extent");
    require(b.nodes.size() == b.values.size() &&
                b.nodes.size() <= unsigned(input_capacity) &&
                b.labels.size() <= unsigned(label_capacity),
            "batch capacity");
    require(b.offsets.front() == 0 && b.label_offsets.front() == 0 &&
                b.offsets.back() == int(b.nodes.size()) &&
                b.label_offsets.back() == int(b.labels.size()),
            "compressed endpoints");
    for (int i = 0; i < b.size(); ++i) {
      require(b.offsets[i] <= b.offsets[i + 1] &&
                  b.label_offsets[i] <= b.label_offsets[i + 1],
              "monotone offsets");
      if (training_labels)
        require(b.label_offsets[i] < b.label_offsets[i + 1],
                "training requires labels");
    }
    for (int n : b.nodes)
      require(n >= 0 && n < inputs, "input ID");
    for (int n : b.labels)
      require(n >= 0 && n < outputs, "label ID");
    for (float x : b.values)
      require(std::isfinite(x), "finite input");
  }
  void transfer(const sparse_batch &b, bool training_labels = true) {
    validate_batch(b, training_labels);
    current_batch = b.size();
    input_nodes.assign(b.nodes);
    input_values.assign(b.values);
    input_offsets.assign(b.offsets);
    label_nodes.assign(b.labels);
    label_offsets.assign(b.label_offsets);
    q.fill(dense_inputs.p, 0.f, max_batch * inputs).wait_and_throw();
    auto ids = input_nodes.p;
    auto val = input_values.p;
    auto off = input_offsets.p;
    auto dst = dense_inputs.p;
    int ni = inputs;
    if (current_batch)
      q.parallel_for(sycl::range<1>(current_batch), [=](sycl::id<1> id) {
         int s = id[0];
         for (int j = off[s]; j < off[s + 1]; ++j)
           dst[s * ni + ids[j]] += val[j];
       }).wait_and_throw();
  }
  void query() {
    int nb = current_batch, no = outputs, nh = hidden;
    auto z = shape;
    auto bp = bins.p;
    auto hp = hidden_values.p;
    auto sp = bucket_sizes.p;
    auto query = query_addresses.p;
    auto len = lengths.p;
    if (nb)
      q.parallel_for(sycl::range<1>(nb * z.tables), [=](sycl::id<1> id) {
         int s = id[0] / z.tables, t = id[0] % z.tables;
         int a = address(bp + t * z.k * z.bin, hp + s * nh, z.k, z.bin,
                         int(sycl::log(float(z.bin))));
         query[id[0]] = a;
         len[id[0]] = sycl::min(sp[t * z.buckets + a], z.capacity);
       }).wait_and_throw();
    scan_lengths(q, lengths.p, gather_offsets.p, nb * z.tables);
    auto go = gather_offsets.p;
    auto raw = gathered.p;
    auto buckets = bucket_nodes.p;
    if (nb)
      q.parallel_for(sycl::range<1>(nb * z.tables), [=](sycl::id<1> id) {
         int t = id[0] % z.tables, a = t * z.buckets + query[id[0]];
         for (int p = go[id[0]]; p < go[id + 1]; ++p)
           raw[p] = buckets[a * z.capacity + p - go[id[0]]];
       }).wait_and_throw();
    q.fill(counts.p, 0, max_batch * outputs).wait_and_throw();
    count_candidates(q, gathered.p, gather_offsets.p, z.tables, nb, no,
                     max_batch * outputs, counts.p);
    activate_labels(q, label_nodes.p, label_offsets.p, random_nodes.p, nb, no,
                    threshold, minimum, seed, counts.p, sizes.p);
    scan_lengths(q, sizes.p, active_offsets.p, nb);
    auto freq = counts.p;
    auto offsets = active_offsets.p;
    auto ids = active_nodes.p;
    int th = threshold;
    if (nb)
      q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
         int b = id[0], pos = offsets[b];
         for (int n = 0; n < no; ++n)
           if (freq[b * no + n] >= th)
             ids[pos++] = n;
       }).wait_and_throw();
  }
  void output_forward() {
    query();
    int nb = current_batch, nh = hidden, no = outputs;
    auto off = active_offsets.p;
    auto ids = active_nodes.p;
    auto hv = hidden_values.p;
    auto w = output_weights.value.p;
    auto bias = output_bias.value.p;
    auto prob = probabilities.p;
    auto delta = output_deltas.p;
    auto labs = label_nodes.p;
    auto loff = label_offsets.p;
    if (nb)
      q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> id) {
         int s = id[0];
         float maximum = -std::numeric_limits<float>::max();
         for (int p = off[s]; p < off[s + 1]; ++p) {
           int n = ids[p];
           float v = bias[n];
           for (int h = 0; h < nh; ++h)
             v += hv[s * nh + h] * w[n * nh + h];
           prob[p] = v;
           maximum = sycl::fmax(maximum, v);
         }
         float sum = 0;
         for (int p = off[s]; p < off[s + 1]; ++p) {
           prob[p] = sycl::exp(prob[p] - maximum);
           sum += prob[p];
         }
         for (int p = off[s]; p < off[s + 1]; ++p) {
           prob[p] /= sum;
           bool label = false;
           for (int j = loff[s]; j < loff[s + 1]; ++j)
             label |= labs[j] == ids[p];
           delta[p] =
               ((label ? 1.f / (loff[s + 1] - loff[s]) : 0.f) - prob[p]) / nb;
         }
       }).wait_and_throw();
  }
  void output_backward() {
    int nb = current_batch, nh = hidden, no = outputs;
    auto off = active_offsets.p;
    auto ids = active_nodes.p;
    auto hv = hidden_values.p;
    auto d = output_deltas.p;
    auto w = output_weights.value.p;
    auto hd = hidden_deltas.p;
    auto gw = output_weights.gradient.p;
    auto gb = output_bias.gradient.p;
    if (nb)
      q.parallel_for(sycl::range<1>(nb * no), [=](sycl::id<1> id) {
         int b = id[0] / no, p = off[b] + id[0] % no;
         if (p >= off[b + 1])
           return;
         for (int h = 0; h < nh; ++h)
           if (hv[b * nh + h] > 0) {
             sycl::atomic_ref<float, sycl::memory_order::relaxed,
                              sycl::memory_scope::device,
                              sycl::access::address_space::global_space>
                 a(hd[b * nh + h]);
             a.fetch_add(d[p] * w[ids[p] * nh + h]);
           }
       }).wait_and_throw();
    q.parallel_for(sycl::range<1>(no * nh), [=](sycl::id<1> id) {
       int o = id[0] / nh, h = id[0] % nh;
       float g = 0;
       for (int b = 0; b < nb; ++b)
         for (int p = off[b]; p < off[b + 1]; ++p)
           if (ids[p] == o)
             g += d[p] * hv[b * nh + h];
       gw[id[0]] += g;
     }).wait_and_throw();
    q.parallel_for(sycl::range<1>(no), [=](sycl::id<1> id) {
       float g = 0;
       for (int b = 0; b < nb; ++b)
         for (int p = off[b]; p < off[b + 1]; ++p)
           if (ids[p] == int(id[0]))
             g += d[p];
       gb[id[0]] += g;
     }).wait_and_throw();
  }
  void hidden_backward() {
    int nb = current_batch, ni = inputs, nh = hidden;
    auto x = dense_inputs.p;
    auto d = hidden_deltas.p;
    auto gw = hidden_weights.gradient.p;
    auto gb = hidden_bias.gradient.p;
    q.parallel_for(sycl::range<1>(ni * nh), [=](sycl::id<1> id) {
       int i = id[0] / nh, h = id[0] % nh;
       float g = 0;
       for (int b = 0; b < nb; ++b)
         g += x[b * ni + i] * d[b * nh + h];
       gw[id[0]] += g;
     }).wait_and_throw();
    q.parallel_for(sycl::range<1>(nh), [=](sycl::id<1> id) {
       float g = 0;
       for (int b = 0; b < nb; ++b)
         g += d[b * nh + id[0]];
       gb[id[0]] += g;
     }).wait_and_throw();
  }
  void update(parameter &p, float rate) {
    auto w = p.value.p, g = p.gradient.p, m = p.moment.p, v = p.velocity.p;
    q.parallel_for(sycl::range<1>(p.value.size), [=](sycl::id<1> id) {
       int i = id[0];
       float grad = g[i];
       m[i] = .9f * m[i] + .1f * grad;
       v[i] = .999f * v[i] + .001f * grad * grad;
       w[i] += rate * m[i] / (sycl::sqrt(v[i]) + 1e-8f);
       g[i] = 0;
     }).wait_and_throw();
  }

public:
  parameter hidden_weights, hidden_bias, output_weights, output_bias;
  usm_array<float> hidden_values, hidden_deltas, probabilities, output_deltas;
  usm_array<int> active_nodes, active_offsets, bucket_nodes, bucket_sizes;
  int rebuild_count = 0, step_count = 0, threshold, minimum;
  unsigned seed;
  Network(sycl::queue &q, int i, int h, int o, int maxbatch, int incap,
          int labcap, const std::vector<float> &hw,
          const std::vector<float> &hb, const std::vector<float> &ow,
          const std::vector<float> &ob,
          const std::vector<int> &permutation_bins, int tables = 3, int k = 2,
          int bin = 4, int buckets = 16, int threshold = 2, int minimum = 1,
          unsigned seed = 40)
      : q(q),
        inputs(validate_configuration(i, h, o, maxbatch, incap, labcap, tables,
                                      k, bin, buckets, threshold, minimum, hw,
                                      hb, ow, ob, permutation_bins)),
        hidden(h), outputs(o), max_batch(maxbatch), input_capacity(incap),
        label_capacity(labcap), shape{h, o, tables, k, bin, 2, 2, buckets, o},
        input_nodes(q, incap), input_offsets(q, maxbatch + 1),
        label_nodes(q, labcap), label_offsets(q, maxbatch + 1),
        hidden_ids(q, maxbatch * h), hidden_offsets(q, maxbatch + 1),
        input_values(q, incap), dense_inputs(q, maxbatch * i),
        bins(q, tables * k * bin), random_nodes(q, o),
        query_addresses(q, maxbatch * tables), lengths(q, maxbatch * tables),
        gather_offsets(q, maxbatch * tables + 1),
        gathered(q, maxbatch * tables * o), counts(q, maxbatch * o),
        sizes(q, maxbatch),
        random_keys_buffer(q, std::max(o, tables * k * bin)),
        sorted_keys(q, std::max(o, tables * k * bin)),
        sort_values(q, std::max(o, tables * k * bin)),
        sorted_values(q, std::max(o, tables * k * bin)), hidden_weights(q, hw),
        hidden_bias(q, hb), output_weights(q, ow), output_bias(q, ob),
        hidden_values(q, maxbatch * h), hidden_deltas(q, maxbatch * h),
        probabilities(q, maxbatch * o), output_deltas(q, maxbatch * o),
        active_nodes(q, maxbatch * o), active_offsets(q, maxbatch + 1),
        bucket_nodes(q, tables * buckets * o),
        bucket_sizes(q, tables * buckets), threshold(threshold),
        minimum(minimum), seed(seed) {
    require(i > 0 && h > 0 && o > 0 && maxbatch > 0 && incap >= 0 &&
                labcap >= 0,
            "network shape");
    require(hw.size() == unsigned(i * h) && hb.size() == unsigned(h) &&
                ow.size() == unsigned(o * h) && ob.size() == unsigned(o),
            "parameter shape");
    require(permutation_bins.size() == unsigned(tables * k * bin),
            "bins extent");
    for (int id : permutation_bins)
      require(id >= 0 && id < h, "bin ID");
    bins.assign(permutation_bins);
    for (int n = 0; n < o; ++n)
      random_nodes[n] = n;
    rebuild(false);
  }
  void rebuild(bool reshuffle) {
    if (reshuffle) {
      int n = bins.size;
      require(n % hidden == 0, "shuffle bins requires complete permutations");
      random_keys(q, random_keys_buffer.p, n, seed, hidden, true);
      for (int i = 0; i < n; ++i)
        sort_values[i] = bins[i];
      sort_keys(q, random_keys_buffer.p, sort_values.p, sorted_keys.p,
                sorted_values.p, n);
      q.memcpy(bins.p, sorted_values.p, n * sizeof(int)).wait_and_throw();
      random_keys(q, random_keys_buffer.p, outputs, seed, outputs, false);
      for (int i = 0; i < outputs; ++i)
        sort_values[i] = random_nodes[i];
      sort_keys(q, random_keys_buffer.p, sort_values.p, sorted_keys.p,
                sorted_values.p, outputs);
      q.memcpy(random_nodes.p, sorted_values.p, outputs * sizeof(int))
          .wait_and_throw();
    }
    q.fill(bucket_sizes.p, 0, bucket_sizes.size).wait_and_throw();
    build_hash(q, bins.p, output_weights.value.p, shape, bucket_nodes.p,
               bucket_sizes.p);
    ++rebuild_count;
  }
  void train(const sparse_batch &batch, float corrected_rate,
             bool rebuild_index, bool reshuffle,
             delta_reset reset = delta_reset::each_step) {
    require(std::isfinite(corrected_rate) && corrected_rate >= 0,
            "finite nonnegative learning rate");
    validate_batch(batch);
    if (batch.size() == 0)
      return;
    TrainingOperations ops;
    ops.hidden_layers = 1;
    ops.transfer = [&] { transfer(batch); };
    ops.hidden_forward = [&](int) {
      dense_ids(q, hidden_ids.p, hidden_offsets.p, hidden, current_batch,
                max_batch * hidden);
      dense_product(q, dense_inputs.p, hidden_weights.value.p,
                    hidden_bias.value.p, hidden_values.p, current_batch, inputs,
                    hidden, false, true);
    };
    ops.output_forward = [&] { output_forward(); };
    ops.reset_hidden_deltas = [&] {
      if (reset == delta_reset::each_step)
        q.fill(hidden_deltas.p, 0.f, hidden_deltas.size).wait_and_throw();
    };
    ops.output_backward = [&] { output_backward(); };
    ops.hidden_backward = [&](int) { hidden_backward(); };
    ops.hidden_update = [&](int) {
      update(hidden_weights, corrected_rate);
      update(hidden_bias, corrected_rate);
    };
    ops.output_update = [&] {
      update(output_weights, corrected_rate);
      update(output_bias, corrected_rate);
    };
    ops.rebuild_index = [&](bool shuffle) { rebuild(shuffle); };
    ops.complete = [&] { q.wait_and_throw(); };
    train_step(ops, rebuild_index, reshuffle);
    ++step_count;
  }
  const int *bin_data() const { return bins.p; }
  // Numerical variant: includes learned output bias. The native cuBLAS
  // forward_dense currently omits it. Evaluation accepts unlabeled samples.
  void evaluate_dense(const sparse_batch &batch, float *logits) {
    transfer(batch, false);
    dense_product(q, dense_inputs.p, hidden_weights.value.p,
                  hidden_bias.value.p, hidden_values.p, current_batch, inputs,
                  hidden, false, true);
    dense_product(q, hidden_values.p, output_weights.value.p,
                  output_bias.value.p, logits, current_batch, hidden, outputs,
                  true, false);
  }
};
} // namespace gslide::sycl_port
