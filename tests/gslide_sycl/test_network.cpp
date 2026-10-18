// TRACE_TEST_ID: GSLIDE-SYCL-PRODUCTION-116
#include "hashing/sycl_backend.hpp"
#include "network.hpp"
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <string>
using namespace gslide::sycl_port;
static int checks = 0;
static double max_error = 0;
static void check(bool value, const char *message) {
  ++checks;
  if (!value)
    throw std::runtime_error(message);
}
static void near(double actual, double expected, double tolerance = 8e-6) {
  max_error = std::max(max_error, std::abs(actual - expected));
  check(std::isfinite(actual) && std::abs(actual - expected) < tolerance,
        "independent FP64 mismatch");
}
using D = std::vector<double>;
using F = std::vector<float>;
constexpr int I = 3, H = 4, O = 4, N = 2, L = 3, K = 2, BIN = 4, B = 16;
static const F hw{.7f, .2f,  .1f, -.4f, .1f, .5f,
                  .7f, -.5f, .2f, -.3f, .1f, .1f};
static const F hb{.05f, .02f, .03f, -.3f};
static const F ow{.6f,  .2f, .1f, -.2f, .5f, -.1f, .49f, -.3f,
                  -.1f, .2f, .7f, -.2f, .1f, .6f,  .2f,  -.1f};
static const F ob{.03f, -.04f, .05f, -.01f};
static std::vector<int> initial_bins() {
  std::vector<int> a(L * K * H);
  for (int t = 0; t < L; ++t)
    for (int h = 0; h < K; ++h)
      for (int p = 0; p < H; ++p)
        a[(t * K + h) * H + p] = (p + 40 + t * K + h) % H;
  return a;
}
static sparse_batch batch1() {
  return {{0, 2, 1, 2}, {0, 2, 4}, {3, 1}, {0, 1, 2}, {1, .5f, 1.2f, -.4f}};
}
static sparse_batch batch2() {
  return {{1, 2, 0, 2}, {0, 2, 4}, {0, 2}, {0, 1, 2}, {.8f, .25f, .6f, -.2f}};
}
struct Param {
  D value, moment, velocity;
  Param(const F &f)
      : value(f.begin(), f.end()), moment(f.size()), velocity(f.size()) {};
  void update(const D &g, double rate) {
    for (unsigned i = 0; i < value.size(); ++i) {
      moment[i] = .9 * moment[i] + .1 * g[i];
      velocity[i] = .999 * velocity[i] + .001 * g[i] * g[i];
      value[i] += rate * moment[i] / (std::sqrt(velocity[i]) + 1e-8);
    }
  }
};
static int address_ref(const D &row, const std::vector<int> &bins, int table) {
  int a = 0;
  for (int j = 0; j < K; ++j) {
    int win = 0;
    for (int p = 1; p < BIN; ++p)
      if (row[bins[(table * K + j) * BIN + p]] >
          row[bins[(table * K + j) * BIN + win]])
        win = p;
    a += win * int(std::pow(2, (K - j - 1) * int(std::log(BIN))));
  }
  return a;
}
struct Oracle {
  Param hw{::hw}, hb{::hb}, ow{::ow}, ob{::ob};
  std::vector<int> bins = initial_bins();
  std::vector<std::set<int>> buckets;
  D hidden, hidden_delta, prob, delta;
  std::vector<std::vector<int>> active;
  int changed = 0;
  Oracle() { rebuild(); }
  void rebuild() {
    std::vector<std::set<int>> next(L * B);
    for (int n = 0; n < O; ++n) {
      D row(ow.value.begin() + n * H, ow.value.begin() + (n + 1) * H);
      for (int t = 0; t < L; ++t)
        next[t * B + address_ref(row, bins, t)].insert(n);
    }
    if (!buckets.empty())
      for (int i = 0; i < L * B; ++i)
        changed += next[i] != buckets[i];
    buckets = next;
  }
  void train(const sparse_batch &batch, double rate) {
    int nb = batch.size();
    D x(nb * I);
    for (int b = 0; b < nb; ++b)
      for (int p = batch.offsets[b]; p < batch.offsets[b + 1]; ++p)
        x[b * I + batch.nodes[p]] += batch.values[p];
    hidden.assign(nb * H, 0);
    hidden_delta.assign(nb * H, 0);
    active.assign(nb, {});
    prob.assign(nb * O, 0);
    delta.assign(nb * O, 0);
    D ghw(I * H), ghb(H), gow(O * H), gob(O);
    for (int b = 0; b < nb; ++b) {
      D row(H);
      for (int h = 0; h < H; ++h) {
        double a = hb.value[h];
        for (int i = 0; i < I; ++i)
          a += x[b * I + i] * hw.value[i * H + h];
        row[h] = hidden[b * H + h] = std::max(0., a);
      }
      std::map<int, int> freq;
      for (int t = 0; t < L; ++t)
        for (int n : buckets[t * B + address_ref(row, bins, t)])
          ++freq[n];
      for (int p = batch.label_offsets[b]; p < batch.label_offsets[b + 1]; ++p)
        freq[batch.labels[p]] += 2;
      for (auto [n, count] : freq)
        if (count >= 2)
          active[b].push_back(n);
      double maximum = -1e300;
      for (int n : active[b]) {
        double v = ob.value[n];
        for (int h = 0; h < H; ++h)
          v += row[h] * ow.value[n * H + h];
        prob[b * O + n] = v;
        maximum = std::max(maximum, v);
      }
      double sum = 0;
      for (int n : active[b])
        sum += prob[b * O + n] = std::exp(prob[b * O + n] - maximum);
      for (int n : active[b]) {
        prob[b * O + n] /= sum;
        bool label = false;
        for (int p = batch.label_offsets[b]; p < batch.label_offsets[b + 1];
             ++p)
          label |= batch.labels[p] == n;
        double d =
            ((label ? 1. / (batch.label_offsets[b + 1] - batch.label_offsets[b])
                    : 0.) -
             prob[b * O + n]) /
            nb;
        delta[b * O + n] = d;
        gob[n] += d;
        for (int h = 0; h < H; ++h) {
          gow[n * H + h] += d * row[h];
          if (row[h] > 0)
            hidden_delta[b * H + h] += d * ow.value[n * H + h];
        }
      }
    }
    for (int b = 0; b < nb; ++b)
      for (int h = 0; h < H; ++h) {
        ghb[h] += hidden_delta[b * H + h];
        for (int i = 0; i < I; ++i)
          ghw[i * H + h] += hidden_delta[b * H + h] * x[b * I + i];
      }
    hw.update(ghw, rate);
    hb.update(ghb, rate);
    ow.update(gow, rate);
    ob.update(gob, rate);
  }
};
static void compare_parameter(parameter &actual, const Param &expected) {
  for (int i = 0; i < actual.value.size; ++i) {
    near(actual.value[i], expected.value[i]);
    near(actual.moment[i], expected.moment[i]);
    near(actual.velocity[i], expected.velocity[i]);
    near(actual.gradient[i], 0, 1e-12);
  }
}
static std::vector<int> active_signature(Network &net, int nb) {
  std::vector<int> a;
  for (int b = 0; b < nb; ++b) {
    a.push_back(-1);
    for (int p = net.active_offsets[b]; p < net.active_offsets[b + 1]; ++p)
      a.push_back(net.active_nodes[p]);
  }
  return a;
}
static void compare(Network &net, Oracle &ref, int nb) {
  compare_parameter(net.hidden_weights, ref.hw);
  compare_parameter(net.hidden_bias, ref.hb);
  compare_parameter(net.output_weights, ref.ow);
  compare_parameter(net.output_bias, ref.ob);
  for (int b = 0; b < nb; ++b) {
    std::set<int> actual;
    for (int p = net.active_offsets[b]; p < net.active_offsets[b + 1]; ++p) {
      int n = net.active_nodes[p];
      actual.insert(n);
      near(net.probabilities[p], ref.prob[b * O + n]);
      near(net.output_deltas[p], ref.delta[b * O + n]);
    }
    check(actual == std::set<int>(ref.active[b].begin(), ref.active[b].end()),
          "active set independent oracle");
    for (int h = 0; h < H; ++h) {
      near(net.hidden_values[b * H + h], ref.hidden[b * H + h]);
      near(net.hidden_deltas[b * H + h], ref.hidden_delta[b * H + h]);
    }
  }
}
static std::unique_ptr<Network> network(sycl::queue &q) {
  return std::make_unique<Network>(q, I, H, O, N, 4, 2, hw, hb, ow, ob,
                                   initial_bins());
}
static void test_connected(sycl::queue &q) {
  bool overflow_rejected = false;
  try {
    Network invalid(q, I, H, O, N, 4, 2, hw, hb, ow, ob, {},
                    std::numeric_limits<int>::max(),
                    std::numeric_limits<int>::max(),
                    std::numeric_limits<int>::max());
  } catch (const std::invalid_argument &) {
    overflow_rejected = true;
  }
  check(overflow_rejected,
        "configuration extent rejects overflow before allocation");
  auto net = network(q), stale = network(q), stale_index = network(q);
  Oracle ref;
  std::vector<int> first_signature;
  std::vector<int> initial_sizes(net->bucket_sizes.p,
                                 net->bucket_sizes.p + L * B);
  int changed = 0;
  for (int step = 1; step <= 2; ++step) {
    auto batch = step == 1 ? batch1() : batch2();
    float rate =
        .01 * std::sqrt(1 - std::pow(.999, step)) / (1 - std::pow(.9, step));
    net->train(batch, rate, true, false);
    stale->train(batch, rate, true, false, delta_reset::diagnostic_keep_stale);
    stale_index->train(batch, rate, false, false);
    ref.train(batch, rate);
    compare(*net, ref, batch.size());
    ref.rebuild();
    for (int a = 0; a < L * B; ++a) {
      check(net->bucket_sizes[a] == int(ref.buckets[a].size()),
            "rebuild count");
      std::set<int> actual(net->bucket_nodes.p + a * O,
                           net->bucket_nodes.p + a * O + net->bucket_sizes[a]);
      check(actual == ref.buckets[a], "rebuild bucket sets");
    }
    if (step == 1)
      first_signature = active_signature(*net, N);
    else {
      check(first_signature != active_signature(*net, N), "active sets change");
      check(active_signature(*net, N) != active_signature(*stale_index, N),
            "stale index changes queried active set");
    }
  }
  check(net->step_count == 2 && net->rebuild_count == 3,
        "production control rebuild cadence");
  check(stale_index->rebuild_count == 1, "rebuild disabled counter");
  check(ref.changed > 0, "genuine Adam update changes WTA buckets");
  double stale_delta_error = 0, stale_parameter_error = 0;
  for (int i = 0; i < N * H; ++i)
    stale_delta_error = std::max(
        stale_delta_error,
        double(std::abs(net->hidden_deltas[i] - stale->hidden_deltas[i])));
  for (int i = 0; i < I * H; ++i)
    stale_parameter_error =
        std::max(stale_parameter_error,
                 double(std::abs(net->hidden_weights.value[i] -
                                 stale->hidden_weights.value[i])));
  check(stale_delta_error > 1e-3 && stale_parameter_error > 1e-5,
        "#43 stale delta negative changes actual parameters");
  bool stale_buckets = false;
  for (int a = 0; a < L * B; ++a)
    stale_buckets |= net->bucket_sizes[a] != stale_index->bucket_sizes[a];
  check(stale_buckets, "stale index negative");
  // The caller resets full owned capacity, even for partial batches. ReLU gated
  // inactive hidden nodes must stay zero; sentinel tail must be cleared too.
  for (int i = 0; i < N * H; ++i)
    net->hidden_deltas[i] = 123;
  sparse_batch partial{{0, 2}, {0, 2}, {1}, {0, 1}, {.2f, .1f}};
  net->train(partial, .001f, false, false);
  ref.train(partial, .001f);
  compare(*net, ref, 1);
  for (int i = H; i < N * H; ++i)
    near(net->hidden_deltas[i], 0, 1e-12);
  check(net->hidden_values[3] == 0 && net->hidden_deltas[3] == 0,
        "ReLU gating after reset");
  usm_array<float> logits(q, O);
  auto unlabeled = partial;
  unlabeled.labels.clear();
  unlabeled.label_offsets = {0, 0};
  net->evaluate_dense(unlabeled, logits.p);
  D dense_hidden(H);
  for (int h = 0; h < H; ++h)
    dense_hidden[h] = std::max(
        0., ref.hb.value[h] + double(partial.values[0]) * ref.hw.value[h] +
                double(partial.values[1]) * ref.hw.value[2 * H + h]);
  for (int o = 0; o < O; ++o) {
    double expected = ref.ob.value[o];
    for (int h = 0; h < H; ++h)
      expected += dense_hidden[h] * ref.ow.value[o * H + h];
    near(logits[o], expected);
  }
  F before(net->output_weights.value.p, net->output_weights.value.p + O * H);
  sparse_batch empty{{}, {0}, {}, {0}, {}};
  int steps = net->step_count;
  net->train(empty, .001f, true, false);
  check(net->step_count == steps, "empty training no-op");
  for (int i = 0; i < O * H; ++i)
    check(before[i] == net->output_weights.value[i],
          "empty parameter unchanged");
  bool reject = false;
  auto invalid = batch1();
  invalid.values.push_back(1);
  invalid.nodes.push_back(0);
  try {
    net->train(invalid, .001f, false, false);
  } catch (std::invalid_argument &) {
    reject = true;
  }
  check(reject, "input capacity rejected");
  net->rebuild(true);
  for (int block = 0; block < L * K; ++block) {
    std::set<int> ids(net->bin_data() + block * H,
                      net->bin_data() + (block + 1) * H);
    check(ids == std::set<int>({0, 1, 2, 3}),
          "RNG key sort permutation contract");
  }
  std::cout << "connected_updates=2 stale_delta_error=" << stale_delta_error
            << " stale_parameter_error=" << stale_parameter_error
            << " changed_buckets=" << ref.changed << '\n';
}
static void test_library_boundary(sycl::queue &q) {
  usm_array<int> lengths(q, 4), offsets(q, 5);
  for (int i = 0; i < 4; ++i)
    lengths[i] = i % 3;
  scan_lengths(q, lengths.p, offsets.p, 4);
  int sum = 0;
  for (int i = 0; i <= 4; ++i) {
    check(offsets[i] == sum, "scan independent offsets");
    if (i < 4)
      sum += lengths[i];
  }
  scan_lengths(q, lengths.p, offsets.p, 0);
  check(offsets[0] == 0, "empty scan");
  usm_array<unsigned> keys(q, 5), sorted(q, 5);
  usm_array<int> values(q, 5), result(q, 5);
  unsigned k[] = {3, 1, 1, 4, 0};
  for (int i = 0; i < 5; ++i) {
    keys[i] = k[i];
    values[i] = 10 + i;
  }
  sort_keys(q, keys.p, values.p, sorted.p, result.p, 5);
  std::multiset<std::pair<unsigned, int>> expected, actual;
  for (int i = 0; i < 5; ++i) {
    expected.emplace(keys[i], values[i]);
    actual.emplace(sorted[i], result[i]);
    if (i)
      check(sorted[i - 1] <= sorted[i], "sort order");
  }
  check(actual == expected, "sort correspondence/ties");
  sort_keys(q, keys.p, values.p, sorted.p, result.p, 0);
  usm_array<float> x(q, 6), row(q, 6), col(q, 6), bias(q, 2), yr(q, 4),
      yc(q, 4);
  for (int i = 0; i < 6; ++i)
    x[i] = (i - 2) * .3f;
  for (int o = 0; o < 2; ++o)
    for (int i = 0; i < 3; ++i)
      row[o * 3 + i] = col[i * 2 + o] = float(i - o) * .4f;
  bias[0] = -.7f;
  bias[1] = .2f;
  dense_product(q, x.p, row.p, bias.p, yr.p, 2, 3, 2, true, false);
  dense_product(q, x.p, col.p, bias.p, yc.p, 2, 3, 2, false, false);
  for (int b = 0; b < 2; ++b)
    for (int o = 0; o < 2; ++o) {
      double expected = bias[o];
      for (int i = 0; i < 3; ++i)
        expected += double(x[b * 3 + i]) * row[o * 3 + i];
      near(yr[b * 2 + o], expected);
      near(yc[b * 2 + o], expected);
    }
  dense_product(q, x.p, row.p, bias.p, yr.p, 0, 3, 2, true, false);
}
static void test_control_order() {
  std::string order;
  gslide::TrainingOperations op;
  op.hidden_layers = 2;
  op.transfer = [&] { order += 'T'; };
  op.hidden_forward = [&](int i) { order += char('a' + i); };
  op.output_forward = [&] { order += 'F'; };
  op.reset_hidden_deltas = [&] { order += 'R'; };
  op.output_backward = [&] { order += 'B'; };
  op.hidden_backward = [&](int i) { order += char('c' + i); };
  op.hidden_update = [&](int i) { order += char('e' + i); };
  op.output_update = [&] { order += 'U'; };
  op.rebuild_index = [&](bool shuffle) { order += shuffle ? 'S' : 'I'; };
  op.complete = [&] { order += 'C'; };
  gslide::train_step(op, false, true);
  check(order == "TabFRBdcefUSC", "production control exact ordering");
}
int main() {
  try {
    auto q = hashing::sycl_backend::make_queue(
        hashing::sycl_backend::device_kind::cpu);
    hashing::sycl_backend::require_cpu(q);
    hashing::sycl_backend::describe_queue(std::cout, q);
    test_control_order();
    test_library_boundary(q);
    test_connected(q);
    std::cout << "GSLIDE_SYCL_PRODUCTION checks=" << checks
              << " max_abs_error=" << max_error
              << " gpu_validated=false native_cuda_train_executed=false\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
