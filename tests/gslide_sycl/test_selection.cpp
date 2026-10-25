#include "selection.hpp"
#include "test_support.hpp"
#include <random>
static void test(sycl::queue &queue) {
  constexpr int NB = 2, N = 4, H = 3, T = 2;
  hash_shape shape{H, N, T, 1, 2, 2, 2, 2, N};
  memory<int> bins(queue, 4), random_nodes(queue, N), bucket_sizes(queue, 4),
      bucket_nodes(queue, 4 * N);
  bins.assign({0, 1, 1, 2});
  random_nodes.assign({2, 0, 3, 1});
  memory<float> weights(queue, N * H), hidden(queue, NB * H);
  weights.assign({1, 0, -1, 0, 2, 1, 3, 1, 4, -.5f, -1, 0});
  hidden.assign({1, 0, 3, 0, 2, 1});
  build_hash(queue, bins.p, weights.p, shape, bucket_nodes.p, bucket_sizes.p);
  memory<int> labels(queue, 2), label_offsets(queue, 3),
      addresses(queue, NB * T), lengths(queue, NB * T),
      gather_offsets(queue, NB * T + 1), gathered(queue, NB * T * N),
      counts(queue, NB * N), sizes(queue, NB), active_offsets(queue, NB + 1),
      active_nodes(queue, NB * N);
  labels.assign({3, 0});
  label_offsets.assign({0, 1, 2});
  auto invoke = [&] {
    select_candidates(queue, shape, NB, H, N, NB * N, 2, 2, 40,
                      {bins.p, random_nodes.p, bucket_sizes.p, bucket_nodes.p,
                       hidden.p, labels.p, label_offsets.p, addresses.p,
                       lengths.p, gather_offsets.p, gathered.p, counts.p,
                       sizes.p, active_offsets.p, active_nodes.p});
  };
  for (int repeat = 0; repeat < 2; ++repeat) {
    invoke();
    int total = 0;
    for (int b = 0; b < NB; ++b) {
      std::vector<int> expected(N, 0);
      for (int t = 0; t < T; ++t) {
        int winner =
            hidden[b * H + bins[t * 2 + 1]] > hidden[b * H + bins[t * 2]] ? 1
                                                                          : 0;
        check(addresses[b * T + t] == winner, "query address");
        for (int node = 0; node < N; ++node) {
          int address = weights[node * H + bins[t * 2 + 1]] >
                                weights[node * H + bins[t * 2]]
                            ? 1
                            : 0;
          if (address == winner)
            ++expected[node];
        }
      }
      expected[labels[b]] += 2;
      int selected = 0;
      for (int value : expected)
        selected += value >= 2;
      std::minstd_rand rng(40);
      rng.discard(b);
      int pos = rng() % N;
      while (selected < 2) {
        int node = random_nodes[pos];
        if (expected[node] < 2)
          ++selected;
        expected[node] += 2;
        pos = (pos + 1) % N;
      }
      check(active_offsets[b] == total, "selected offsets");
      for (int node = 0; node < N; ++node) {
        check(counts[b * N + node] == expected[node],
              "query/frequency/label/padding oracle");
        if (expected[node] >= 2)
          check(active_nodes[total++] == node, "selected node order");
      }
      check(sizes[b] == selected, "selected count");
    }
    check(active_offsets[NB] == total, "selected terminal offset");
  }
}
int main() { return run_device_suite("selection", test); }
