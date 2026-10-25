#include "candidates.hpp"
#include "test_support.hpp"
static unsigned oracle_random(unsigned seed, int id) {
  std::uint64_t x = seed % 2147483647u;
  if (!x)
    x = 1;
  for (int i = 0; i <= id; ++i)
    x = x * 48271 % 2147483647u;
  return x;
}
static void test_candidates(sycl::queue &q) {
  constexpr int batch = 65, nodes = 7, L = 3, threshold = 2;
  memory<int> off(q, batch * L + 1), raw(q, batch * 6),
      counts(q, batch * nodes + 1), labels(q, batch * 2), loff(q, batch + 1),
      perm(q, nodes), sizes(q, batch + 1);
  std::vector<std::vector<int>> expected(batch, std::vector<int>(nodes));
  for (int b = 0; b < batch; ++b) {
    off[b * L] = b * 6;
    off[b * L + 1] = b * 6 + 3;
    off[b * L + 2] = b * 6 + 3;
    off[b * L + 3] = b * 6 + 6;
    for (int j = 0; j < 6; ++j) {
      int n = (b + j / 2) % nodes;
      raw[b * 6 + j] = n;
      ++expected[b][n];
    }
    loff[b] = b * 2;
    labels[b * 2] = (b + 3) % nodes;
    labels[b * 2 + 1] = (b + 3) % nodes;
  }
  loff[batch] = batch * 2;
  for (int n = 0; n < nodes; ++n)
    perm[n] = (n * 3) % nodes;
  counts[batch * nodes] = -91;
  sizes[batch] = -92;
  count_candidates(q, raw.p, off.p, L, batch, nodes, batch * nodes, counts.p);
  for (int b = 0; b < batch; ++b)
    for (int n = 0; n < nodes; ++n)
      check(counts[b * nodes + n] == expected[b][n],
            "wrapper L/sample frequency");
  activate_labels(q, labels.p, loff.p, nullptr, batch, nodes, threshold, 0, 40,
                  counts.p, sizes.p);
  for (int b = 0; b < batch; ++b) {
    expected[b][(b + 3) % nodes] += 4;
    int size = 0;
    for (int n = 0; n < nodes; ++n) {
      check(counts[b * nodes + n] == expected[b][n],
            "label wrapper duplicate count");
      size += expected[b][n] >= threshold;
    }
    check(sizes[b] == size, "label-only size");
  }
  for (int floor : {0, 5, 7}) {
    q.fill(counts.p, 0, batch * nodes).wait_and_throw();
    activate_labels(q, labels.p, loff.p, perm.p, batch, nodes, threshold, floor,
                    40, counts.p, sizes.p);
    for (int b = 0; b < batch; ++b) {
      std::vector<int> ref(nodes);
      ref[(b + 3) % nodes] = 4;
      int size = 1, i = oracle_random(40, b) % nodes;
      while (size < floor) {
        int n = perm[i];
        if (ref[n] < threshold)
          ++size;
        ref[n] += threshold;
        i = (i + 1) % nodes;
      }
      for (int n = 0; n < nodes; ++n)
        check(counts[b * nodes + n] == ref[n],
              "random padding fixed seed/cycle");
      check(sizes[b] == std::max(1, floor), "minimum floor");
    }
  }
  check(counts[batch * nodes] == -91 && sizes[batch] == -92,
        "candidate guards");
  bool rejected = false;
  try {
    activate_labels(q, labels.p, loff.p, perm.p, batch, nodes, threshold, 8, 40,
                    counts.p, sizes.p);
  } catch (std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "impossible floor reject");
  // Empty/partial samples and reuse after clear.
  q.fill(counts.p, 0, batch * nodes).wait_and_throw();
  off[0] = off[1] = off[2] = off[3] = 0;
  count_candidates(q, raw.p, off.p, L, 1, nodes, batch * nodes, counts.p);
  loff[0] = loff[1] = 0;
  activate_labels(q, labels.p, loff.p, nullptr, 1, nodes, threshold, 0, 40,
                  counts.p, sizes.p);
  check(sizes[0] == 0, "empty reuse");
  count_candidates(q, raw.p, off.p, L, 0, nodes, batch * nodes, counts.p);
}
int main() { return run_device_suite("candidates", test_candidates); }
