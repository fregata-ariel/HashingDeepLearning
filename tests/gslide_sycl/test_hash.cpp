#include "hash_index.hpp"
#include "test_support.hpp"
static int oracle_address(const int *bins, const float *row, hash_shape z,
                          int table) {
  int a = 0;
  for (int j = 0; j < z.k; ++j) {
    int best = 0;
    for (int k = 1; k < z.bin; ++k)
      if (row[bins[(table * z.k + j) * z.bin + k]] >
          row[bins[(table * z.k + j) * z.bin + best]])
        best = k;
    a += best * int(std::pow(2, (z.k - 1 - j) * int(std::log(z.bin))));
  }
  return a;
}
static void test_hash(sycl::queue &q) {
  for (int nodes : {1, 31, 33, 65})
    for (int per_thread : {1, 2, 4}) {
      hash_shape z{7, nodes, 5, 2, 4, 4, per_thread, 32, 128};
      int nb = z.tables * z.buckets;
      memory<int> bins(q, z.tables * z.k * z.bin), buckets(q, nb * z.capacity),
          sizes(q, nb);
      memory<float> w(q, z.nodes * z.previous);
      for (int i = 0; i < bins.size; ++i)
        bins[i] = (i * 3 + i / 4) % z.previous;
      for (int n = 0; n < nodes; ++n)
        for (int j = 0; j < z.previous; ++j)
          w[n * z.previous + j] =
              n % 3 == 0 ? -5.f : float((n * 7 + j * 3) % 11) - 12;
      build_hash(q, bins.p, w.p, z, buckets.p, sizes.p, 64);
      std::vector<std::set<int>> expected(nb);
      for (int n = 0; n < nodes; ++n)
        for (int t = 0; t < z.tables; ++t)
          expected[t * z.buckets +
                   oracle_address(bins.p, w.p + n * z.previous, z, t)]
              .insert(n);
      for (int a = 0; a < nb; ++a) {
        check(sizes[a] == int(expected[a].size()), "hash bucket count");
        std::set<int> actual(buckets.p + a * z.capacity,
                             buckets.p + a * z.capacity + sizes[a]);
        check(actual == expected[a], "hash membership/ties/tail/lane staging");
      }
      // Query independently generated vectors: address and membership
      // consistency.
      memory<int> query(q, z.tables);
      q.parallel_for(sycl::range<1>(z.tables), [=, bp = bins.p, wp = w.p,
                                                out = query.p](sycl::id<1> it) {
         int t = it[0];
         out[t] = address(bp + t * z.k * z.bin, wp, z.k, z.bin,
                          int(sycl::log(float(z.bin))));
       }).wait_and_throw();
      for (int t = 0; t < z.tables; ++t) {
        check(query[t] == oracle_address(bins.p, w.p, z, t), "query address");
        check(expected[t * z.buckets + query[t]].count(0) == 1,
              "query rebuilt membership");
      }
    }
  // Capacity boundary: no defined survivor order under overflow; count and
  // retained IDs are checked for a ring with multiple concurrent producers.
  hash_shape z{2, 65, 1, 1, 2, 1, 1, 2, 2};
  memory<int> bins(q, 2), b(q, 4), s(q, 2);
  memory<float> w(q, 130);
  bins[1] = 1;
  build_hash(q, bins.p, w.p, z, b.p, s.p);
  check(s[0] == 65, "ring total count");
  check(b[0] >= 0 && b[0] < 65 && b[1] >= 0 && b[1] < 65, "ring bounded IDs");
}
int main() { return run_device_suite("hash", test_hash); }
