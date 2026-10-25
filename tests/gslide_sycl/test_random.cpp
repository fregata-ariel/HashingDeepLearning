#include "random.hpp"
#include "test_support.hpp"
static unsigned oracle_random(unsigned seed, int id) {
  std::uint64_t x = seed % 2147483647u;
  if (!x)
    x = 1;
  for (int i = 0; i <= id; ++i)
    x = x * 48271 % 2147483647u;
  return x;
}
static void test(sycl::queue &q) {
  for (unsigned seed : {0u, 1u, 40u, 2147483647u}) {
    memory<unsigned> a(q, 131), b(q, 131);
    random_keys(q, a.p, 129, seed, 7, false);
    random_keys(q, b.p, 129, seed, 7, true);
    std::set<unsigned> seen;
    for (int i = 0; i < 129; ++i) {
      auto x = oracle_random(seed, i);
      check(a[i] == x && a[i] > 0 && a[i] < 2147483647u,
            "RNG minstd range/exact sequence");
      check(b[i] == (unsigned(i / 7) << 16 | (x & 32768u)),
            "packed original precedence variant");
      seen.insert(a[i]);
    }
    check(seen.size() == 129, "RNG nonconstant");
    random_keys(q, b.p, 129, seed, 7, false);
    for (int i = 0; i < 129; ++i)
      check(a[i] == b[i], "RNG repeat");
  }
}
int main() { return run_device_suite("random", test); }
