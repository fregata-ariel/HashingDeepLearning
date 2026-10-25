#include "dense_ids.hpp"
#include "test_support.hpp"
static void test(sycl::queue &q) {
  for (int nodes : {0, 1, 3, 65})
    for (int batch : {0, 1, 3}) {
      int n = nodes * batch;
      memory<int> ids(q, n + 1), off(q, batch + 2);
      ids[n] = -71;
      off[batch + 1] = -73;
      dense_ids(q, ids.p, off.p, nodes, batch, n);
      for (int i = 0; i < n; ++i)
        check(ids[i] == i % nodes, "dense ids");
      for (int i = 0; i <= batch; ++i)
        check(off[i] == i * nodes, "dense offsets");
      check(ids[n] == -71 && off[batch + 1] == -73, "dense guards");
    }
  memory<int> i(q, 1), o(q, 2);
  bool rejected = false;
  try {
    dense_ids(q, i.p, o.p, 2, 1, 1);
  } catch (std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "dense reject capacity");
}
int main() { return run_device_suite("dense_ids", test); }
