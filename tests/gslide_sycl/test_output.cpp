#include "component_api.hpp"
#include "test_support.hpp"
static void test(sycl::queue &queue) {
  constexpr int B = 2, H = 3, O = 4, I = 2;
  memory<int> off(queue, 3), ids(queue, 3), labs(queue, 2), loff(queue, 3);
  off.assign({0, 2, 3});
  ids.assign({3, 1, 0});
  labs.assign({3, 0});
  loff.assign({0, 1, 2});
  memory<float> hv(queue, B * H), w(queue, O * H), bias(queue, O),
      prob(queue, 3), delta(queue, 3);
  hv.assign({1, 0, 2, 0, 3, 0});
  bias.assign({-.3f, .4f, .2f, -.1f});
  for (int i = 0; i < O * H; ++i)
    w[i] = .2f * i - .5f;
  double p[3], d[3];
  for (int b = 0; b < B; ++b) {
    double maximum = -1e30, total = 0;
    for (int j = off[b]; j < off[b + 1]; ++j) {
      p[j] = bias[ids[j]];
      for (int h = 0; h < H; ++h)
        p[j] += double(hv[b * H + h]) * w[ids[j] * H + h];
      maximum = std::max(maximum, p[j]);
    }
    for (int j = off[b]; j < off[b + 1]; ++j)
      total += p[j] = std::exp(p[j] - maximum);
    for (int j = off[b]; j < off[b + 1]; ++j) {
      p[j] /= total;
      d[j] = ((ids[j] == labs[b] ? 1. : 0.) - p[j]) / B;
    }
  }
  SelectedOutput input{{read_only(hv.view()), B, H, Layout::row_major},
                       {read_only(w.view()), O, H, Layout::row_major},
                       {read_only(off.view()), read_only(ids.view()), B, 3}};
  selected_softmax(queue, input, read_only(bias.view()),
                   {read_only(loff.view()), read_only(labs.view()), B, 2},
                   prob.view(), delta.view());
  for (int j = 0; j < 3; ++j) {
    near(prob[j], p[j]);
    near(delta[j], d[j]);
  }
  memory<float> hd(queue, B * H), gw(queue, O * H), gb(queue, O);
  double expected_hd[B * H] = {}, expected_gw[O * H] = {}, expected_gb[O] = {};
  // Two contributions exercise accumulation without a Network/reset controller.
  for (int repeat = 0; repeat < 2; ++repeat) {
    for (int b = 0; b < B; ++b)
      for (int j = off[b]; j < off[b + 1]; ++j) {
        expected_gb[ids[j]] += d[j];
        for (int h = 0; h < H; ++h) {
          expected_gw[ids[j] * H + h] += d[j] * hv[b * H + h];
          if (hv[b * H + h] > 0)
            expected_hd[b * H + h] += d[j] * w[ids[j] * H + h];
        }
      }
    accumulate_output_gradients(queue, input, read_only(delta.view()),
                                {{hd.view(), B, H, Layout::row_major},
                                 {gw.view(), O, H, Layout::row_major},
                                 gb.view()});
    for (int i = 0; i < B * H; ++i)
      near(hd[i], expected_hd[i]);
    for (int i = 0; i < O * H; ++i)
      near(gw[i], expected_gw[i]);
    for (int i = 0; i < O; ++i)
      near(gb[i], expected_gb[i]);
  }
  memory<float> x(queue, B * I), hgw(queue, I * H), hgb(queue, H);
  x.assign({1, -.5f, 2, .25f});
  accumulate_hidden_gradients(
      queue, {read_only(x.view()), B, I, Layout::row_major},
      {read_only(hd.view()), B, H, Layout::row_major},
      {hgw.view(), H, I, Layout::column_major}, hgb.view());
  for (int h = 0; h < H; ++h) {
    double expected = 0;
    for (int b = 0; b < B; ++b)
      expected += expected_hd[b * H + h];
    near(hgb[h], expected);
    for (int i = 0; i < I; ++i) {
      expected = 0;
      for (int b = 0; b < B; ++b)
        expected += double(x[b * I + i]) * expected_hd[b * H + h];
      near(hgw[i * H + h], expected);
    }
  }
  // Characterize the released membership/listed-count rule separately from
  // the normal distinct-label oracle above. Duplicates are not deduplicated.
  memory<int> duplicate_labels(queue, 3), duplicate_offsets(queue, 3);
  duplicate_labels.assign({3, 3, 0});
  duplicate_offsets.assign({0, 2, 3});
  memory<float> duplicate_prob(queue, 3), duplicate_delta(queue, 3);
  selected_softmax(queue, input, read_only(bias.view()),
                   {read_only(duplicate_offsets.view()),
                    read_only(duplicate_labels.view()), B, 3},
                   duplicate_prob.view(), duplicate_delta.view());
  for (int j = 0; j < 3; ++j) {
    near(duplicate_prob[j], p[j]);
    double target = j == 0 ? .5 : j == 2 ? 1. : 0.;
    near(duplicate_delta[j], (target - p[j]) / B);
  }
  near(duplicate_delta[0] + duplicate_delta[1], -.5 / B);
}
int main() { return run_device_suite("output", test); }
