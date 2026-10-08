// Executes selected transform_kernel; expectations use independent bit events
// and FP64 binomial mass, not the production cancellation expression.
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace probability_fixture {
int checks = 0;
int states = 0;
int correction_cases = 0;
int domain_cases = 0;
double worst = 0;

void require(bool ok, const std::string& label) {
  ++checks;
  if (!ok) {
    std::cerr << "MAGICPIG_PROBABILITY_FAIL " << label << '\n';
    std::exit(1);
  }
}

void close(double actual, double expected, double tolerance,
           const std::string& label) {
  require(std::isfinite(actual) && std::isfinite(expected), label + " finite");
  const double error = std::abs(actual - expected);
  worst = std::max(worst, error);
  require(error <= tolerance, label + " error=" + std::to_string(error));
}

// Sum the positive binomial masses for 2..L hits directly. This avoids the
// subtraction of zero/one-hit probabilities used by the selected body.
double inclusion(double p, int l) {
  double sum = 0;
  double choose = 1;
  for (int h = 0; h <= l; ++h) {
    if (h >= 2)
      sum += choose * std::pow(p, h) * std::pow(1 - p, l - h);
    if (h < l) choose *= double(l - h) / (h + 1);
  }
  return sum;
}

// Independently enumerate individual bit outcomes. All K bits must match to
// hit a table; at least two table hits include the candidate. This neither
// calls inclusion() nor copies the source's closed-form subtraction.
double enumeration(double r, int k, int l) {
  double sum = 0;
  for (unsigned mask = 0; mask < (1U << (k * l)); ++mask) {
    double mass = 1;
    int hits = 0;
    for (int table = 0; table < l; ++table) {
      bool hit = true;
      for (int bit = 0; bit < k; ++bit) {
        const bool one = (mask >> (table * k + bit)) & 1U;
        mass *= one ? r : 1 - r;
        hit = hit && one;
      }
      hits += hit;
    }
    if (hits >= 2) sum += mass;
    ++states;
  }
  return sum;
}

float execute(float score, float qnorm, float knorm, int k, int l,
              float scale = 2) {
  float s[] = {314159.25f, score, 314159.25f};
  float norm[] = {314159.25f, knorm, 314159.25f};
  int ids[] = {1};
  transform_kernel(s + 1, 1, qnorm, norm, k, l, scale, ids);
  require(s[0] == 314159.25f && s[2] == 314159.25f, "score guards");
  require(norm[0] == 314159.25f && norm[1] == knorm &&
              norm[2] == 314159.25f && ids[0] == 1,
          "borrowed inputs unchanged");
  return s[1];
}

void probabilities() {
  for (int k = 1; k <= 3; ++k)
    for (int l = 2; l <= 4; ++l)
      for (double r : {.25, .5, .75})
        close(enumeration(r, k, l), inclusion(std::pow(r, k), l), 2e-13,
              "exhaustive bits versus binomial");
}

void corrections() {
  const double pi = std::acos(-1.);
  for (float c : {-1.f, -.9f, -.5f, 0.f, .5f, .9f, 1.f})
    for (int k : {1, 2, 3, 4})
      for (int l : {2, 3, 5, 8, 16}) {
        const double p = std::pow(1 - std::acos(double(c)) / pi, k);
        // Near zero inclusion, log amplifies FP32 subtraction error. This
        // bounded-fixture acceptance budget allows 6e-7 inclusion error plus
        // base scalar rounding; it is not an arbitrary-K/L accuracy proof.
        close(execute(c, 1, 1, k, l),
              c / 2. - std::log(inclusion(p, l) + 1e-4),
              2e-5 + 6e-7 / (inclusion(p, l) + 1e-4),
              "regularized correction");
        ++correction_cases;
      }

  // True inclusion 2^-32 is lost by source FP32 cancellation. Gate that
  // reproduced behavior separately, including its difference from FP64.
  const double tiny = inclusion(std::pow(.5, 16), 2);
  require(tiny > 0 && tiny < 1e-9, "tiny true inclusion");
  const float actual = execute(0, 1, 1, 16, 2);
  close(actual, -std::log(1e-4), 1e-6, "cancellation zero");
  require(std::abs(double(actual) + std::log(tiny + 1e-4)) > 1e-6,
          "cancellation differs FP64");
  ++correction_cases;
}

void domains() {
  // Characterize current unguarded input domains using valid arrays/indices.
  // Expected nonfinite classifications are assertions, not numerical passes.
  auto nan = [&](float s, float q, float n, int k, int l, float scale,
                 const char* label) {
    require(std::isnan(execute(s, q, n, k, l, scale)), label);
    ++domain_cases;
  };
  nan(std::nextafter(1.f, 2.f), 1, 1, 2, 3, 2, "acos above one");
  nan(std::nextafter(-1.f, -2.f), 1, 1, 2, 3, 2, "acos below minus one");
  nan(0, 0, 1, 2, 3, 2, "zero query norm");
  nan(0, 1, 0, 2, 3, 2, "zero key norm");
  nan(.5, 0, 1, 2, 3, 2, "nonzero score zero norm");
  nan(0, 1, 1, -1, 3, 2, "negative K");
  nan(0, 1, 1, 2, 3, 0, "zero scale zero score");
  require(std::isinf(execute(.5, 1, 1, 2, 3, 0)), "zero scale infinity");
  ++domain_cases;

  close(execute(0, 1, 1, 0, 3), -std::log(1.0001), 1e-6, "K zero accepted");
  ++domain_cases;
  for (int l : {0, 1}) {
    close(execute(0, 1, 1, 2, l), -std::log(1e-4), 1e-6,
          "L below two accepted");
    ++domain_cases;
  }
  close(execute(.5, 1, -1, 2, 3),
        .25 - std::log(inclusion(std::pow(1 - std::acos(-.5) /
                                              std::acos(-1.), 2), 3) + 1e-4),
        1e-5, "negative norm accepted");
  ++domain_cases;

  float score = 7;
  float norm = 1;
  int id = 0;
  transform_kernel(&score, 0, 1, &norm, 2, 3, 2, &id);
  require(score == 7, "zero nnz unchanged");
  ++domain_cases;
  transform_kernel(&score, -1, 1, &norm, 2, 3, 2, &id);
  require(score == 7, "negative nnz unchanged");
  ++domain_cases;
}

void estimator() {
  // Synthetic independent candidate table processes, not geometric LSH:
  // shared hyperplanes can correlate candidates. This reference example
  // does not execute a production attention estimator or Softmax.
  const int l = 3;
  const double p[2] = {.25, .5};
  const double a[2] = {1, 2};
  const double v[2] = {0, 1};
  const double inc[2] = {inclusion(p[0], l), inclusion(p[1], l)};
  double ht = 0;
  double reg = 0;
  double sn = 0;
  double nonempty = 0;
  double total = 0;
  for (unsigned mask = 0; mask < 64; ++mask) {
    double mass = 1;
    bool selected[2];
    for (int item = 0; item < 2; ++item) {
      int hits = 0;
      for (int table = 0; table < l; ++table) {
        const bool hit = (mask >> (item * l + table)) & 1U;
        hits += hit;
        mass *= hit ? p[item] : 1 - p[item];
      }
      selected[item] = hits >= 2;
    }
    total += mass;
    double numerator = 0;
    double denominator = 0;
    for (int item = 0; item < 2; ++item)
      if (selected[item]) {
        ht += mass * a[item] * v[item] / inc[item];
        reg += mass * a[item] * v[item] / (inc[item] + 1e-4);
        const double weight = a[item] / (inc[item] + 1e-4);
        numerator += weight * v[item];
        denominator += weight;
      }
    if (denominator > 0) {
      nonempty += mass;
      sn += mass * numerator / denominator;
    }
  }
  close(total, 1, 1e-14, "sampling mass");
  close(ht, 2, 1e-13, "unsmoothed HT unbiased");
  close(reg, 2 * inc[1] / (inc[1] + 1e-4), 1e-13,
        "regularized HT expectation");
  require(reg < ht, "regularizer bias");
  const double dense = 2. / 3.;
  const double conditional = sn / nonempty;
  require(std::abs(conditional - dense) > .01, "finite self normalized bias");
  std::cout << "self_normalized_conditional=" << conditional
            << " dense=" << dense << " nonempty_probability=" << nonempty << '\n';
}
} // namespace probability_fixture

int main() {
  using namespace probability_fixture;
  probabilities();
  corrections();
  domains();
  estimator();
  std::cout << std::setprecision(12)
            << "TRACE_TEST_ID: MAGICPIG-PROBABILITY-CPU\n"
            << "MAGICPIG_PROBABILITY_PASS correction_cases=" << correction_cases
            << " domain_cases=" << domain_cases << " enumerated_states=" << states
            << " checks=" << checks << " max_absolute_error=" << worst << '\n';
}
