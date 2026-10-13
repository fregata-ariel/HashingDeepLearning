// Executes maintained transform_kernel; expectations use independent bit events
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
int stable_probability_cases = 0;
double worst_log_error = 0;
double worst_probability_relative_error = 0;
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
        // Stable maintained correction: only bounded scalar/log and final
        // FP32 rounding remain in this deterministic fixture.
        close(execute(c, 1, 1, k, l),
              c / 2. - std::log(inclusion(p, l) + 1e-4),
              2e-6,
              "regularized correction");
        ++correction_cases;
      }

  // Deliberately replace the #57 known-defect expectation after maintained
  // Issue #68 repair. True inclusion 2^-32 must survive the correction.
  const double tiny = inclusion(std::pow(.5, 16), 2);
  require(tiny > 0 && tiny < 1e-9, "tiny true inclusion");
  const float actual = execute(0, 1, 1, 16, 2);
  close(actual, -std::log(tiny + 1e-4), 1e-6, "tiny corrected FP64 agreement");
  require(std::abs(double(actual) + std::log(1e-4)) > 1e-6,
          "tiny correction differs old cancelled zero");
  ++correction_cases;
}

// BEGIN DECIMAL PROBABILITY REFERENCE
struct DecimalReference { int k, l; long double log_inclusion; };
const DecimalReference decimal_reference[] = {
    {1, 2, -1.38629436111989061883446424291635313615100026872051050824136L},
    {1, 3, -0.693147180559945309417232121458176568075500134360255254120680L},
    {1, 8, -0.0357891078515852792753420982122404025613538688227060980084343L},
    {1, 16, -0.000259433063909800907710690381749510617159628484955052731230362L},
    {1, 128, -3.79096928140187721319917533254174231168376882521661552192100e-37L},
    {1, 512, -3.82612874000000000000000000000000000000000000000000000000000e-152L},
    {4, 2, -5.54517744447956247533785697166541254460400107488204203296544L},
    {4, 3, -4.48912477023024871278280050422974613038812116312237205332884L},
    {4, 8, -2.46474949566860292411776397497238744474685689021150708546849L},
    {4, 16, -1.33137648130046352956456639761407366782031799255292188410136L},
    {4, 128, -0.00246662806736450230505979117861870513615668438144933036867176L},
    {4, 512, -1.56679751053828431892374676253987752239848298675027899904451e-13L},
    {8, 2, -11.0903548889591249506757139433308250892080021497640840659309L},
    {8, 3, -9.99435016369809612660328425638559081355515795207409317871616L},
    {8, 8, -7.77378296823785439634664421259726890752968069224260654089686L},
    {8, 16, -6.33929132816242351286340054426856705367908084496568984298066L},
    {8, 128, -2.40898775462322203726383788378376037595933262115118580561703L},
    {8, 512, -0.519994464474288196496377067187700768388042763177176028493887L},
    {16, 2, -22.1807097779182499013514278866616501784160042995281681318618L},
    {16, 3, -21.0821076618279223705450212413870747015314073295761714972965L},
    {16, 8, -18.8485663030157089307153256009368199043891125238560326915060L},
    {16, 16, -17.3933604500480323738620714049436030214359902959222408097002L},
    {16, 128, -13.1789212510264948799406944875513055538879852626261903693465L},
    {16, 512, -10.4043490783886361075147665137137195219406983074308568837328L},
    {24, 2, -33.2710646668773748520271418299924752676240064492922521977926L},
    {24, 3, -32.1724524179456958003842627964645387494135690409172305241387L},
    {24, 8, -29.9388603951207518060065230717144873098849073686080263447932L},
    {24, 16, -28.4835734804053398533691932144963071168189677444215145502101L},
    {24, 128, -24.2679995038478180184058766302135679219842075663100782596965L},
    {24, 512, -21.4895378977481161534686152216868499602384659088092087818377L},
    {32, 2, -44.3614195558364998027028557733233003568320085990563362637235L},
    {32, 3, -43.2628072673236105404222369798115824075732433742630353472681L},
    {32, 8, -41.0292150465926184534056223568924905855691810676252319119879L},
    {32, 16, -39.5739278152275398157858628306095289986507574364798474976901L},
    {32, 128, -35.3583494055760107178780315457041138047938042617114999533228L},
    {32, 512, -32.5798726003156513507269628925093555729400163637723309646755L},
    {64, 2, -88.7228391116729996054057115466466007136640171981126725274470L},
    {64, 3, -87.6242268230048899140466063821402584901507614527360446258199L},
    {64, 8, -85.3906346014977956816827349947841687346788921279228170324163L},
    {64, 16, -83.9353473688909536116639716259499264011288553146967818258269L},
    {64, 128, -79.7197689418547364804085271346203675319649966900244935910459L},
    {64, 512, -76.9412920769897327143018299063530412257871982761632167386846L},
    {512, 2, -709.782712893383996843245692373172805709312137584901380219576L},
    {512, 3, -708.684100604715887151850447136250280004664647027078630767842L},
    {512, 8, -706.450508383208792919305875386813272843524052586599008522876L},
    {512, 16, -704.995221150601950848997991438649562660912545269729347283567L},
    {512, 128, -700.779642723565733713694858836707453905985227654813762931945L},
    {512, 512, -698.001165858700729933710373800625670864166298935565879333556L},
    {2048, 2, -2839.13085157353598737298276949269122283724855033960552087831L},
    {2048, 3, -2838.03223928486787768158752425576869713260105978178277142657L},
    {2048, 8, -2835.79864706336078344904295250633168997146046534130314918160L},
    {2048, 16, -2834.34335983075394137873506855816797978884895802443348794230L},
    {2048, 128, -2830.12778140371772424343193595622587103392164040951790359067L},
    {2048, 512, -2827.34930453885272046344745092014408799210271169027001999228L},
};
// END DECIMAL PROBABILITY REFERENCE

long double positive_binomial(long double p, int l) {
  long double sum = 0, choose = 1;
  for (int h = 0; h <= l; ++h) {
    if (h >= 2) sum += choose * std::pow(p, h) * std::pow(1 - p, l - h);
    if (h < l) choose *= static_cast<long double>(l - h) / (h + 1);
  }
  return sum;
}

void stable_probabilities() {
  auto log_check = [&](double cosine, int k, int l, long double expected) {
    double actual;
    magicpig_inclusion_log_probability(cosine, k, l, &actual);
    require(std::isfinite(actual) && actual <= 0, "finite nonpositive log inclusion");
    const double error = static_cast<double>(std::abs(static_cast<long double>(actual) - expected));
    worst_log_error = std::max(worst_log_error, error);
    require(error <= 2e-11 + 1e-15 * std::abs(static_cast<double>(expected)),
            "stable log inclusion versus independent positive mass");
    const long double probability = std::exp(expected);
    // Compare positive probabilities only when double can represent them
    // normally; otherwise finite log inclusion is the preserved contract.
    if (probability >= std::numeric_limits<double>::min()) {
      const double observed = std::exp(actual);
      const double relative = std::abs((observed - probability) / probability);
      worst_probability_relative_error = std::max(worst_probability_relative_error, relative);
      require(observed > 0 && observed <= 1 && relative <= 5e-12,
              "stable positive probability relative error");
    }
    ++stable_probability_cases;
  };
  for (const auto& row : decimal_reference) {
    log_check(0, row.k, row.l, row.log_inclusion);
    const double expected = -std::log(1e-4) -
        std::log1p(std::exp(row.log_inclusion - std::log(1e-4)));
    close(execute(0, 1, 1, row.k, row.l), expected, 1e-6,
          "Decimal-reference regularized correction");
  }
  // Independent direct positive masses in extended precision across both
  // production algorithm branches and geometrically asymmetric cosines.
  const long double pi = std::acos(-1.0L);
  for (float cosine : {std::nextafter(-1.f, 0.f), -.9f, -.5f, 0.f,
                       .5f, .9f, std::nextafter(1.f, 0.f)})
    for (int k : {1, 2, 4, 8, 16, 32})
      for (int l : {2, 3, 8, 16, 128, 512}) {
        const long double collision = 1 - std::acos(static_cast<long double>(cosine)) / pi;
        const long double probability = positive_binomial(std::pow(collision, k), l);
        require(probability > 0 && probability <= 1 + 1e-15L, "positive reference mass");
        log_check(cosine, k, l, std::log(probability));
        close(execute(cosine, 1, 1, k, l), cosine / 2. - std::log(probability + 1e-4L),
              2e-6, "extended-precision stable correction");
      }
  // Cross the L*p=0.5 algorithm boundary from both sides. For L=2 the
  // independent exact inclusion is p^2 throughout the transition.
  for (float cosine : {-.0001f, .0001f}) {
    const long double collision = 1 - std::acos(static_cast<long double>(cosine)) / pi;
    log_check(cosine, 2, 2, 4 * std::log(collision));
    close(execute(cosine, 1, 1, 2, 2),
          cosine / 2. - std::log(std::pow(collision, 4) + 1e-4L),
          2e-6, "stable branch transition correction");
  }
  // L=2 has exactly p^2, including values far below exponent range. No
  // arbitrary positive-domain table/count cap or unsafe allocation is needed.
  log_check(0, std::numeric_limits<int>::max(), 2,
            -2.L * std::numeric_limits<int>::max() * std::log(2.L));
  double log_probability = 7;
  magicpig_inclusion_log_probability(-1, 2, 3, &log_probability);
  require(std::isinf(log_probability) && log_probability < 0, "true zero endpoint log");
  magicpig_inclusion_log_probability(1, 2, 3, &log_probability);
  require(log_probability == 0, "unit endpoint log");
  magicpig_inclusion_log_probability(0, 1, std::numeric_limits<int>::max(), &log_probability);
  require(log_probability == 0, "large count saturated inclusion");
  stable_probability_cases += 3;
}

void domains() {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto reject = [&](float s, float q, float n, int k, int l, float scale,
                    int count, const char* label) {
    float scores[] = {314159.25f, s, .25f, 314159.25f};
    float norms[] = {1.f, n, 314159.25f};
    const int ids[] = {1, 0};
    float before[4], norm_before[3];
    std::memcpy(before, scores, sizeof scores);
    std::memcpy(norm_before, norms, sizeof norms);
    bool threw = false;
    try { transform_kernel(scores + 1, count, q, norms, k, l, scale, ids); }
    catch (const std::invalid_argument&) { threw = true; }
    require(threw, label);
    require(std::memcmp(before, scores, sizeof scores) == 0, "invalid scores unchanged");
    require(std::memcmp(norm_before, norms, sizeof norms) == 0, "invalid norms unchanged");
    ++domain_cases;
  };
  for (float s : {nan, inf, -inf}) reject(s, 1, 1, 2, 3, 2, 1, "nonfinite score rejected");
  for (float q : {0.f, -1.f, nan, inf, -inf}) reject(0, q, 1, 2, 3, 2, 1, "invalid query norm");
  for (float n : {0.f, -1.f, nan, inf, -inf}) reject(0, 1, n, 2, 3, 2, 1, "invalid key norm");
  for (float scale : {0.f, -1.f, nan, inf, -inf}) reject(0, 1, 1, 2, 3, scale, 1, "invalid scaling");
  for (int k : {-1, 0}) reject(0, 1, 1, k, 3, 2, 1, "invalid K");
  for (int l : {-1, 0, 1}) reject(0, 1, 1, 2, l, 2, 1, "invalid L");
  reject(0, 1, 1, 2, 3, 2, -1, "negative count");
  const float tolerance = 8 * std::numeric_limits<float>::epsilon();
  for (float s : {1.f + 2 * tolerance, -1.f - 2 * tolerance, 2.f, -2.f})
    reject(s, 1, 1, 2, 3, 2, 1, "large cosine overshoot rejected");
  for (float s : {std::nextafter(1.f, 2.f), 1.f + tolerance,
                  std::nextafter(-1.f, -2.f), -1.f - tolerance}) {
    const double expected = double(s) / 2 - std::log(s > 0 ? 1.0001 : 1e-4);
    close(execute(s, 1, 1, 2, 3), expected, 2e-6, "bounded cosine overshoot clamped");
    ++domain_cases;
  }
  // Finite FP32 norm products may underflow or overflow in FP32. The
  // maintained double normalization accepts both without NaN propagation.
  for (float n : {std::numeric_limits<float>::denorm_min(),
                  std::numeric_limits<float>::max()}) {
    close(execute(0, n, n, 2, 3), -std::log(inclusion(.25, 3) + 1e-4),
          2e-6, "finite extreme norm product");
    ++domain_cases;
  }
  // Empty valid input needs no buffers; bad scalar parameters still reject.
  transform_kernel(nullptr, 0, 1, nullptr, 2, 3, 2, nullptr);
  ++domain_cases;
  reject(0, 0, 1, 2, 3, 2, 0, "empty invalid query norm");
  {
    float scores[] = {.25f, .5f};
    float norms[] = {1.f, 0.f};
    int ids[] = {0, 1};
    bool threw = false;
    try { transform_kernel(scores, 2, 1, norms, 2, 3, 2, ids); }
    catch (const std::invalid_argument&) { threw = true; }
    require(threw && scores[0] == .25f && scores[1] == .5f,
            "later invalid entry prevents partial mutation");
    ++domain_cases;
    ids[1] = -1;
    threw = false;
    try { transform_kernel(scores, 2, 1, norms, 2, 3, 2, ids); }
    catch (const std::invalid_argument&) { threw = true; }
    require(threw && scores[0] == .25f && scores[1] == .5f,
            "negative index rejected before access or mutation");
    ++domain_cases;
  }
  {
    float score = std::numeric_limits<float>::max();
    float norm = score;
    int id = 0;
    bool threw = false;
    try { transform_kernel(&score, 1, norm, &norm, 2, 3,
                           std::numeric_limits<float>::denorm_min(), &id); }
    catch (const std::overflow_error&) { threw = true; }
    require(threw && score == std::numeric_limits<float>::max(),
            "unrepresentable corrected score rejected unchanged");
    ++domain_cases;
  }
  {
    float score = .25f, norm = 1;
    int id = 0;
    for (int missing = 0; missing < 3; ++missing) {
      bool threw = false;
      try { transform_kernel(missing == 0 ? nullptr : &score, 1, 1,
                             missing == 1 ? nullptr : &norm, 2, 3, 2,
                             missing == 2 ? nullptr : &id); }
      catch (const std::invalid_argument&) { threw = true; }
      require(threw && score == .25f, "nonempty null buffer rejected");
      ++domain_cases;
    }
  }
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
  stable_probabilities();
  estimator();
  std::cout << std::setprecision(12)
            << "TRACE_TEST_ID: MAGICPIG-PROBABILITY-CPU\n"
            << "TRACE_TEST_ID: MAGICPIG-PROBABILITY-DOMAIN-CPU\n"
            << "TRACE_TEST_ID: MAGICPIG-PROBABILITY-STABILITY-CPU\n"
            << "MAGICPIG_PROBABILITY_PASS correction_cases=" << correction_cases
            << " domain_cases=" << domain_cases << " enumerated_states=" << states
            << " stable_probability_cases=" << stable_probability_cases
            << " checks=" << checks << " max_absolute_error=" << worst
            << " max_log_probability_error=" << worst_log_error
            << " max_probability_relative_error=" << worst_probability_relative_error << '\n';
}
