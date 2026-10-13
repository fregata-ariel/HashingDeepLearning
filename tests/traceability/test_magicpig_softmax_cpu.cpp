// TRACE_TEST_ID: MAGICPIG-SOFTMAX-EMPTY-CPU
// TRACE_TEST_ID: MAGICPIG-SOFTMAX-BOUNDARY-CPU
// TRACE_TEST_ID: MAGICPIG-SOFTMAX-EXTREME-CPU
// TRACE_TEST_ID: MAGICPIG-EXP-NATIVE-CPU
// Actual selected bodies precede this fixture; independent FP64 expectations
// use fixed float32 input literals and host exp/log, not the implementation.
#include <array>
#include <iomanip>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>

namespace magicpig_softmax_test {
constexpr float sentinel=123.25f;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
struct Oracle { std::vector<double> p; double max2, lse2; };
Oracle oracle(const std::vector<float>& raw, double scale) {
    std::vector<double> scaled;
    for (float x:raw) scaled.push_back(double(x)/scale);
    const double maximum=*std::max_element(scaled.begin(),scaled.end());
    double total=0;
    for (double x:scaled) total+=std::exp(x-maximum);
    Oracle out{{},maximum/std::log(2.0),(maximum+std::log(total))/std::log(2.0)};
    for (double x:scaled) out.p.push_back(std::exp(x-maximum)/total);
    return out;
}
bool close_prob(float actual,double expected,bool polynomial) {
#ifdef MAGIC_NATIVE_AVX512
    const double tolerance=polynomial ? 2e-6+2e-5*std::abs(expected) : 2e-6;
    return std::isfinite(actual) && actual>=0 && std::abs(actual-expected)<=tolerance;
#else
    (void)polynomial;
    return std::isfinite(actual) && actual>=0 && std::abs(actual-expected)<=2e-6;
#endif
}
void compare(const float* actual,const Oracle& expected,float max2,float lse2,bool polynomial) {
    double sum=0;
    for (std::size_t i=0;i<expected.p.size();++i) {
        require(close_prob(actual[i],expected.p[i],polynomial),"probability differs from dense FP64 oracle");
        sum+=actual[i];
    }
    require(std::abs(sum-1)<2e-5,"normalization sum differs from one");
    const double float_roundoff=2e-6*(1+std::abs(expected.max2));
    double approximation=0;
#ifdef MAGIC_NATIVE_AVX512
    approximation=polynomial ? 5e-5 : 0;
#else
    (void)polynomial;
#endif
    require(std::isfinite(max2) && std::abs(max2-expected.max2)<=float_roundoff,"base-2 maximum differs");
    require(std::isfinite(lse2) && std::abs(lse2-expected.lse2)<=float_roundoff+approximation,"base-2 log-sum-exp differs");
}
int run() {
    const std::array<int,8> lengths{1,2,15,16,17,31,32,33};
    const std::array<float,3> shifts{0,-32,-128};
    const std::array<float,3> scales{1,2,4};
    std::size_t sparse_cases=0,full_cases=0,optimized_cases=0,exact_cases=0;
    double largest_error=0;
    for (int n:lengths) for (float shift:shifts) for (float scale:scales) {
        std::vector<float> raw(n);
        for (int i=0;i<n;++i) raw[i]=float((i*7)%13-6)*0.25f+shift;
        const Oracle expected=oracle(raw,scale);
        // Sparse takes already-scaled scores; full and optimized take raw dots.
        std::vector<float> sparse(n+2,sentinel);
        for (int i=0;i<n;++i) sparse[i]=raw[i]/scale;
        float maximum=0,lse=0;
        softmax_kernel(sparse.data(),n,&maximum,&lse);
        compare(sparse.data(),expected,maximum,lse,true);
        require(sparse[n]==sentinel && sparse[n+1]==sentinel,"sparse wrote outside logical length");
        ++sparse_cases;
        // Guard lanes detect logical writes even in ordinary builds.
        std::vector<float> full(n+2,sentinel),optimized(n+2,sentinel);
        std::copy(raw.begin(),raw.end(),full.begin());
        std::copy(raw.begin(),raw.end(),optimized.begin());
        softmax_kernel_full(full.data(),n,scale,&maximum,&lse);
        compare(full.data(),expected,maximum,lse,false);
        require(full[n]==sentinel && full[n+1]==sentinel,"full wrote beyond logical length");
        ++full_cases;
        softmax_kernel_optimized(optimized.data(),n,scale,&maximum,&lse);
        compare(optimized.data(),expected,maximum,lse,true);
        require(optimized[n]==sentinel && optimized[n+1]==sentinel,"optimized wrote beyond logical length");
        ++optimized_cases;
        // Exactly nnz floats: sanitizer redzones reject any rounded tail read
        // or write; independent oracle checks every valid probability and LSE.
        for (int variant=0;variant<3;++variant) {
            std::unique_ptr<float[]> exact(new float[n]);
            for (int i=0;i<n;++i) exact[i]=variant==0 ? raw[i]/scale : raw[i];
            if (variant==0) softmax_kernel(exact.get(),n,&maximum,&lse);
            else if (variant==1) softmax_kernel_full(exact.get(),n,scale,&maximum,&lse);
            else softmax_kernel_optimized(exact.get(),n,scale,&maximum,&lse);
            compare(exact.get(),expected,maximum,lse,variant!=1);
            ++exact_cases;
        }
        for (int i=0;i<n;++i) largest_error=std::max(largest_error,std::abs(double(sparse[i])-expected.p[i]));
    }
    // Known literal expectations independently pin the log convention.
    std::vector<float> equal(16,0); float maximum=0,lse=0;
    softmax_kernel(equal.data(),16,&maximum,&lse);
    for (float x:equal) require(x==0.0625f,"equal logits must have probability 1/16");
    require(maximum==0 && lse==4,"base-2 LSE of sixteen unit exponents is four");
    // Zero candidates define zero mass consistently, even with a null score.
    std::size_t empty_cases=0;
    for (int variant=0;variant<3;++variant) for (bool null_score:{false,true}) {
        std::vector<float> guard(2,sentinel);
        float* scores=null_score ? nullptr : guard.data();
        maximum=sentinel; lse=sentinel;
        if (variant==0) softmax_kernel(scores,0,&maximum,&lse);
        else if (variant==1) softmax_kernel_full(scores,0,2,&maximum,&lse);
        else softmax_kernel_optimized(scores,0,2,&maximum,&lse);
        require(std::isinf(maximum) && maximum<0 && std::isinf(lse) && lse<0,
                "empty support must report zero-mass -Inf metadata");
        require(guard[0]==sentinel && guard[1]==sentinel,"empty support touched scores");
        ++empty_cases;
    }
    require(empty_cases==6,"empty support matrix incomplete");
    std::cout<<"MAGICPIG-SOFTMAX-EMPTY-CPU cases="<<empty_cases<<'\n';
    // Extreme finite logits pin underflow repair, both sparse and optimized.
    // Shifted subtraction can overflow to -Inf even for finite raw logits.
    std::size_t extreme_cases=0;
    for (float extreme_shift : {-87.0f,-90.0f,-100.0f,-104.0f,-1000.0f,
                                 -std::numeric_limits<float>::max()}) {
        std::vector<float> raw(32,0);
        for (int i=16;i<32;++i) raw[i]=extreme_shift;
        const Oracle expected_extreme=oracle(raw,1);
        for (int variant=0;variant<3;++variant) {
            auto extreme=raw;
            if (variant==0) softmax_kernel(extreme.data(),32,&maximum,&lse);
            else if (variant==1) softmax_kernel_full(extreme.data(),32,1,&maximum,&lse);
            else softmax_kernel_optimized(extreme.data(),32,1,&maximum,&lse);
            compare(extreme.data(),expected_extreme,maximum,lse,variant!=1);
            ++extreme_cases;
        }
    }
    // A common large positive offset checks finite base-2 metadata; the lower
    // logits are still real candidates and must not acquire wrapped exponents.
    for (int variant=0;variant<3;++variant) {
        std::vector<float> extreme(32,10000);
        for (int i=16;i<32;++i) extreme[i]=9000;
        const Oracle expected_extreme=oracle(extreme,1);
        if (variant==0) softmax_kernel(extreme.data(),32,&maximum,&lse);
        else if (variant==1) softmax_kernel_full(extreme.data(),32,1,&maximum,&lse);
        else softmax_kernel_optimized(extreme.data(),32,1,&maximum,&lse);
        compare(extreme.data(),expected_extreme,maximum,lse,variant!=1);
        ++extreme_cases;
    }
    // Finite logits can produce an infinite negative shifted difference in
    // FP32; base-2 metadata remains representable for this positive maximum.
    for (int variant=0;variant<3;++variant) {
        std::vector<float> extreme(32,1e38f);
        for (int i=16;i<32;++i) extreme[i]=-std::numeric_limits<float>::max();
        const Oracle expected_extreme=oracle(extreme,1);
        if (variant==0) softmax_kernel(extreme.data(),32,&maximum,&lse);
        else if (variant==1) softmax_kernel_full(extreme.data(),32,1,&maximum,&lse);
        else softmax_kernel_optimized(extreme.data(),32,1,&maximum,&lse);
        compare(extreme.data(),expected_extreme,maximum,lse,variant!=1);
        ++extreme_cases;
    }
    require(extreme_cases==24,"extreme fixture matrix incomplete");
    std::cout<<"MAGICPIG-SOFTMAX-EXTREME-CPU cases="<<extreme_cases<<'\n';
#ifdef MAGIC_NATIVE_AVX512
    // This directly calls the extracted production exp with actual intrinsics,
    // and is never reported as executed by the software-host-exp suite.
    std::vector<float> arguments;
    for (int i=0;i<=2800;++i) arguments.push_back(-87.0f+float(i)/16);
    const std::array<float,20> edges{
        -INFINITY, INFINITY, std::numeric_limits<float>::quiet_NaN(),
        -std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),
        -1000,-104,-103.9f,-100,-90,-87.001f,-87,
        88,88.001f,88.7f,88.72f,88.73f,100,1000,-0.0f};
    arguments.insert(arguments.end(),edges.begin(),edges.end());
    double largest_relative=0; std::size_t direct_cases=0;
    for (std::size_t base=0;base<arguments.size();base+=16) {
        float inputs[16]{},actual[16];
        const std::size_t lanes=std::min(std::size_t(16),arguments.size()-base);
        for (std::size_t j=0;j<lanes;++j) inputs[j]=arguments[base+j];
        _mm512_storeu_ps(actual,avx512_exp_ps(_mm512_loadu_ps(inputs)));
        for (std::size_t j=0;j<lanes;++j) {
            const double expected_value=std::exp(double(inputs[j]));
            const float rounded_expected=float(expected_value);
            if (std::isnan(expected_value)) require(std::isnan(actual[j]),"exp NaN policy");
            else if (std::isinf(rounded_expected))
                require(std::isinf(actual[j]) && actual[j]>0,"exp overflow/+Inf policy");
            else if (rounded_expected==0) require(actual[j]==0,"exp underflow/-Inf policy");
            else {
                const double error=std::abs(double(actual[j])-expected_value);
                const double bound=2e-5*expected_value+2*double(std::numeric_limits<float>::denorm_min());
                require(std::isfinite(actual[j]) && actual[j]>=0 && error<=bound,
                        "native exp differs from FP64 bound");
                if (expected_value>=std::numeric_limits<float>::min())
                    largest_relative=std::max(largest_relative,error/expected_value);
            }
            ++direct_cases;
        }
    }
    require(direct_cases==2821,"direct native exp matrix incomplete");
    std::cout<<"MAGICPIG-EXP-NATIVE-CPU cases="<<direct_cases
             <<" max_relative_error="<<largest_relative<<'\n';
#else
    std::cout<<"MAGICPIG-EXP-NATIVE-SKIP portable mode uses host exp\n";
#endif
    require(sparse_cases==72 && full_cases==72 && optimized_cases==72 && exact_cases==216,"coverage matrix incomplete");
    std::cout<<std::setprecision(12)<<"MAGICPIG-SOFTMAX-BOUNDARY-CPU sparse="<<sparse_cases<<" full="<<full_cases<<" optimized="<<optimized_cases<<" exact_capacity="<<exact_cases<<" max_probability_error="<<largest_error<<"\n";
    return 0;
}
} // namespace magicpig_softmax_test
int main(int argc,char** argv) {
    try {
        (void)argv;
        if (argc!=1) throw std::runtime_error("expected no probe options");
        return magicpig_softmax_test::run();
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
