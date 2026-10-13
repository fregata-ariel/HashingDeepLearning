// TRACE_TEST_ID: MAGICPIG-SOFTMAX-EMPTY-CPU
// TRACE_TEST_ID: MAGICPIG-SOFTMAX-BOUNDARY-CPU
// TRACE_TEST_ID: MAGICPIG-SOFTMAX-DEFECT-CPU
// Actual selected bodies precede this fixture; independent FP64 expectations
// use fixed float32 input literals and host exp/log, not the implementation.
#include <array>
#include <iomanip>
#include <memory>
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
    const double tolerance=polynomial ? 0.005+0.02*std::abs(expected) : 2e-6;
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
    approximation=polynomial ? 0.03 : 0;
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
    // The native polynomial is not an all-finite-domain exponential. Its
    // bit-constructed exponent must not be confused with host underflow.
    std::size_t extreme_cases=0;
    for (float extreme_shift : {-90.0f,-100.0f,-1000.0f}) {
        std::vector<float> extreme(32,0);
        for (int i=16;i<32;++i) extreme[i]=extreme_shift;
        const Oracle extreme_expected=oracle(extreme,1);
        softmax_kernel(extreme.data(),32,&maximum,&lse);
        bool mismatch=false;
        for (int i=0;i<32;++i)
            mismatch |= !std::isfinite(extreme[i]) || extreme[i]<0 ||
                        std::abs(double(extreme[i])-extreme_expected.p[i])>0.005;
#ifdef MAGIC_NATIVE_AVX512
        require(mismatch,"native extreme-domain defect disappeared; revise contract");
        std::cout<<"MAGICPIG-SOFTMAX-DEFECT polynomial-extreme shift="<<extreme_shift
                 <<" first_probability="<<extreme[0]
                 <<" shifted_probability="<<extreme[16]<<" base2_lse="<<lse<<'\n';
#else
        require(!mismatch,"host exponential underflow baseline failed");
        compare(extreme.data(),extreme_expected,maximum,lse,false);
#endif
        ++extreme_cases;
    }
#ifndef MAGIC_NATIVE_AVX512
    std::cout<<"MAGICPIG-SOFTMAX-DEFECT polynomial-extreme not executed in software mode\n";
#endif
    require(extreme_cases==3,"extreme fixture matrix incomplete");
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
