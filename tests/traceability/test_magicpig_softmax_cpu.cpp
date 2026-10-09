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
void compare(const std::vector<float>& actual,const Oracle& expected,float max2,float lse2,bool polynomial) {
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
void isolated_probe(const std::string& option) {
    float maximum=0, lse=0;
    if (option=="--probe-sparse-empty" || option=="--probe-full-empty") {
        std::cout<<"MAGICPIG-SOFTMAX-PROBE "<<option.substr(8)<<std::endl;
        if (option=="--probe-sparse-empty") softmax_kernel(nullptr,0,&maximum,&lse);
        else softmax_kernel_full(nullptr,0,2,&maximum,&lse);
    } else if (option=="--probe-full-tail" || option=="--probe-optimized-tail") {
        // Exactly one element: rounded-vector reads have no legal padding.
        std::unique_ptr<float[]> scores(new float[1]); scores[0]=0;
        std::cout<<"MAGICPIG-SOFTMAX-PROBE "<<option.substr(8)<<std::endl;
        if (option=="--probe-full-tail") softmax_kernel_full(scores.get(),1,2,&maximum,&lse);
        else softmax_kernel_optimized(scores.get(),1,2,&maximum,&lse);
    } else throw std::runtime_error("unknown isolated probe");
    throw std::runtime_error("unsafe source call unexpectedly returned");
}
int run() {
    const std::array<int,8> lengths{1,2,15,16,17,31,32,33};
    const std::array<float,3> shifts{0,-32,-128};
    const std::array<float,3> scales{1,2,4};
    std::size_t sparse_cases=0,full_cases=0,optimized_cases=0,tail_defects=0;
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
        compare(sparse,expected,maximum,lse,true);
        require(sparse[n]==sentinel && sparse[n+1]==sentinel,"sparse wrote outside logical length");
        ++sparse_cases;
        // Full actually rounds up its scaling/normalizing writes. Supply legal
        // allocation padding, then assert that the unexpected writes occurred.
        const int rounded=(n+15)&~15;
        std::vector<float> full(rounded+2,sentinel);
        std::copy(raw.begin(),raw.end(),full.begin());
        softmax_kernel_full(full.data(),n,scale,&maximum,&lse);
        compare(full,expected,maximum,lse,false);
        require(full[rounded]==sentinel && full[rounded+1]==sentinel,"full crossed rounded allocation");
        if (n!=rounded) {
            // For n=1/scale=1 the two invalid writes can preserve the bits.
            if (scale!=1 || n!=1) require(full[n]!=sentinel,"full-tail defect disappeared; revise contract");
            ++tail_defects;
        }
        ++full_cases;
        // Optimized is supported only for complete tiles. Partial tiles below
        // are a separate deterministic defect characterization whose padding
        // carries the same common shift, keeping relative logits moderate.
        std::vector<float> optimized(rounded+2,shift);
        optimized[rounded]=optimized[rounded+1]=sentinel;
        std::copy(raw.begin(),raw.end(),optimized.begin());
        softmax_kernel_optimized(optimized.data(),n,scale,&maximum,&lse);
        require(optimized[rounded]==sentinel && optimized[rounded+1]==sentinel,"optimized crossed rounded allocation");
        if (n==rounded) { compare(optimized,expected,maximum,lse,true); ++optimized_cases; }
        else {
            double actual_sum=0;
            for (int i=0;i<n;++i) actual_sum+=optimized[i];
            require(std::isfinite(actual_sum) && actual_sum<0.999,"optimized-tail defect disappeared; revise contract");
            require(optimized[n]>0 && std::isfinite(optimized[n]),"optimized did not include invalid padded lane");
            ++tail_defects;
        }
        for (int i=0;i<n;++i) largest_error=std::max(largest_error,std::abs(double(sparse[i])-expected.p[i]));
    }
    // Known literal expectations independently pin the log convention.
    std::vector<float> equal(16,0); float maximum=0,lse=0;
    softmax_kernel(equal.data(),16,&maximum,&lse);
    for (float x:equal) require(x==0.0625f,"equal logits must have probability 1/16");
    require(maximum==0 && lse==4,"base-2 LSE of sixteen unit exponents is four");
    // Empty optimized input does not dereference score, but returns -Inf
    // metadata with no status. This is characterization, not a valid softmax.
    std::vector<float> empty_guard(2,sentinel);
    softmax_kernel_optimized(empty_guard.data(),0,2,&maximum,&lse);
    require(std::isinf(maximum) && maximum<0 && std::isinf(lse) && lse<0,"optimized empty-domain outcome changed");
    require(empty_guard[0]==sentinel && empty_guard[1]==sentinel,"optimized empty touched scores");
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
        compare(extreme,extreme_expected,maximum,lse,false);
#endif
        ++extreme_cases;
    }
#ifndef MAGIC_NATIVE_AVX512
    std::cout<<"MAGICPIG-SOFTMAX-DEFECT polynomial-extreme not executed in software mode\n";
#endif
    require(extreme_cases==3,"extreme fixture matrix incomplete");
    require(sparse_cases==72 && full_cases==72 && optimized_cases==18 && tail_defects==108,"coverage matrix incomplete");
    std::cout<<std::setprecision(12)<<"MAGICPIG-SOFTMAX-BOUNDARY-CPU sparse="<<sparse_cases<<" full_padded="<<full_cases<<" optimized_full_tile="<<optimized_cases<<" expected_tail_defects="<<tail_defects<<" max_probability_error="<<largest_error<<"\n";
    return 0;
}
} // namespace magicpig_softmax_test
int main(int argc,char** argv) {
    try {
        if (argc==2) magicpig_softmax_test::isolated_probe(argv[1]);
        else if (argc!=1) throw std::runtime_error("expected zero or one probe option");
        return magicpig_softmax_test::run();
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
