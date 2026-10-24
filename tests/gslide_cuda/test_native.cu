#include "Network.h"
#include "kernel.h"
#include "lshKnl.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

__global__ void original_get_dense_acts_knl(CscActNodes, int, int);
__global__ void original_init_hash_knl(const int*, const float*, int, int, int,
                                      int, int, int, int, int, int, int, int*, int*);
namespace {
int checks = 0;
double max_error = 0;
void require(bool ok, const char *message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
void near(double actual, double expected, double tolerance = 8e-6) {
  max_error = std::max(max_error, std::abs(actual - expected));
  require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          "numerical oracle mismatch");
}
void complete() {
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
template<class T> struct Buffer {
  T *p;
  int n;
  explicit Buffer(int n): n(n) {
    CUDA_CHECK(cudaMallocManaged(&p, sizeof(T) * n));
    std::fill_n(p, n, T{});
  }
  ~Buffer() { CUDA_CHECK(cudaFree(p)); }
  Buffer(const Buffer&) = delete;
};
std::vector<float> read(const float *p, int n) {
  std::vector<float> result(n);
  CUDA_CHECK(cudaMemcpy(result.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}
void write(float *p, const std::vector<float>& v) {
  CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
}
void check_values(const float *p, const std::vector<double>& expected) {
  auto actual = read(p, expected.size());
  for (size_t i=0; i<expected.size(); ++i) near(actual[i], expected[i]);
}

// TRACE_TEST_ID: GSLIDE-DENSE-CUDA-124
void dense() {
  // Exact archived body, legal padded launch: independently assert the defect.
  CscActNodes original(2, 2, false, true);
  std::fill_n(original.d_offsets, 3, -71);
  original_get_dense_acts_knl<<<1,32>>>(original,1,2);
  complete();
  require(original.d_nodes[0]==0 && original.d_nodes[1]==0 &&
          original.d_offsets[0]==0 && original.d_offsets[1]==1 &&
          original.d_offsets[2]==-71, "original dense negative control changed");
  std::cout << "ORIGINAL_DENSE terminal=-71 independent_expected=2\n";
  original.free();
  for (int nodes: {0,1,3,33}) for (int batch: {0,1,2,33})
    for (int threads: {32,64,128}) {
      int size=nodes*batch;
      CscActNodes acts(batch+1, size+2, false, true);
      std::fill_n(acts.d_nodes,size+2,-91);
      std::fill_n(acts.d_offsets,batch+2,-71);
      // Keep one guard on each side of the borrowed ID span.
      int *base=acts.d_nodes;
      acts.d_nodes=base+1; acts.node_capacity=size;
      get_dense_acts(acts,nodes,batch,threads);
      complete();
      for (int i=0;i<size;++i) require(acts.d_nodes[i]==i%nodes,"dense IDs");
      for (int i=0;i<=batch;++i) require(acts.d_offsets[i]==i*nodes,"dense offset");
      require(base[0]==-91 && base[size+1]==-91 && acts.d_offsets[batch+1]==-71,
              "dense allocation guard");
      if (size>0) {
        acts.node_capacity=size-1;
        bool rejected=false;
        try { get_dense_acts(acts,nodes,batch,threads); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"dense capacity must reject");
      }
      acts.d_nodes=base;acts.free();
    }
  CscActNodes invalid(2,2,false,true);
  for(const auto& args: std::vector<std::vector<int>>{
      {-1,1,32},{1,-1,32},{1,3,32},{1,1,0},{1,1,31},{1,1,1056}}) {
    bool rejected=false;
    try {get_dense_acts(invalid,args[0],args[1],args[2]);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"invalid dense API input must reject before launch");
  }
  invalid.free();
  // Both production ReLU caller overloads, especially the nodes=1 terminal.
  Layer input(1,1,2,2);
  input.csc_acts.extract_from({0,0},{1.f,2.f},{0,1,2});
  ReluLayer output(1,1,2,2);
  write(output.d_weights,{2.f});write(output.d_biases,{.5f});
  for (bool use_layer: {false,true}) {
    if (use_layer) output.forward(input,2,32,1);
    else output.forward(input.csc_acts,2,32,1);
    complete();
    std::vector<int> ids,offsets;std::vector<float> vals;
    output.csc_acts.extract_to(ids,vals,offsets,2);
    require(offsets==std::vector<int>({0,1,2}),"ReLU caller terminal");
    near(vals[0],2.5);near(vals[1],4.5);
  }
  output.forward(input.csc_acts,0,32,1);complete();
  std::cout << "DENSE_PASS cases=48 production_overloads=2 empty_batch=true\n";
}

// Independent scalar WTA. This fixture uses K=1/bin=2, so packing is exact;
// no claim is made about the released natural-log K>1 variant.
int winner(const float *weights, int node, int mode) {
  float a=mode==2 ? -1.f : (node%2 ? 1.f : 2.f);
  float b=mode==2 ? -1.f : (node%2 ? 2.f : 1.f);
  if (mode==1) {a=-a;b=-b;}
  (void)weights;
  return b>a ? 1 : 0;
}
// TRACE_TEST_ID: GSLIDE-HASH-CUDA-125
void hash() {
  Buffer<int> bins(8),buckets(8*66+2),sizes(8+2);
  Buffer<float> weights(65*2);
  for(int i=0;i<8;++i)bins.p[i]=i%2;
  for(int n=0;n<64;++n){weights.p[2*n]=n%2?1:2;weights.p[2*n+1]=n%2?2:1;}
  std::fill_n(buckets.p,buckets.n,-1);
  original_init_hash_knl<<<2,32,(4+32*2)*sizeof(int)>>>(
      bins.p,weights.p,2,64,4,2,1,2,2,1,4,64,buckets.p,sizes.p);
  complete();
  require(sizes.p[0]==32 && sizes.p[5]==32 && sizes.p[1]==0 && sizes.p[4]==0,
          "original hash count negative control");
  for(int i=0;i<8*64;++i) require(buckets.p[i]<32,"original missing node control");
  std::cout << "ORIGINAL_HASH nodes_per_table=32 independent_expected=64 nodes32to63_absent=true\n";
  int cases=0;
  for(int lanes: {1,2,4})for(int threads: {32,64,128})
    for(int nodes: {1,31,33,64,65})for(int mode: {0,1,2}) {
      const int tables=lanes+1, tile=lanes, capacity=nodes, count=tables*2;
      Buffer<int> payload(count*capacity+2), counts(count+2), permutation(tile*2+2);
      for(int i=0;i<permutation.n;++i)permutation.p[i]=i%2;
      for(int n=0;n<nodes;++n) {
        float a=n%2?1:2,b=n%2?2:1;
        if(mode==1){a=-a;b=-b;} if(mode==2){a=b=-1;}
        weights.p[2*n]=a;weights.p[2*n+1]=b;
      }
      for(int rebuild=0;rebuild<2;++rebuild) {
        std::fill_n(payload.p,payload.n,-71);
        std::fill_n(counts.p,counts.n,0);counts.p[0]=counts.p[count+1]=-91;
        int blocks=(nodes*lanes+threads-1)/threads;
        size_t shared=sizeof(int)*(2*tile+(threads/lanes)*2);
        init_hash_knl<<<blocks,threads,shared>>>(permutation.p,weights.p,2,nodes,
            tables*2,tables,1,2,tile,1,2,capacity,payload.p+1,counts.p+1);
        complete();
        require(payload.p[0]==-71 && payload.p[payload.n-1]==-71 &&
                counts.p[0]==-91 && counts.p[count+1]==-91,"hash guard");
        for(int t=0;t<tables;++t)for(int address=0;address<2;++address) {
          std::set<int> expected;
          for(int n=0;n<nodes;++n)if(winner(weights.p,n,mode)==address)expected.insert(n);
          int bucket=t*2+address;
          require(counts.p[bucket+1]==int(expected.size()),"hash bucket count");
          std::set<int> actual(payload.p+1+bucket*capacity,
                               payload.p+1+bucket*capacity+counts.p[bucket+1]);
          require(actual==expected,"hash bucket membership");
        }
      }
      ++cases;
    }
  // Exercise actual LSH::build alternate dispatch and query, without shuffle.
  for(int mode: {0,1}) {
    for(int n=0;n<65;++n){weights.p[2*n]=mode?(n%2?2:1):(n%2?1:2);
                         weights.p[2*n+1]=3-weights.p[2*n];}
    LSH lsh(65,2,1,1,3,2,2,65,1,0,4,1,128,128);
    for(int rebuild=0;rebuild<2;++rebuild) {
      lsh.build(weights.p,false);complete();
      require(std::string(lsh.build_launch.kernel)=="init_hash_knl","production alternate branch");
      require(lsh.build_launch.blocks*lsh.build_launch.threads>=65*4,"production hash launch coverage");
      std::cout << "HASH_DISPATCH kernel=" << lsh.build_launch.kernel
                << " blocks=" << lsh.build_launch.blocks << " threads=" << lsh.build_launch.threads
                << " shared_bytes=" << lsh.build_launch.shared_bytes << '\n';
      CscActNodes query(1,2),active(1,65,false);
      query.extract_from({0,1},{2.f,1.f},{0,2});
      lsh.query_act_nodes(query,1,active);complete();
      std::vector<int> ids,offsets;active.extract_to(ids,offsets,1);
      std::set<int> expected;
      for(int n=0;n<65;++n)if(weights.p[2*n]>weights.p[2*n+1])expected.insert(n);
      require(std::set<int>(ids.begin(),ids.end())==expected,"production LSH query membership");
      query.free();active.free();
    }
  }
  std::cout << "HASH_PASS cases=" << cases << " rebuilds=2 production_build_query=true\n";
}

struct Params {
  std::vector<double> v,m,s;
  explicit Params(std::vector<float> input):v(input.begin(),input.end()),m(v.size()),s(v.size()){}
  void update(const std::vector<double>& g, float rate) {
    for(size_t i=0;i<v.size();++i){m[i]=.9*m[i]+.1*g[i];s[i]=.999*s[i]+.001*g[i]*g[i];
      v[i]+=rate*m[i]/(std::sqrt(s[i])+1e-8);}
  }
  void check(const float *value, const Layer::Adam& adam) const {
    check_values(value,v);check_values(adam.d_moms,m);check_values(adam.d_vels,s);
    check_values(adam.d_ts,std::vector<double>(v.size(),0));
  }
};
// TRACE_TEST_ID: GSLIDE-TRAINING-CUDA-43
void training() {
  constexpr int I=2,H=3,O=4,B=2;
  Network net({H,O},{8,8},I,B,4,2,1,3,2,2,O,1,O,2,1,16,32);
  Layer &hidden=net.get_layer(0),&output=net.get_layer(1);
  std::vector<float> hw{.5f,.2f,-.3f,-.4f,.6f,.1f},hb{.1f,-.05f,0.f};
  std::vector<float> ow{.2f,-.3f,.4f,-.1f,.5f,.2f,.3f,.1f,-.2f,-.4f,.2f,.3f},ob{0.f,.1f,-.1f,.05f};
  write(hidden.d_weights,hw);write(hidden.d_biases,hb);
  write(output.d_weights,ow);write(output.d_biases,ob);net.rebuild(false);
  Params wh(hw),bh(hb),wo(ow),bo(ob);
  for(int step=1;step<=2;++step) {
    std::vector<float> x=step==1?std::vector<float>{1,.5f,-.25f,1}:
                                std::vector<float>{-.5f,1,1,-.5f};
    std::vector<double> h(B*H),p(B*O),dh(B*H),gw(H*I),gb(H),go(O*H),gob(O);
    for(int b=0;b<B;++b) {
      for(int j=0;j<H;++j){double a=bh.v[j];for(int i=0;i<I;++i)a+=x[b*I+i]*wh.v[i*H+j];h[b*H+j]=std::max(0.,a);}
      double mx=-1e30;
      for(int o=0;o<O;++o){double a=bo.v[o];for(int j=0;j<H;++j)a+=h[b*H+j]*wo.v[o*H+j];p[b*O+o]=a;mx=std::max(mx,a);}
      double sum=0;for(int o=0;o<O;++o){p[b*O+o]=std::exp(p[b*O+o]-mx);sum+=p[b*O+o];}
      for(int o=0;o<O;++o){p[b*O+o]/=sum+1e-8;double d=((o==(b?3:0)?1.:0.)-p[b*O+o])/B;gob[o]+=d;
        for(int j=0;j<H;++j){go[o*H+j]+=h[b*H+j]*d;if(h[b*H+j]>0)dh[b*H+j]+=d*wo.v[o*H+j];}}
      for(int j=0;j<H;++j){gb[j]+=dh[b*H+j];for(int i=0;i<I;++i)gw[i*H+j]+=x[b*I+i]*dh[b*H+j];}
    }
    // Poison the complete allocation; production must clear active and spare entries.
    write(hidden.d_cmprs_bp_deltas,std::vector<float>(8,17.f));
    float rate=.01*std::sqrt(1-std::pow(.999,step))/(1-std::pow(.9,step));
    net.train({0,1,0,1},x,{0,2,4},{0,3},{0,1,2},{H,O},B,rate,1,64,true,false);
    complete();
    check_values(hidden.csc_acts.d_vals,h);
    auto hidden_delta=read(hidden.d_cmprs_bp_deltas,8);
    for(int i=0;i<B*H;++i)near(hidden_delta[i],dh[i]);
    near(hidden_delta[6],0,0);near(hidden_delta[7],0,0);
    std::vector<int> ids,offsets;std::vector<float> probs;
    output.csc_acts.extract_to(ids,probs,offsets,B);
    require(offsets==std::vector<int>({0,4,8}),"production output selection extent");
    for(int b=0;b<B;++b)for(int k=offsets[b];k<offsets[b+1];++k)near(probs[k],p[b*O+ids[k]]);
    wh.update(gw,rate);bh.update(gb,rate);wo.update(go,rate);bo.update(gob,rate);
    wh.check(hidden.d_weights,hidden.weight_adam);bh.check(hidden.d_biases,hidden.bias_adam);
    wo.check(output.d_weights,output.weight_adam);bo.check(output.d_biases,output.bias_adam);
  }
  require(read(hidden.d_weights,hw.size())!=hw,"connected parameters must change");
  // Native cuBLAS eval call, after training. It returns accuracy, not probabilities.
  int correct=net.eval({0,1,0,1},{1,.5f,-.25f,1},{0,2,4},{0,3},{0,1,2},B,64);
  std::vector<double> logits(B*O);
  std::vector<float> x{1,.5f,-.25f,1};
  int expected_correct=0;
  for(int b=0;b<B;++b) {
    std::vector<double> h(H);
    for(int j=0;j<H;++j){double a=bh.v[j];for(int i=0;i<I;++i)a+=x[b*I+i]*wh.v[i*H+j];h[j]=std::max(0.,a);}
    // Released dense eval omits biases. Validate that exact public behavior.
    for(int o=0;o<O;++o)for(int j=0;j<H;++j)logits[b*O+o]+=h[j]*wo.v[o*H+j];
    int best=std::max_element(logits.begin()+b*O,logits.begin()+(b+1)*O)-(logits.begin()+b*O);
    expected_correct+=best==(b?3:0);
  }
  check_values(static_cast<SoftmaxLayer&>(output).d_dense_activations,logits);
  require(correct==expected_correct,"production cuBLAS eval argmax");
  // Exact ties must choose the first ID, as the released max_element did.
  write(output.d_weights,std::vector<float>(O*H,0));
  require(net.eval({0,1,0,1},x,{0,2,4},{0,0},{0,1,2},B,64)==B,
          "eval argmax tie must choose node zero");
  std::cout << "TRAINING_PASS native_Network_train=true steps=2 full_capacity_reset=true relu_gates_change=true adam_oracle=true cublas_eval=true\n";
}

void stale() {
  CscActNodes active(1,2),previous(1,3);
  active.extract_from({0,1},{1.f,1.f},{0,2});
  previous.extract_from({0,1,2},{1.f,0.f,2.f},{0,3});
  Buffer<float>w(6),delta(2),dh(3),gradient(6),bias(2);
  float initial[]={.2f,.3f,.4f,-.1f,.5f,.6f};std::copy_n(initial,6,w.p);
  delta.p[0]=.5;delta.p[1]=-.25;
  std::vector<double> expected{.125,0,.05};
  for(int pass=1;pass<=2;++pass) {
    bp_rowmajor_knl<<<1,64,2*2*sizeof(float)>>>(active,previous,w.p,delta.p,3,2,dh.p,gradient.p,bias.p);
    complete();for(int i=0;i<3;++i)near(dh.p[i],pass*expected[i]);
  }
  // Change compressed IDs/extent and gate before reuse; reset full capacity.
  previous.extract_from({2,0},{0.f,1.f},{0,2});
  CUDA_CHECK(cudaMemset(dh.p,0,3*sizeof(float)));
  bp_rowmajor_knl<<<1,64,2*2*sizeof(float)>>>(active,previous,w.p,delta.p,3,2,dh.p,gradient.p,bias.p);
  complete();near(dh.p[0],0);near(dh.p[1],.125);near(dh.p[2],0);
  active.free();previous.free();
  std::cout << "STALE_PASS retained_positive_delta_confirmed=true changed_compressed_ids=true reset_restores_oracle=true\n";
}
} // namespace

int main() {
  try {
    int count=0;CUDA_CHECK(cudaGetDeviceCount(&count));require(count>0,"NVIDIA GPU required");
    CUDA_CHECK(cudaSetDevice(0));cudaDeviceProp prop;CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    int driver=0,runtime=0;CUDA_CHECK(cudaDriverGetVersion(&driver));CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::cout << "CUDA_DEVICE name=" << prop.name << " sm=" << prop.major << prop.minor
              << " driver=" << driver << " runtime=" << runtime << " memory_bytes=" << prop.totalGlobalMem << '\n';
    dense();hash();stale();training();complete();
    std::cout << "GSLIDE_CUDA_PASS checks=" << checks << " max_abs_error=" << max_error
              << " tolerance=8e-6 original_negative_controls=2\n";
    CUDA_CHECK(cudaDeviceReset());return 0;
  } catch(const std::exception &error) {std::cerr << "GSLIDE_CUDA_FAIL " << error.what() << '\n';return 1;}
}
