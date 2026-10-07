// Included after selected actual CUDA definitions by tools/check_gslide_cpu.py.
#include <map>

// TRACE_TEST_ID: GSLIDE-WTA-LSH-CPU
static void test_hash_and_gather() {
  const int D=8, N=3, L=3, K=2, B=36, C=3;
  std::vector<int> bins(L*K*D);
  blockDim.x=32;
  for (int block=0; block<2; ++block) for (int thread=0; thread<32; ++thread) {
    blockIdx.x=block; threadIdx.x=thread;
    init_bins_knl(bins.data(), D, bins.size());
  }
  for (size_t i=0; i<bins.size(); ++i) assert(bins[i] == static_cast<int>(i)%D);
  const int second[8]={1,2,3,4,0,5,6,7};
  for (int table=0; table<L; ++table) for (int i=0; i<D; ++i)
    bins[(table*K+1)*D+i] = table==1 ? 7-i : second[i];
  std::vector<float> weights(N*D, -3);
  for (int node=0; node<N; ++node) weights[node*D+node]=2;
  std::vector<int> buckets(L*B*C, -1), sizes(L*B, 0);
  for (int tile=0; tile<2; ++tile) {
    cpu_block(tile,2);
    init_hash_no_sw_knl(bins.data(), weights.data(), D,N,bins.size(),L,K,D,2,B,C,buckets.data(),sizes.data());
  }
  // Independently enumerated addresses: floor(ln(8))=2 preserves the
  // upstream alias (0,4) == (1,0). The paper's 3-bit packing would differ.
  const int expected[3][3]={{4,4,9},{7,10,13},{4,4,9}};
  for (int table=0; table<L; ++table) for (int bucket=0; bucket<B; ++bucket) {
    int count=0;
    for (int node=0; node<N; ++node) if (expected[table][node]==bucket) {
      assert(buckets[(table*B+bucket)*C+count]==node); ++count;
    }
    assert(sizes[table*B+bucket]==count);
  }
  std::vector<float> inputs(2*D,-3); inputs[0]=2; inputs[D+2]=2;
  std::vector<int> ids(2*L,-1), lengths(2*L,-1);
  for (int tile=0; tile<2; ++tile) {
    cpu_block(tile,2);
    get_hash_knl(bins.data(),inputs.data(),sizes.data(),D,bins.size(),L,K,D,2,2,B,C,ids.data(),lengths.data());
  }
  assert((ids==std::vector<int>{4,7,4,9,13,9}));
  assert((lengths==std::vector<int>{2,1,2,1,1,1}));
  // Prefix scan is a host adapter here; Thrust's GPU scan is not tested.
  OwnedCsc gathered(std::vector<int>(8,-1), {}, {0,2,3,5,6,7,8});
  blockDim.x=32;
  for (int thread=0; thread<32; ++thread) {
    threadIdx.x=thread; blockIdx.x=0;
    gather_buckets_knl(ids.data(),buckets.data(),L,2,B,C,gathered);
  }
  const int expected_nodes[8]={0,1,0,0,1,2,2,2};
  for (int i=0; i<8; ++i) assert(gathered.d_nodes[i]==expected_nodes[i]);
  // All equal (negative) values choose the first bin, not its largest ID.
  std::fill(inputs.begin(),inputs.end(),-7);
  for (int tile=0; tile<2; ++tile) {
    cpu_block(tile,2);
    get_hash_knl(bins.data(),inputs.data(),sizes.data(),D,bins.size(),L,K,D,2,2,B,C,ids.data(),lengths.data());
  }
  for (int id:ids) assert(id==0);
}

// TRACE_TEST_ID: GSLIDE-CANDIDATE-COUNT-CPU
static void test_count() {
  OwnedTable table(2,1,8,2); // Force all keys into one chain.
  int raw[]={2,7,2,9,7,2,4};
  cpu_block(0,1); table.d_block_reduce_cnt(raw,0,7,0);
  assert(table.d_multi_tbl_sizes[0]==2);
  assert(table.d_multi_tbl_pool_used_sizes[0]==3);
  int labels[]={9,6};
  table.d_activate_labels_seq(labels,0,2,0);
  assert(table.d_multi_tbl_sizes[0]==4);
  std::map<int,int> counts;
  for (int i=0; i<9; ++i) if(table.d_multi_tbl_keys[i]!=-1) counts[table.d_multi_tbl_keys[i]]=table.d_multi_tbl_vals[i];
  assert((counts==std::map<int,int>{{2,3},{7,2},{9,3},{4,1},{6,2}}));
  cpu_block(0,1); table.d_block_reduce_cnt(raw,7,7,1);
  assert(table.d_multi_tbl_sizes[1]==0 && table.d_multi_tbl_pool_used_sizes[1]==0);
  for (int i=0; i<2; ++i) assert(table.d_multi_tbl_locks[i]==0);
}

// TRACE_TEST_ID: GSLIDE-SPARSE-FORWARD-CPU
static void test_relu() {
  OwnedCsc inputs({0,2,1},{2,-1,3},{0,2,3});
  OwnedCsc outputs({2,0,1,2},std::vector<float>(4),{0,2,4});
  const float weights[]={1,-2,3, 4,5,-6, 7,8,9}; // columns: input -> output
  const float biases[]={.5f,-.5f,5};
  for (int batch=0; batch<2; ++batch) {
    cpu_block(batch,2); relu_fwd_slide_in_knl(inputs,weights,biases,3,3,outputs);
  }
  const double expected[]={2,0,14.5,0};
  for (int i=0; i<4; ++i) near(outputs.d_vals[i],expected[i]);
}

// TRACE_TEST_ID: GSLIDE-SOFTMAX-CPU
static void test_softmax() {
  OwnedCsc inputs({0,2,1},{2,-1,3},{0,2,3});
  OwnedCsc labels({2,0,1},{},{0,2,3});
  OwnedCsc outputs({2,0,1,2},std::vector<float>(4),{0,2,4});
  const float weights[]={1,0,2, 0,1,0, 2,-1,1}; // rows: output -> input
  float biases[]={.5f,-.5f,1}; float deltas[4]={};
  // Row logits: {4,.5} and {2.5,-2}; include large common negative shift.
  for (float shift: {0.f,-1000.f}) {
    float shifted[]={biases[0]+shift,biases[1]+shift,biases[2]+shift};
    for (int batch=0; batch<2; ++batch) {
      cpu_block(batch,2);
      softmax_fwd_bp_rowmajor_all_sm_knl(inputs,weights,shifted,labels,3,3,3,2,outputs,deltas);
    }
    const double p0=1/(1+std::exp(-3.5)), p2=1/(1+std::exp(-4.5));
    const double expected[]={p0,1-p0,p2,1-p2};
    const double target[]={.5,.5,1,0};
    for (int i=0; i<4; ++i) {
      near(outputs.d_vals[i],expected[i]); near(deltas[i],(target[i]-expected[i])/2);
    }
  }
}

// TRACE_TEST_ID: GSLIDE-SPARSE-GRADIENT-CPU
static void test_gradient() {
  OwnedCsc inputs({0,2,1},{2,-1,3},{0,2,3});
  OwnedCsc acts({2,0,1,2},std::vector<float>(4),{0,2,4});
  const float deltas[]={.25f,-.5f,.125f,-.25f};
  float gradient[9]={}, bias[3]={};
  for (int batch=0; batch<2; ++batch) {
    cpu_block(batch,2); bp_first_layer_knl(acts,inputs,deltas,3,3,gradient,bias);
  }
  const double expected[]={-1,0,.5, 0,.375,-.75, .5,0,-.25};
  for (int i=0; i<9; ++i) near(gradient[i],expected[i]);
  near(bias[0],-.5); near(bias[1],.125); near(bias[2],0);
}

// TRACE_TEST_ID: GSLIDE-ADAM-CPU
static void test_adam() {
  const int n=35; // Non-multiple of block size; include inactive tail lanes.
  std::vector<float> weights(n,.5f), grad(n), mom(n,0), vel(n,0);
  for (int step=0; step<2; ++step) {
    const float t=step==0 ? .25f : -.125f;
    std::fill(grad.begin(),grad.end(),t);
    blockDim.x=32;
    for (int block=0; block<2; ++block) for (int thread=0; thread<32; ++thread) {
      blockIdx.x=block; threadIdx.x=thread;
      update_weights_knl(weights.data(),grad.data(),mom.data(),vel.data(),.01f,n);
    }
    const double m0=.025, v0=.0000625;
    const double m=step==0 ? m0 : .9*m0-.0125;
    const double v=step==0 ? v0 : .999*v0+.001*.125*.125;
    const double w0=.5+.01*m0/(std::sqrt(v0)+1e-8);
    const double w=step==0 ? w0 : w0+.01*m/(std::sqrt(v)+1e-8);
    for (int i=0; i<n; ++i) {
      near(weights[i],w); near(mom[i],m); near(vel[i],v); near(grad[i],0);
    }
  }
}
int main() {
  test_hash_and_gather(); test_count(); test_relu(); test_softmax(); test_gradient(); test_adam();
  std::cout << "GSLIDE_CPU_ORACLES_PASS cases=6\n";
}
