// Included after extracted, unchanged maintained CUDA bodies. No Network::train
// or native CUDA launch runs here. Host adapters below are named substitutions.
#include <map>
#include <numeric>
#include <set>
#include <iomanip>

namespace training {
constexpr int I=3, H=4, O=4, N=2, L=3, K=2, B=16, C=4, THRESHOLD=2;
constexpr unsigned fixture_seed=40; // host rotations, not CUDA/Thrust RNG
constexpr double beta1=.9, beta2=.999, epsilon=1e-8, learning_rate=.01;
constexpr double arithmetic_tolerance=3e-6, parameter_tolerance=8e-6;
using D = std::vector<double>;
using F = std::vector<float>;
static double max_error=0;
static void near(float actual,double expected,double tolerance=2e-6) {
  max_error=std::max(max_error,std::abs(actual-expected));
  ::near(actual,expected,tolerance);
}

struct Params {
  F value, grad, moment, velocity;
  D ref, ref_grad, ref_moment, ref_velocity;
  Params(std::initializer_list<float> initial): value(initial), grad(value.size()),
      moment(value.size()), velocity(value.size()), ref(value.begin(),value.end()),
      ref_grad(value.size()), ref_moment(value.size()), ref_velocity(value.size()) {}
  void compare_gradient() const {
    for(size_t i=0;i<grad.size();++i) near(grad[i],ref_grad[i],arithmetic_tolerance);
  }
  void update(int step) {
    compare_gradient();
    // Explicit host emulation of main.cu bias correction. Kernel consumes lr.
    const float lr=static_cast<float>(learning_rate*std::sqrt(1-std::pow(beta2,step))/(1-std::pow(beta1,step)));
    for(size_t i=0;i<value.size();++i) {
      cpu_block(i,value.size());
      update_weights_knl(value.data(),grad.data(),moment.data(),velocity.data(),lr,value.size());
      const double g=ref_grad[i];
      ref_moment[i]=beta1*ref_moment[i]+(1-beta1)*g;
      ref_velocity[i]=beta2*ref_velocity[i]+(1-beta2)*g*g;
      ref[i]+=lr*ref_moment[i]/(std::sqrt(ref_velocity[i])+epsilon);
      near(value[i],ref[i],parameter_tolerance);
      near(moment[i],ref_moment[i],arithmetic_tolerance);
      near(velocity[i],ref_velocity[i],arithmetic_tolerance);
      near(grad[i],0,0); // atomicExch consumes every accumulator, biases too
      ref_grad[i]=0;
    }
  }
};

// Host inclusive scan substitute for Thrust; capacities are checked before use.
static std::vector<int> host_scan_lengths(const std::vector<int>& lengths) {
  std::vector<int> offsets{0};
  for(int length:lengths) { assert(length>=0 && length<=C); offsets.push_back(offsets.back()+length); }
  return offsets;
}

// Host traversal/filter substitutes for get_act_nodes' Thrust copy_if. Traverse
// links to verify reachability, then emit storage order as the source does.
static std::vector<int> host_linked_filter(const OwnedTable& table, int sample) {
  const int capacity=table.bucket_num_per_tbl+table.pool_size;
  const int base=sample*capacity;
  std::set<int> reachable;
  for(size_t bucket=0;bucket<table.bucket_num_per_tbl;++bucket) {
    int entry=bucket;
    while(entry!=-1 && table.d_multi_tbl_keys[base+entry]!=-1) {
      assert(entry>=0 && entry<capacity && reachable.insert(entry).second);
      entry=table.d_multi_tbl_nexts[base+entry];
    }
  }
  std::vector<int> active;
  for(int entry=0;entry<capacity;++entry) if(table.d_multi_tbl_keys[base+entry]!=-1) {
    assert(reachable.count(entry));
    if(table.d_multi_tbl_vals[base+entry]>=THRESHOLD) active.push_back(table.d_multi_tbl_keys[base+entry]);
  }
  assert(static_cast<int>(active.size())==table.d_multi_tbl_sizes[sample]);
  assert(table.d_multi_tbl_pool_used_sizes[sample]<=static_cast<int>(table.pool_size));
  assert(table.d_multi_tbl_locks[sample]==0);
  return active;
}

// Independent scalar WTA math over dense vectors. Natural-log packing is the
// maintained variant; retaining it does not assert paper bit concatenation.
static int oracle_address(const D& vector, const std::vector<int>& bins, int table) {
  int address=0;
  for(int hash=0;hash<K;++hash) {
    int winner=0;
    for(int pos=1;pos<H;++pos)
      if(vector[bins[(table*K+hash)*H+pos]]>vector[bins[(table*K+hash)*H+winner]]) winner=pos;
    address+=winner*static_cast<int>(std::pow(2,(K-1-hash)*static_cast<int>(std::log(H))));
  }
  assert(address>=0 && address<B);
  return address;
}

struct Result { F values; std::vector<int> active_signature; D losses; double stale_max=0; int changed_addresses=0; };

// TRACE_TEST_ID: GSLIDE-TRAINING-COMPOSITION-CPU
static Result connected_two_updates() {
  near(BETA1,beta1,1e-7); near(BETA2,beta2,1e-7); near(EPS,epsilon,1e-14);
  Params hidden_w{.7f,.2f,.1f,-.4f, .1f,.5f,.7f,-.5f, .2f,-.3f,.1f,.1f};
  Params hidden_b{.05f,.02f,.03f,-.3f};
  // Node 1's near tie is deliberately crossed by the connected first Adam
  // update: its node-0 gradient is negative and node-2 gradient is positive.
  Params output_w{.6f,.2f,.1f,-.2f, .5f,-.1f,.49f,-.3f,
                  -.1f,.2f,.7f,-.2f, .1f,.6f,.2f,-.1f};
  Params output_b{.03f,-.04f,.05f,-.01f};
  const F initial_hidden=hidden_w.value, initial_output=output_w.value;
  const F initial_hidden_bias=hidden_b.value, initial_output_bias=output_b.value;
  OwnedCsc inputs({0,2,1,2},{1,.5f,1.2f,-.4f},{0,2,4});
  OwnedCsc labels({3,1},{},{0,1,2});
  // Dense hidden IDs/offsets supplied by host; get_dense_acts_knl is not selected.
  OwnedCsc hidden({0,1,2,3,0,1,2,3},F(N*H),{0,H,2*H});
  const D dense_x={1,0,.5, 0,static_cast<double>(1.2f),static_cast<double>(-.4f)};
  std::vector<int> bins(L*K*H);
  for(int i=0;i<L*K*H;++i) { cpu_block(i,L*K*H); init_bins_knl(bins.data(),H,bins.size()); }
  // Explicit deterministic host permutations replace GPU random-key sorting.
  for(int t=0;t<L;++t) for(int h=0;h<K;++h) for(int p=0;p<H;++p)
    bins[(t*K+h)*H+p]=(p+fixture_seed+t*K+h)%H;
  F hidden_delta(N*H), prior_delta(N*H);
  std::vector<int> previous_buckets, previous_sizes, previous_node1_addresses;
  Result result;
  for(int step=1;step<=2;++step) {
    D dense_hidden(N*H);
    for(int sample=0;sample<N;++sample) {
      cpu_block(sample,N);
      relu_fwd_slide_in_knl(inputs,hidden_w.value.data(),hidden_b.value.data(),H,H,hidden);
      for(int node=0;node<H;++node) {
        double activation=hidden_b.ref[node];
        for(int input=0;input<I;++input) activation+=dense_x[sample*I+input]*hidden_w.ref[input*H+node];
        dense_hidden[sample*H+node]=std::max(0.,activation);
        near(hidden.d_vals[sample*H+node],dense_hidden[sample*H+node],arithmetic_tolerance);
      }
    }
    // Reconstruct the index from current, updated output weights every step.
    std::vector<int> buckets(L*B*C,-1), sizes(L*B,0);
    for(int tile=0;tile<2;++tile) {
      cpu_block(tile,2);
      init_hash_no_sw_knl(bins.data(),output_w.value.data(),H,O,bins.size(),L,K,H,2,B,C,buckets.data(),sizes.data());
    }
    std::vector<std::vector<int>> ref_buckets(L*B);
    for(int table=0;table<L;++table) for(int node=0;node<O;++node) {
      D row(output_w.ref.begin()+node*H,output_w.ref.begin()+(node+1)*H);
      ref_buckets[table*B+oracle_address(row,bins,table)].push_back(node);
    }
    std::vector<int> current_node1_addresses;
    D node1_row(output_w.ref.begin()+H,output_w.ref.begin()+2*H);
    for(int table=0;table<L;++table) current_node1_addresses.push_back(oracle_address(node1_row,bins,table));
    if(step==2) {
      // A stale pre-Adam index must differ observably from this rebuild.
      assert(buckets!=previous_buckets && sizes!=previous_sizes);
      for(int table=0;table<L;++table)
        result.changed_addresses+=(current_node1_addresses[table]!=previous_node1_addresses[table]);
      assert(result.changed_addresses>0);
    }
    for(int bucket=0;bucket<L*B;++bucket) {
      assert(sizes[bucket]==static_cast<int>(ref_buckets[bucket].size()) && sizes[bucket]<=C);
      for(int pos=0;pos<sizes[bucket];++pos) assert(buckets[bucket*C+pos]==ref_buckets[bucket][pos]);
    }
    std::vector<int> ids(N*L), lengths(N*L);
    for(int tile=0;tile<2;++tile) {
      cpu_block(tile,2);
      get_hash_knl(bins.data(),hidden.d_vals,sizes.data(),H,bins.size(),L,K,H,2,N,B,C,ids.data(),lengths.data());
    }
    if(step==2) {
      bool stale_query_differs=false;
      for(int sample=0;sample<N;++sample) for(int table=0;table<L;++table)
        stale_query_differs |= lengths[sample*L+table]!=previous_sizes[table*B+ids[sample*L+table]];
      assert(stale_query_differs); // rebuilding changes the connected query too
    }
    previous_buckets=buckets; previous_sizes=sizes; previous_node1_addresses=current_node1_addresses;
    std::vector<int> ref_raw;
    std::vector<std::map<int,int>> ref_counts(N);
    for(int sample=0;sample<N;++sample) for(int table=0;table<L;++table) {
      D query(dense_hidden.begin()+sample*H,dense_hidden.begin()+(sample+1)*H);
      int address=oracle_address(query,bins,table);
      assert(ids[sample*L+table]==address);
      assert(lengths[sample*L+table]==static_cast<int>(ref_buckets[table*B+address].size()));
      for(int node:ref_buckets[table*B+address]) { ref_raw.push_back(node); ++ref_counts[sample][node]; }
    }
    const auto raw_offsets=host_scan_lengths(lengths);
    OwnedCsc raw(std::vector<int>(raw_offsets.back(),-1),{},raw_offsets);
    for(int block=0;block<N*L;++block) {
      cpu_block(block,N*L); gather_buckets_knl(ids.data(),buckets.data(),L,N,B,C,raw);
    }
    assert(std::equal(ref_raw.begin(),ref_raw.end(),raw.d_nodes));
    OwnedTable counts(N,1,O,THRESHOLD); // collisions intentional, pool safe
    std::vector<int> act_nodes, act_offsets{0};
    std::vector<std::set<int>> ref_active(N);
    for(int sample=0;sample<N;++sample) {
      cpu_block(sample,N);
      counts.d_block_reduce_cnt(raw.d_nodes,raw.d_offsets[sample*L],raw.d_offsets[(sample+1)*L],sample);
      // Step 1 inserts both absent labels. After the rank swap, sample 1's
      // label is already retrieved and receives the source's count promotion.
      assert((ref_counts[sample].count(labels.d_nodes[sample])!=0)==(step==2 && sample==1));
      ref_counts[sample][labels.d_nodes[sample]]+=THRESHOLD;
      counts.d_activate_labels_seq(labels.d_nodes,sample,sample+1,sample);
      for(const auto& entry:ref_counts[sample]) if(entry.second>=THRESHOLD) ref_active[sample].insert(entry.first);
      const auto active=host_linked_filter(counts,sample);
      assert(std::set<int>(active.begin(),active.end())==ref_active[sample]);
      for(int slot=0;slot<1+O;++slot) if(counts.d_multi_tbl_keys[sample*(1+O)+slot]!=-1)
        assert(counts.d_multi_tbl_vals[sample*(1+O)+slot]==ref_counts[sample].at(counts.d_multi_tbl_keys[sample*(1+O)+slot]));
      assert(active.size()>=2 && active.size()<O);
      act_nodes.insert(act_nodes.end(),active.begin(),active.end()); act_offsets.push_back(act_nodes.size());
      result.active_signature.insert(result.active_signature.end(),active.begin(),active.end());
      result.active_signature.push_back(-1);
    }
    OwnedCsc output(act_nodes,F(act_nodes.size()),act_offsets);
    F output_delta(act_nodes.size());
    D ref_delta(N*O), ref_hidden_delta(N*H);
    double loss=0, actual_loss=0;
    for(int sample=0;sample<N;++sample) {
      cpu_block(sample,N);
      softmax_fwd_bp_rowmajor_all_sm_knl(hidden,output_w.value.data(),output_b.value.data(),labels,H,H,O,1,output,output_delta.data());
      D logits(O); double largest=-1e100;
      for(int node:ref_active[sample]) {
        logits[node]=output_b.ref[node];
        for(int h=0;h<H;++h) logits[node]+=output_w.ref[node*H+h]*dense_hidden[sample*H+h];
        largest=std::max(largest,logits[node]);
      }
      double norm=epsilon; for(int node:ref_active[sample]) norm+=std::exp(logits[node]-largest);
      double probability_sum=0;
      for(int slot=act_offsets[sample];slot<act_offsets[sample+1];++slot) {
        const int node=act_nodes[slot];
        double probability=std::exp(logits[node]-largest)/norm;
        probability_sum+=probability;
        ref_delta[sample*O+node]=((node==labels.d_nodes[sample]?1.:0.)-probability)/N;
        near(output.d_vals[slot],probability,arithmetic_tolerance);
        near(output_delta[slot],ref_delta[sample*O+node],arithmetic_tolerance);
        if(node==labels.d_nodes[sample]) {
          loss-=std::log(probability)/N;
          actual_loss-=std::log(static_cast<double>(output.d_vals[slot]))/N;
        }
      }
      near(probability_sum,1,2e-8);
      for(int h=0;h<H;++h) for(int node:ref_active[sample]) {
        if(dense_hidden[sample*H+h]>0) ref_hidden_delta[sample*H+h]+=output_w.ref[node*H+h]*ref_delta[sample*O+node];
        output_w.ref_grad[node*H+h]+=dense_hidden[sample*H+h]*ref_delta[sample*O+node];
      }
      for(int node:ref_active[sample]) output_b.ref_grad[node]+=ref_delta[sample*O+node];
    }
    assert(std::isfinite(actual_loss) && actual_loss>0);
    near(actual_loss,loss,arithmetic_tolerance); result.losses.push_back(actual_loss);
    // TRACE_TEST_ID: GSLIDE-HIDDEN-DELTA-STALE-CPU
    // At step 2, replay actual backward with the previous step's deltas. The
    // source adds that state for positive activations. This is a characterization
    // of a wiring gap, not a fix or native Network::train execution.
    if(step==2) {
      F stale=prior_delta, scratch_w(O*H), scratch_b(O);
      for(int sample=0;sample<N;++sample) {
        cpu_block(sample,N);
        bp_rowmajor_knl(output,hidden,output_w.value.data(),output_delta.data(),H,O,stale.data(),scratch_w.data(),scratch_b.data());
      }
      for(int i=0;i<N*H;++i) {
        const double retained=dense_hidden[i]>0?prior_delta[i]:0.;
        near(stale[i],ref_hidden_delta[i]+retained,arithmetic_tolerance);
        result.stale_max=std::max(result.stale_max,std::abs(stale[i]-ref_hidden_delta[i]));
      }
      assert(result.stale_max>1e-3);
    }
    // Explicit host per-step reset substitution. No claim production resets.
    std::fill(hidden_delta.begin(),hidden_delta.end(),0);
    for(int sample=0;sample<N;++sample) {
      cpu_block(sample,N);
      bp_rowmajor_knl(output,hidden,output_w.value.data(),output_delta.data(),H,O,hidden_delta.data(),output_w.grad.data(),output_b.grad.data());
    }
    for(int i=0;i<N*H;++i) near(hidden_delta[i],ref_hidden_delta[i],arithmetic_tolerance);
    prior_delta=hidden_delta;
    for(int sample=0;sample<N;++sample) {
      cpu_block(sample,N);
      bp_first_layer_knl(hidden,inputs,hidden_delta.data(),H,H,hidden_w.grad.data(),hidden_b.grad.data());
      for(int h=0;h<H;++h) {
        hidden_b.ref_grad[h]+=ref_hidden_delta[sample*H+h];
        for(int input=0;input<I;++input) hidden_w.ref_grad[input*H+h]+=dense_x[sample*I+input]*ref_hidden_delta[sample*H+h];
      }
    }
    hidden_w.update(step); hidden_b.update(step); output_w.update(step); output_b.update(step);
  }
  assert(hidden_w.value!=initial_hidden && output_w.value!=initial_output);
  assert(hidden_b.value!=initial_hidden_bias && output_b.value!=initial_output_bias);
  assert(result.losses[1]<result.losses[0]);
  for(const auto* parameter:{&hidden_w,&hidden_b,&output_w,&output_b})
    result.values.insert(result.values.end(),parameter->value.begin(),parameter->value.end());
  return result; // OwnedCsc/OwnedTable RAII invokes normal host teardown
}
} // namespace training

int main() {
  const auto first=training::connected_two_updates(), repeat=training::connected_two_updates();
  assert(first.values==repeat.values && first.active_signature==repeat.active_signature && first.losses==repeat.losses
         && first.changed_addresses==repeat.changed_addresses);
  std::cout<<std::setprecision(9)<<"GSLIDE_TRAINING_COMPOSITION_PASS updates=2 repeat=2 seed="<<training::fixture_seed
           <<" loss0="<<first.losses[0]<<" loss1="<<first.losses[1]<<" stale_delta_max="<<first.stale_max
           <<" changed_wta_addresses="<<first.changed_addresses
           <<" max_observed_abs_error="<<training::max_error
           <<" max_parameter_tolerance="<<training::parameter_tolerance<<"\n";
}
