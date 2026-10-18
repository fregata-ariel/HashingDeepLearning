#pragma once
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

// TRACE_TEST_ID: GSLIDE-SYCL-DEVICE-115
// Support: translated shared kernels, not execution of the original CUDA bodies.
// Shared CPU/GPU SYCL implementation; all pointers are USM in q's context.
// Calls complete before returning, including asynchronous failures. Shapes and
// capacities are host-validated; inputs must be finite and IDs in range.
namespace gslide::sycl_port {
inline void require(bool ok, const char* msg) { if (!ok) throw std::invalid_argument(msg); }
inline std::size_t padded(std::size_t n, std::size_t local) { return (n+local-1)/local*local; }
inline void launch_policy(sycl::queue& q, int local) {
  require(local>0 && std::size_t(local)<=q.get_device().get_info<sycl::info::device::max_work_group_size>(), "local size exceeds device");
}
// minstd_rand (48271, 0, 2147483647), with seed zero normalized to one.
// Exponentiation gives discard(id)+one draw without a quadratic per-lane loop.
inline std::uint32_t random_at(std::uint32_t seed, std::uint64_t id) {
  std::uint64_t value=seed%2147483647u; if(!value) value=1;
  std::uint64_t power=48271, exponent=id+1;
  while(exponent) { if(exponent&1) value=value*power%2147483647u; power=power*power%2147483647u; exponent>>=1; }
  return value;
}
inline void random_keys(sycl::queue& q, unsigned* keys, int n, unsigned seed, int previous, bool packed, int local=64) {
  require(n>=0 && previous>0 && n/previous<65536,"RNG shape/packed prefix"); launch_policy(q,local); if(!n)return;
  q.parallel_for(sycl::nd_range<1>(padded(n,local),local),[=](sycl::nd_item<1> it){ int i=it.get_global_linear_id(); if(i<n) { auto r=random_at(seed,i); keys[i]=packed ? (unsigned(i/previous)<<16)|(r&(1u<<15)) : r; }}).wait_and_throw();
}
inline void dense_ids(sycl::queue& q,int* ids,int* offsets,int nodes,int batch,int capacity,int local=64) {
  require(nodes>=0 && batch>=0 && std::int64_t(nodes)*batch<=capacity,"dense capacity"); launch_policy(q,local);
  auto n=std::max(std::size_t(nodes)*batch,std::size_t(batch)+1);
  q.parallel_for(sycl::nd_range<1>(padded(n,local),local),[=](sycl::nd_item<1> it){ auto i=it.get_global_linear_id(); if(i<std::size_t(nodes)*batch)ids[i]=i%nodes; if(i<=std::size_t(batch))offsets[i]=i*nodes; }).wait_and_throw();
}
// Subgroup semantics use the actual device subgroup, never CUDA warp width.
// Block reduction explicitly stages all work-items in local memory. Returned
// group results are valid in every lane; input tails use the proper identity.
inline void reductions(sycl::queue& q,const float* input,int n,float* sums,float* maxima,float* subgroup_sums,float* subgroup_maxima,int local=64) {
  require(n>=0,"reduction extent"); launch_policy(q,local); if(!n)return;
  q.submit([&](sycl::handler& h){
    sycl::local_accessor<float,1> s(local,h),m(local,h);
    h.parallel_for(sycl::nd_range<1>(padded(n,local),local),[=](sycl::nd_item<1> it){
      int i=it.get_global_linear_id(), lane=it.get_local_linear_id(); auto sg=it.get_sub_group();
      float x=i<n?input[i]:0.f, y=i<n?input[i]:-std::numeric_limits<float>::max();
      auto ss=sycl::reduce_over_group(sg,x,sycl::plus<float>()); auto sm=sycl::reduce_over_group(sg,y,sycl::maximum<float>());
      subgroup_sums[i]=ss; subgroup_maxima[i]=sm;
      s[lane]=x; m[lane]=y; it.barrier(sycl::access::fence_space::local_space);
      // Odd-sized groups are supported as well as powers of two.
      for(int active=local;active>1;) { int next=(active+1)/2; if(lane<active-next){s[lane]+=s[lane+next]; m[lane]=sycl::fmax(m[lane],m[lane+next]);} it.barrier(sycl::access::fence_space::local_space); active=next; }
      if(!lane){sums[it.get_group_linear_id()]=s[0]; maxima[it.get_group_linear_id()]=m[0];}
    });
  }).wait_and_throw();
}
struct hash_shape { int previous,nodes,tables,k,bin,tile,per_thread,buckets,capacity; };
inline int address(const int* bins,const float* row,int k,int bin,int shift) {
  int result=0; for(int j=0;j<k;++j){ float maximum=-std::numeric_limits<float>::max(); int winner=0; for(int p=0;p<bin;++p)if(row[bins[j*bin+p]]>maximum){maximum=row[bins[j*bin+p]];winner=p;} result+=winner<<((k-1-j)*shift); } return result;
}
// per_thread==tile is original tt mode. Other values share each node across
// lanes; staging and grid use node_idx, correcting the original lane/node bug.
inline void build_hash(sycl::queue& q,const int* bins,const float* weights,hash_shape z,int* buckets,int* sizes,int local=64) {
  require(z.previous>0&&z.nodes>=0&&z.tables>0&&z.k>0&&z.bin>0&&z.tile>0&&z.per_thread>0&&z.tile%z.per_thread==0&&z.capacity>0,"hash shape");
  int lanes=z.tile/z.per_thread, shift=int(std::log(float(z.bin)));
  require(local%lanes==0 && (z.k-1)*shift<30,"hash lanes/shift"); launch_policy(q,local);
  std::uint64_t bound=0; for(int j=0;j<z.k;++j)bound+=std::uint64_t(z.bin-1)<<((z.k-1-j)*shift);
  require(bound<unsigned(z.buckets),"bucket address capacity"); if(!z.nodes)return;
  q.submit([&](sycl::handler& h){ sycl::local_accessor<int,1> tile(z.tile*z.k*z.bin,h); sycl::local_accessor<float,1> rows((local/lanes)*z.previous,h);
    h.parallel_for(sycl::nd_range<1>(padded(z.nodes*lanes,local),local),[=](sycl::nd_item<1> it){
      int lane=it.get_local_linear_id(), node=it.get_global_linear_id()/lanes, slot=lane/lanes, tile_lane=lane%lanes;
      int node_begin=it.get_group_linear_id()*(local/lanes);
      for(int i=lane;i<(local/lanes)*z.previous;i+=local){int global=node_begin*z.previous+i; rows[i]=global<z.nodes*z.previous?weights[global]:0;}
      for(int t=0;t<z.tables;t+=z.tile){
        for(int i=lane;i<z.tile*z.k*z.bin;i+=local){int global=t*z.k*z.bin+i;tile[i]=global<z.tables*z.k*z.bin?bins[global]:0;}
        it.barrier(sycl::access::fence_space::local_space);
        if(node<z.nodes) for(int j=tile_lane*z.per_thread;j<(tile_lane+1)*z.per_thread&&t+j<z.tables;++j){
          int a=address(tile.get_multi_ptr<sycl::access::decorated::no>().get()+j*z.k*z.bin,rows.get_multi_ptr<sycl::access::decorated::no>().get()+slot*z.previous,z.k,z.bin,shift)+(t+j)*z.buckets;
          sycl::atomic_ref<int,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> count(sizes[a]); int pos=count.fetch_add(1)%z.capacity; sycl::atomic_ref<int,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> entry(buckets[a*z.capacity+pos]); entry.store(node);
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
    });
  }).wait_and_throw();
}
// Backend variant: bounded direct-address frequency storage [sample,node]
// replaces the CUDA lock-based linked pool. Capacity is exactly batch*nodes;
// no spinning lock or CUDA lock progress claim. Groups map to samples; L
// offsets per sample map to the same gathered extent as the native wrapper.
inline void count_candidates(sycl::queue& q,const int* raw,const int* offsets,int tables,int batch,int nodes,int capacity,int* counts,int local=64) {
  require(tables>0&&batch>=0&&nodes>0&&std::int64_t(batch)*nodes<=capacity,"candidate capacity"); launch_policy(q,local); if(!batch)return;
  q.parallel_for(sycl::nd_range<1>(batch*local,local),[=](sycl::nd_item<1> it){int b=it.get_group_linear_id(); for(int i=offsets[b*tables]+it.get_local_linear_id();i<offsets[(b+1)*tables];i+=local){sycl::atomic_ref<int,sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space> c(counts[b*nodes+raw[i]]);c.fetch_add(1);}}).wait_and_throw();
}
inline void activate_labels(sycl::queue& q,const int* labels,const int* offsets,const int* permutation,int batch,int nodes,int threshold,int minimum,unsigned seed,int* counts,int* sizes,int local=64) {
  require(batch>=0&&nodes>0&&threshold>0&&minimum>=0&&minimum<=nodes,"activation shape/minimum"); launch_policy(q,local); if(!batch)return;
  q.parallel_for(sycl::nd_range<1>(padded(batch,local),local),[=](sycl::nd_item<1> it){int b=it.get_global_linear_id();if(b>=batch)return; int* row=counts+b*nodes; int size=0;for(int n=0;n<nodes;++n)size+=row[n]>=threshold;
    for(int i=offsets[b];i<offsets[b+1];++i){int n=labels[i];if(row[n]<threshold)++size;row[n]+=threshold;}
    if(permutation){int i=random_at(seed,b)%nodes;for(int visited=0;size<minimum&&visited<nodes;++visited){int n=permutation[i];if(row[n]<threshold)++size;row[n]+=threshold;i=(i+1)%nodes;}}
    sizes[b]=size;
  }).wait_and_throw();
}
} // namespace gslide::sycl_port
