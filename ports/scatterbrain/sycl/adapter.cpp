// SPDX-License-Identifier: MIT
#include <scatterbrain/local_product.hpp>
#include <hashing/sycl_backend.hpp>
#include <algorithm>
#include <iostream>
#include <string>
namespace {
thread_local std::string last_error;
sycl::queue cpu_queue() {
 auto queue=hashing::sycl_backend::make_queue(hashing::sycl_backend::device_kind::cpu);
 hashing::sycl_backend::require_cpu(queue); return queue;
}
std::vector<float> copy(const float* p,int n) { if(!p) throw std::invalid_argument("null tensor"); return {p,p+n}; }
void store(float* p,const std::vector<float>& value) { if(!p) throw std::invalid_argument("null output"); std::copy(value.begin(),value.end(),p); }
}
// Narrow synchronous C ABI for the value-owned C++ SYCL API. Caller guarantees
// buffers have the documented contiguous sizes; Python adapter validates them.
extern "C" const char* sb_last_error() { return last_error.c_str(); }
extern "C" int sb_describe() {
 try { hashing::sycl_backend::describe_queue(std::cerr,cpu_queue()); return 0; }
 catch(const std::exception& e) { last_error=e.what(); return 1; }
}
extern "C" int sb_run(int operation,const int* dims,const float* a,const float* b,
 const float* grad,const float* mask,const std::int64_t* lengths,float* out1,float* out2) {
 try {
  if(!dims) throw std::invalid_argument("null dimensions");
  hashing::scatterbrain::shape s{dims[0],dims[1],dims[2],dims[3],dims[4],dims[5],dims[6]}; s.validate();
  auto queue=cpu_queue();
  using namespace hashing::scatterbrain;
  if(operation==0||operation==1) {
   if(!lengths) throw std::invalid_argument("null lengths");
   auto q=copy(a,s.b*s.h*s.q*s.e),k=copy(b,s.b*s.h*s.k*s.e);
   std::vector<std::int64_t> lens(lengths,lengths+s.b);
   if(operation==0) store(out1,local_dot_product(queue,q,k,copy(mask,s.q*s.k),lens,s));
   else { auto result=local_dot_backward(queue,q,k,lens,copy(grad,s.b*s.h*s.q*s.w),s); store(out1,result.first); store(out2,result.second); }
  } else if(operation==2||operation==3) {
   auto attention=copy(a,s.b*s.h*s.q*s.w),values=copy(b,s.b*s.h*s.k*s.v);
   if(operation==2) store(out1,local_weighted_average(queue,attention,values,s));
   else { auto result=local_weighted_average_backward(queue,attention,values,copy(grad,s.b*s.h*s.q*s.v),s); store(out1,result.first); store(out2,result.second); }
  } else throw std::invalid_argument("unknown operation");
  return 0;
 } catch(const std::exception& e) { last_error=e.what(); return 1; }
}
