// SPDX-License-Identifier: MIT
// TRACE_TEST_ID: SCATTERBRAIN-LOCAL-PRODUCT-SYCL-CPU
// Relation: support; local products implement the selected sparse boundary.
// SYCL rewrite of Idiap Research Institute local products, copyright2020.
// Original author Angelos Katharopoulos; retained notice: ../../../notices/idiap-fast-transformers-LICENSE.
#pragma once
#include <sycl/sycl.hpp>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <limits>
namespace hashing::scatterbrain {
struct shape {
  int b,h,q,k,e,v,w;
  void validate() const {
    if(b<1||h<1||q<1||k<1||e<1||v<1||w<1) throw std::invalid_argument("positive dimensions required");
    // Conservative bound keeps every kernel index and product within int32.
    std::uint64_t product=1;
    for(int d:{b,h,q,k,e,v,w}) {
      if(product>std::uint64_t(std::numeric_limits<int>::max())/d) throw std::invalid_argument("shape exceeds bounded int32 indexing");
      product*=d;
    }
  }
};
inline void check(const std::vector<float>& x,std::size_t size) {
  if(x.size()!=size) throw std::invalid_argument("tensor shape mismatch");
  for(float value:x) if(!std::isfinite(value)) throw std::invalid_argument("finite FP32 inputs required");
}
inline void lengths_check(const std::vector<std::int64_t>& lengths,shape s) {
  if(lengths.size()!=std::size_t(s.b)) throw std::invalid_argument("length shape");
  for(auto value:lengths) if(value<0||value>s.k) throw std::invalid_argument("prefix length outside physical keys");
}
inline std::vector<float> local_dot_product(sycl::queue& queue,const std::vector<float>& q,
 const std::vector<float>& k,const std::vector<float>& mask,const std::vector<std::int64_t>& lengths,shape s) {
  s.validate(); check(q,s.b*s.h*s.q*s.e); check(k,s.b*s.h*s.k*s.e); check(mask,s.q*s.k); lengths_check(lengths,s);
  std::vector<float> out(s.b*s.h*s.q*s.w);
  {
    sycl::buffer<float,1> qb(q.data(),sycl::range<1>(q.size())),kb(k.data(),sycl::range<1>(k.size())),mb(mask.data(),sycl::range<1>(mask.size())),ob(out.data(),sycl::range<1>(out.size()));
    sycl::buffer<std::int64_t,1> lb(lengths.data(),sycl::range<1>(lengths.size()));
    queue.submit([&](sycl::handler& cgh){
      auto qa=qb.get_access<sycl::access::mode::read>(cgh),ka=kb.get_access<sycl::access::mode::read>(cgh),ma=mb.get_access<sycl::access::mode::read>(cgh);
      auto la=lb.get_access<sycl::access::mode::read>(cgh); auto oa=ob.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(out.size()),[=](sycl::id<1> id){
        int slot=id[0]%s.w,row=id[0]/s.w,qi=row%s.q,bh=row/s.q,ki=qi-s.w/2+slot;
        float value=-1e24f;
        if(ki>=0&&ki<la[bh/s.h]) { value=0; for(int e=0;e<s.e;++e) value+=qa[row*s.e+e]*ka[(bh*s.k+ki)*s.e+e]; value+=ma[qi*s.k+ki]; }
        oa[id]=value;
      });
    }); queue.wait_and_throw();
  } return out;
}
inline std::pair<std::vector<float>,std::vector<float>> local_dot_backward(sycl::queue& queue,
 const std::vector<float>& q,const std::vector<float>& k,const std::vector<std::int64_t>& lengths,
 const std::vector<float>& grad,shape s) {
  s.validate(); check(q,s.b*s.h*s.q*s.e); check(k,s.b*s.h*s.k*s.e); check(grad,s.b*s.h*s.q*s.w); lengths_check(lengths,s);
  std::vector<float> dq(q.size()),dk(k.size());
  {
    sycl::buffer<float,1> qb(q.data(),sycl::range<1>(q.size())),kb(k.data(),sycl::range<1>(k.size())),gb(grad.data(),sycl::range<1>(grad.size())),dqb(dq.data(),sycl::range<1>(dq.size())),dkb(dk.data(),sycl::range<1>(dk.size()));
    sycl::buffer<std::int64_t,1> lb(lengths.data(),sycl::range<1>(lengths.size()));
    queue.submit([&](sycl::handler& cgh){
      auto ka=kb.get_access<sycl::access::mode::read>(cgh),ga=gb.get_access<sycl::access::mode::read>(cgh);
      auto la=lb.get_access<sycl::access::mode::read>(cgh); auto da=dqb.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(dq.size()),[=](sycl::id<1> id){
        int e=id[0]%s.e,row=id[0]/s.e,qi=row%s.q,bh=row/s.q; float value=0;
        for(int slot=0;slot<s.w;++slot) { int ki=qi-s.w/2+slot; if(ki>=0&&ki<la[bh/s.h]) value+=ga[row*s.w+slot]*ka[(bh*s.k+ki)*s.e+e]; }
        da[id]=value;
      });
    });
    queue.submit([&](sycl::handler& cgh){
      auto qa=qb.get_access<sycl::access::mode::read>(cgh),ga=gb.get_access<sycl::access::mode::read>(cgh);
      auto la=lb.get_access<sycl::access::mode::read>(cgh); auto da=dkb.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(dk.size()),[=](sycl::id<1> id){
        int e=id[0]%s.e,ki=(id[0]/s.e)%s.k,bh=id[0]/(s.e*s.k); float value=0;
        if(ki<la[bh/s.h]) for(int qi=0;qi<s.q;++qi) { int slot=ki-qi+s.w/2; if(slot>=0&&slot<s.w) value+=ga[(bh*s.q+qi)*s.w+slot]*qa[(bh*s.q+qi)*s.e+e]; }
        da[id]=value;
      });
    }); queue.wait_and_throw();
  } return {dq,dk};
}
inline std::vector<float> local_weighted_average(sycl::queue& queue,const std::vector<float>& a,const std::vector<float>& v,shape s) {
  s.validate(); check(a,s.b*s.h*s.q*s.w); check(v,s.b*s.h*s.k*s.v); std::vector<float> out(s.b*s.h*s.q*s.v);
  {
    sycl::buffer<float,1> ab(a.data(),sycl::range<1>(a.size())),vb(v.data(),sycl::range<1>(v.size())),ob(out.data(),sycl::range<1>(out.size()));
    queue.submit([&](sycl::handler& cgh){
      auto aa=ab.get_access<sycl::access::mode::read>(cgh),va=vb.get_access<sycl::access::mode::read>(cgh); auto oa=ob.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(out.size()),[=](sycl::id<1> id){
        int e=id[0]%s.v,row=id[0]/s.v,qi=row%s.q,bh=row/s.q; float value=0;
        for(int slot=0;slot<s.w;++slot) { int ki=qi-s.w/2+slot; if(ki>=0&&ki<s.k) value+=aa[row*s.w+slot]*va[(bh*s.k+ki)*s.v+e]; }
        oa[id]=value;
      });
    }); queue.wait_and_throw();
  } return out;
}
inline std::pair<std::vector<float>,std::vector<float>> local_weighted_average_backward(sycl::queue& queue,
 const std::vector<float>& a,const std::vector<float>& v,const std::vector<float>& grad,shape s) {
  s.validate(); check(a,s.b*s.h*s.q*s.w); check(v,s.b*s.h*s.k*s.v); check(grad,s.b*s.h*s.q*s.v);
  std::vector<float> da(a.size()),dv(v.size());
  {
    sycl::buffer<float,1> ab(a.data(),sycl::range<1>(a.size())),vb(v.data(),sycl::range<1>(v.size())),gb(grad.data(),sycl::range<1>(grad.size())),dab(da.data(),sycl::range<1>(da.size())),dvb(dv.data(),sycl::range<1>(dv.size()));
    queue.submit([&](sycl::handler& cgh){
      auto va=vb.get_access<sycl::access::mode::read>(cgh),ga=gb.get_access<sycl::access::mode::read>(cgh); auto dst=dab.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(da.size()),[=](sycl::id<1> id){
        int slot=id[0]%s.w,row=id[0]/s.w,qi=row%s.q,bh=row/s.q,ki=qi-s.w/2+slot; float value=0;
        if(ki>=0&&ki<s.k) for(int e=0;e<s.v;++e) value+=ga[row*s.v+e]*va[(bh*s.k+ki)*s.v+e];
        dst[id]=value;
      });
    });
    queue.submit([&](sycl::handler& cgh){
      auto aa=ab.get_access<sycl::access::mode::read>(cgh),ga=gb.get_access<sycl::access::mode::read>(cgh); auto dst=dvb.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(dv.size()),[=](sycl::id<1> id){
        int e=id[0]%s.v,ki=(id[0]/s.v)%s.k,bh=id[0]/(s.v*s.k); float value=0;
        for(int qi=0;qi<s.q;++qi) { int slot=ki-qi+s.w/2; if(slot>=0&&slot<s.w) value+=aa[(bh*s.q+qi)*s.w+slot]*ga[(bh*s.q+qi)*s.v+e]; }
        dst[id]=value;
      });
    }); queue.wait_and_throw();
  } return {da,dv};
}
}
