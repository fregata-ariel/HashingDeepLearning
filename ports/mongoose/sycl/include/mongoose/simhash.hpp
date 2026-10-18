// SPDX-License-Identifier: MIT
// TRACE_TEST_ID: MONGOOSE-PACKING-SYCL-CPU
// Relation: support for released sign-packing/native-LSH infrastructure.
// Derived from HazyResearch MONGOOSE sign packing; see ../SYCL_CONTRACT.md.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <limits>
#include <cmath>

namespace hashing::mongoose {
inline void dimensions(int n, int k, int tables) {
  if(n <= 0 || k < 1 || k > 32 || tables <= 0 ||
     std::uint64_t(n)*k*tables > std::numeric_limits<int>::max())
    throw std::invalid_argument("SimHash requires N,L > 0, 1 <= k <= 32 and int-sized storage");
}
// Synchronous, value-owned API. Buffers borrow host storage until wait_and_throw
// and buffer destruction copy results back. No device pointer escapes the call.
inline std::vector<std::int32_t> pack(sycl::queue& queue,
    const std::vector<float>& projection, int n, int k, int tables, bool transpose=false) {
  dimensions(n,k,tables);
  if(projection.size()!=std::size_t(n)*k*tables) throw std::invalid_argument("projection shape");
  for(float x: projection) if(!std::isfinite(x)) throw std::invalid_argument("finite projection required");
  std::vector<std::uint32_t> bits(n*tables);
  {
    sycl::buffer<float,1> input(projection.data(),sycl::range<1>(projection.size()));
    sycl::buffer<std::uint32_t,1> output(bits.data(),sycl::range<1>(bits.size()));
    queue.submit([&](sycl::handler& cgh) {
      auto src=input.get_access<sycl::access::mode::read>(cgh);
      auto dst=output.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(bits.size()), [=](sycl::id<1> id) {
        std::uint32_t value=0;
        // One work-item owns a code; no warp-width or subgroup assumption.
        for(int bit=0;bit<k;++bit) if(src[id[0]*k+bit]>0) value|=std::uint32_t(1)<<bit;
        dst[id]=value;
      });
    });
    queue.wait_and_throw();
  }
  std::vector<std::int32_t> result(bits.size());
  for(int row=0;row<n;++row) for(int table=0;table<tables;++table) {
    std::int32_t signed_bits;
    std::memcpy(&signed_bits,&bits[row*tables+table],sizeof(signed_bits));
    result[transpose ? table*n+row : row*tables+table]=signed_bits;
  }
  return result;
}
inline std::vector<std::int32_t> hash(sycl::queue& queue,
    const std::vector<float>& data, const std::vector<float>& weights,
    int n, int d, int k, int tables, bool transpose=false) {
  dimensions(n,k,tables);
  if(d<=0 || data.size()!=std::size_t(n)*d || weights.size()!=std::size_t(d)*k*tables)
    throw std::invalid_argument("data/weight shape");
  for(float x:data) if(!std::isfinite(x)) throw std::invalid_argument("finite data required");
  for(float x:weights) if(!std::isfinite(x)) throw std::invalid_argument("finite weights required");
  std::vector<float> projected(std::size_t(n)*k*tables);
  {
    sycl::buffer<float,1> a(data.data(),sycl::range<1>(data.size()));
    sycl::buffer<float,1> b(weights.data(),sycl::range<1>(weights.size()));
    sycl::buffer<float,1> p(projected.data(),sycl::range<1>(projected.size()));
    queue.submit([&](sycl::handler& cgh) {
      auto x=a.get_access<sycl::access::mode::read>(cgh);
      auto w=b.get_access<sycl::access::mode::read>(cgh);
      auto out=p.get_access<sycl::access::mode::write>(cgh);
      cgh.parallel_for(sycl::range<1>(projected.size()),[=](sycl::id<1> id) {
        int row=id[0]/(k*tables), col=id[0]%(k*tables); float value=0;
        for(int e=0;e<d;++e) value+=x[row*d+e]*w[e*k*tables+col];
        out[id]=value;
      });
    }); queue.wait_and_throw();
  }
  return pack(queue,projected,n,k,tables,transpose);
}
}
