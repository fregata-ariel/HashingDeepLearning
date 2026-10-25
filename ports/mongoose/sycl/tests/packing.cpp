// SPDX-License-Identifier: MIT
// TRACE_TEST_ID: MONGOOSE-PACKING-SYCL-CPU
#include <mongoose/simhash.hpp>
#include <hashing/sycl_backend.hpp>
#include "LSH.h"
#include <iostream>
#include <stdexcept>
static void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
int main() {
 try {
  auto queue=hashing::sycl_backend::make_validation_queue();
  hashing::sycl_backend::describe_queue(std::cerr,queue);
  int cases=0;
  // Literal independent expectations: bit0 is first projection, zero is false.
  const std::vector<float> literal={1,0,-1,1, -1,2,3,0, 0,0,0,0, 1,1,1,1};
  require(hashing::mongoose::pack(queue,literal,2,4,2)==std::vector<std::int32_t>({9,6,0,15}),"literal codes");
  require(hashing::mongoose::pack(queue,literal,2,4,2,true)==std::vector<std::int32_t>({9,0,6,15}),"literal transpose");
  for(int k:{1,2,7,31,32}) {
    int n=3,tables=2,d=3; std::vector<float> x={1,0,0, 0,1,0, 1,-1,1};
    std::vector<float> w(d*k*tables);
    for(int e=0;e<d;++e) for(int c=0;c<k*tables;++c) w[e*k*tables+c]=float((c+2*e)%3-1);
    std::vector<std::int32_t> expected;
    // Independent FP64 scalar projection and arithmetic code (no production helper).
    for(int row=0;row<n;++row) for(int table=0;table<tables;++table) {
      std::int64_t code=0,power=1;
      for(int bit=0;bit<k;++bit,power*=2) {
        double projection=0; for(int e=0;e<d;++e) projection+=double(x[row*d+e])*w[e*k*tables+table*k+bit];
        if(projection>0) code+=power;
      }
      if(code>=2147483648LL) code-=4294967296LL;
      expected.push_back(std::int32_t(code));
    }
    auto actual=hashing::mongoose::hash(queue,x,w,n,d,k,tables);
    require(actual==expected,"projection packing oracle");
    auto transposed=hashing::mongoose::hash(queue,x,w,n,d,k,tables,true);
    for(int row=0;row<n;++row) for(int table=0;table<tables;++table)
      require(transposed[table*n+row]==expected[row*tables+table],"projection transpose");
    // Execute maintained native table, with SYCL-produced int32 keys.
    LSH native(k,tables,1); native.insert_multi(actual.data(),n);
    for(int row=0;row<n;++row) {
      auto found=native.query(actual.data()+row*tables);
      std::unordered_set<int> oracle;
      for(int item=0;item<n;++item) for(int table=0;table<tables;++table)
        if(expected[item*tables+table]==expected[row*tables+table]) oracle.insert(item);
      require(found==oracle,"native LSH bucket oracle");
    }
    ++cases;
  }
  for(int bad:{0,33,63}) {
    bool rejected=false; try { hashing::mongoose::pack(queue,{1},1,bad,1); } catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"k rejection");
  }
  auto boundary=hashing::mongoose::pack(queue,std::vector<float>(32,1),1,32,1);
  require(boundary[0]==-1,"int32 all bits");
  std::cout<<"{\"status\":\"passed\",\"id\":\""<<(queue.get_device().is_gpu()?"MONGOOSE-PACKING-SYCL-NVIDIA":"MONGOOSE-PACKING-SYCL-CPU")<<"\",\"projection_cases\":"<<cases
    <<",\"native_lsh\":true,\"real_cpu\":"<<std::boolalpha<<queue.get_device().is_cpu()
    <<",\"original_cuda_executed\":false,\"native_gpu_executed\":"<<queue.get_device().is_gpu()<<"}\n";
 } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
