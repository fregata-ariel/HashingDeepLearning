// Actual qk_kernel_bf16_impl is extracted before this fixture.
// bfloat16 storage is uint16_t, matching archived FBGEMM Types.h; no Torch or
// FBGEMM library executes. This checks exactly16 rows,32 coordinates/one tile.
// TRACE_TEST_ID: MAGICPIG-BF16-SELECTED-CPU
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
int main() {
  constexpr int rows=16, dim=32;
  std::array<bfloat16,rows*dim> keys{};
  std::array<bfloat16,dim> query{};
  std::array<float,rows*dim> dense_keys{};
  std::array<float,dim> dense_query{};
  auto encode=[](float x){ std::uint32_t bits; std::memcpy(&bits,&x,4); return static_cast<bfloat16>(bits>>16); };
  for(int d=0;d<dim;++d) { dense_query[d]=.125f*((d%9)-4); query[d]=encode(dense_query[d]); }
  for(int r=0;r<rows;++r)for(int d=0;d<dim;++d) {
    dense_keys[r*dim+d]=.0625f*(((r*3+d*7)%17)-8);
    keys[r*dim+d]=encode(dense_keys[r*dim+d]);
  }
  std::array<int,rows> ids{}; for(int r=0;r<rows;++r)ids[r]=(r*7)%rows;
  std::array<float,rows+2> guarded{}; guarded.front()=321.25f; guarded.back()=-123.5f;
  qk_kernel_bf16_impl(keys.data(),ids.data(),query.data(),guarded.data()+1,dim,rows);
  double maximum=0;
  for(int r=0;r<rows;++r){
    double expected=0; for(int d=0;d<dim;++d)expected+=static_cast<double>(dense_query[d])*dense_keys[ids[r]*dim+d];
    double error=std::abs(guarded[r+1]-expected); maximum=std::max(maximum,error);
    if(!std::isfinite(guarded[r+1])||error>1e-6) { std::cerr<<"BF16 dense dot mismatch\n"; return 1; }
  }
  if(guarded.front()!=321.25f||guarded.back()!=-123.5f)return 2;
  std::cout<<"MAGICPIG_BF16_BODY_PASS rows=16 head_dim=32 max_abs_error="<<maximum<<"\n";
}
