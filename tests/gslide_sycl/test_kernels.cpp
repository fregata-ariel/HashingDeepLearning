// TRACE_TEST_ID: GSLIDE-SYCL-DEVICE-115
#include "kernels.hpp"
#include "hashing/sycl_backend.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
using namespace gslide::sycl_port;
static int checks=0;
static void check(bool x,const char* msg){++checks;if(!x)throw std::runtime_error(msg);}
template<class T> struct memory {
 sycl::queue& q; T* p; int n;
 memory(sycl::queue& q,int n):q(q),p(sycl::malloc_shared<T>(std::max(n,1),q)),n(n){if(!p)throw std::bad_alloc();std::fill(p,p+n,T{});}
 ~memory(){sycl::free(p,q);} T& operator[](int i){return p[i];}
};
static unsigned oracle_random(unsigned seed,int id){std::uint64_t x=seed%2147483647u;if(!x)x=1;for(int i=0;i<=id;++i)x=x*48271%2147483647u;return x;}
static int oracle_address(const int* bins,const float* row,hash_shape z,int table){int a=0;for(int j=0;j<z.k;++j){int best=0;for(int k=1;k<z.bin;++k)if(row[bins[(table*z.k+j)*z.bin+k]]>row[bins[(table*z.k+j)*z.bin+best]])best=k;a+=best*int(std::pow(2,(z.k-1-j)*int(std::log(z.bin))));}return a;}
static void test_reductions(sycl::queue& q){
 for(int local:{32,64,128})for(int n:{1,31,33,65,129}){
  int extent=padded(n,local),groups=extent/local;memory<float>x(q,extent),s(q,groups),m(q,groups),ss(q,extent),sm(q,extent);memory<int>sgid(q,extent),sgbegin(q,extent),sgsize(q,extent);
  for(int i=0;i<n;++i)x[i]=float((i*17)%31-29)/8;
  reductions(q,x.p,n,s.p,m.p,ss.p,sm.p,local);
  q.parallel_for(sycl::nd_range<1>(extent,local),[=,a=sgid.p,b=sgbegin.p,c=sgsize.p](sycl::nd_item<1> it){int i=it.get_global_linear_id();auto sg=it.get_sub_group();a[i]=sg.get_group_linear_id();b[i]=it.get_group_linear_id();c[i]=sg.get_local_linear_range();}).wait_and_throw();
  for(int g=0;g<groups;++g){double sum=0;float maximum=-std::numeric_limits<float>::max();for(int i=g*local;i<std::min(n,(g+1)*local);++i){sum+=x[i];maximum=std::max(maximum,x[i]);}check(std::abs(s[g]-sum)<1e-5,"block sum");check(m[g]==maximum,"block max");}
  for(int i=0;i<extent;++i){double sum=0;float maximum=-std::numeric_limits<float>::max();for(int j=0;j<n;++j)if(sgbegin[j]==sgbegin[i]&&sgid[j]==sgid[i]){sum+=x[j];maximum=std::max(maximum,x[j]);}check(std::abs(ss[i]-sum)<1e-5,"subgroup sum");check(sm[i]==maximum,"subgroup max");}
 }
}
static void test_dense_rng(sycl::queue& q){
 for(int nodes:{0,1,3,65})for(int batch:{0,1,3}){int n=nodes*batch;memory<int>ids(q,n+1),off(q,batch+2);ids[n]=-71;off[batch+1]=-73;dense_ids(q,ids.p,off.p,nodes,batch,n);for(int i=0;i<n;++i)check(ids[i]==i%nodes,"dense ids");for(int i=0;i<=batch;++i)check(off[i]==i*nodes,"dense offsets");check(ids[n]==-71&&off[batch+1]==-73,"dense guards");}
 memory<int>i(q,1),o(q,2);bool rejected=false;try{dense_ids(q,i.p,o.p,2,1,1);}catch(std::invalid_argument&){rejected=true;}check(rejected,"dense reject capacity");
 for(unsigned seed:{0u,1u,40u,2147483647u}){memory<unsigned>a(q,131),b(q,131);random_keys(q,a.p,129,seed,7,false);random_keys(q,b.p,129,seed,7,true);std::set<unsigned>seen;for(int i=0;i<129;++i){auto x=oracle_random(seed,i);check(a[i]==x&&a[i]>0&&a[i]<2147483647u,"RNG minstd range/exact sequence");check(b[i]==(unsigned(i/7)<<16|(x&32768u)),"packed original precedence variant");seen.insert(a[i]);}check(seen.size()==129,"RNG nonconstant");random_keys(q,b.p,129,seed,7,false);for(int i=0;i<129;++i)check(a[i]==b[i],"RNG repeat");}
}
static void test_hash(sycl::queue& q){
 for(int nodes:{1,31,33,65})for(int per_thread:{1,2,4}){
  hash_shape z{7,nodes,5,2,4,4,per_thread,32,128};int nb=z.tables*z.buckets;
  memory<int>bins(q,z.tables*z.k*z.bin),buckets(q,nb*z.capacity),sizes(q,nb);memory<float>w(q,z.nodes*z.previous);
  for(int i=0;i<bins.n;++i)bins[i]=(i*3+i/4)%z.previous;
  for(int n=0;n<nodes;++n)for(int j=0;j<z.previous;++j)w[n*z.previous+j]=n%3==0?-5.f:float((n*7+j*3)%11)-12;
  build_hash(q,bins.p,w.p,z,buckets.p,sizes.p,64);
  std::vector<std::set<int>>expected(nb);
  for(int n=0;n<nodes;++n)for(int t=0;t<z.tables;++t)expected[t*z.buckets+oracle_address(bins.p,w.p+n*z.previous,z,t)].insert(n);
  for(int a=0;a<nb;++a){check(sizes[a]==int(expected[a].size()),"hash bucket count");std::set<int>actual(buckets.p+a*z.capacity,buckets.p+a*z.capacity+sizes[a]);check(actual==expected[a],"hash membership/ties/tail/lane staging");}
  // Query independently generated vectors: address and membership consistency.
  memory<int>query(q,z.tables);q.parallel_for(sycl::range<1>(z.tables),[=,bp=bins.p,wp=w.p,out=query.p](sycl::id<1> it){int t=it[0];out[t]=address(bp+t*z.k*z.bin,wp,z.k,z.bin,int(sycl::log(float(z.bin))));}).wait_and_throw();
  for(int t=0;t<z.tables;++t){check(query[t]==oracle_address(bins.p,w.p,z,t),"query address");check(expected[t*z.buckets+query[t]].count(0)==1,"query rebuilt membership");}
 }
 // Capacity boundary: no defined survivor order under overflow; count and
 // retained IDs are checked for a ring with multiple concurrent producers.
 hash_shape z{2,65,1,1,2,1,1,2,2};memory<int>bins(q,2),b(q,4),s(q,2);memory<float>w(q,130);bins[1]=1;build_hash(q,bins.p,w.p,z,b.p,s.p);check(s[0]==65,"ring total count");check(b[0]>=0&&b[0]<65&&b[1]>=0&&b[1]<65,"ring bounded IDs");
}
static void test_candidates(sycl::queue& q){
 constexpr int batch=65,nodes=7,L=3,threshold=2;
 memory<int>off(q,batch*L+1),raw(q,batch*6),counts(q,batch*nodes+1),labels(q,batch*2),loff(q,batch+1),perm(q,nodes),sizes(q,batch+1);
 std::vector<std::vector<int>>expected(batch,std::vector<int>(nodes));
 for(int b=0;b<batch;++b){off[b*L]=b*6;off[b*L+1]=b*6+3;off[b*L+2]=b*6+3;off[b*L+3]=b*6+6;for(int j=0;j<6;++j){int n=(b+j/2)%nodes;raw[b*6+j]=n;++expected[b][n];}loff[b]=b*2;labels[b*2]=(b+3)%nodes;labels[b*2+1]=(b+3)%nodes;}loff[batch]=batch*2;for(int n=0;n<nodes;++n)perm[n]=(n*3)%nodes;counts[batch*nodes]=-91;sizes[batch]=-92;
 count_candidates(q,raw.p,off.p,L,batch,nodes,batch*nodes,counts.p);
 for(int b=0;b<batch;++b)for(int n=0;n<nodes;++n)check(counts[b*nodes+n]==expected[b][n],"wrapper L/sample frequency");
 activate_labels(q,labels.p,loff.p,nullptr,batch,nodes,threshold,0,40,counts.p,sizes.p);
 for(int b=0;b<batch;++b){expected[b][(b+3)%nodes]+=4;int size=0;for(int n=0;n<nodes;++n){check(counts[b*nodes+n]==expected[b][n],"label wrapper duplicate count");size+=expected[b][n]>=threshold;}check(sizes[b]==size,"label-only size");}
 for(int floor:{0,5,7}){q.fill(counts.p,0,batch*nodes).wait_and_throw();activate_labels(q,labels.p,loff.p,perm.p,batch,nodes,threshold,floor,40,counts.p,sizes.p);
  for(int b=0;b<batch;++b){std::vector<int>ref(nodes);ref[(b+3)%nodes]=4;int size=1,i=oracle_random(40,b)%nodes;while(size<floor){int n=perm[i];if(ref[n]<threshold)++size;ref[n]+=threshold;i=(i+1)%nodes;}for(int n=0;n<nodes;++n)check(counts[b*nodes+n]==ref[n],"random padding fixed seed/cycle");check(sizes[b]==std::max(1,floor),"minimum floor");}
 }
 check(counts[batch*nodes]==-91&&sizes[batch]==-92,"candidate guards");bool rejected=false;try{activate_labels(q,labels.p,loff.p,perm.p,batch,nodes,threshold,8,40,counts.p,sizes.p);}catch(std::invalid_argument&){rejected=true;}check(rejected,"impossible floor reject");
 // Empty/partial samples and reuse after clear.
 q.fill(counts.p,0,batch*nodes).wait_and_throw();off[0]=off[1]=off[2]=off[3]=0;count_candidates(q,raw.p,off.p,L,1,nodes,batch*nodes,counts.p);loff[0]=loff[1]=0;activate_labels(q,labels.p,loff.p,nullptr,1,nodes,threshold,0,40,counts.p,sizes.p);check(sizes[0]==0,"empty reuse");count_candidates(q,raw.p,off.p,L,0,nodes,batch*nodes,counts.p);
}
int main(){try{auto q=hashing::sycl_backend::make_queue(hashing::sycl_backend::device_kind::cpu);hashing::sycl_backend::require_cpu(q);hashing::sycl_backend::describe_queue(std::cout,q);test_reductions(q);test_dense_rng(q);test_hash(q);test_candidates(q);std::cout<<"GSLIDE_SYCL_KERNELS checks="<<checks<<" original_cuda_executed=false gpu_validated=false\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
