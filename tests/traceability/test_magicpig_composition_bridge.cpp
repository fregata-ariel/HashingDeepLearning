// TRACE_TEST_ID: MAGICPIG-COMPOSITION-BRIDGE-CPU
// Selected maintained bodies are emitted by the common driver. The protocol
// connects actual LSH results directly to actual corrected QK/Softmax/WV.
#include <iomanip>
#include <numeric>
#include <sstream>
namespace composition {
constexpr int D=128,C=32,K=1,L=2;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class T> void read(std::istream& in, std::vector<T>& a) {
    for(auto& x:a) require(bool(in>>x),"truncated bridge payload");
}
template<class T> void json(const std::vector<T>& a) {
    std::cout<<'['; for(size_t i=0;i<a.size();++i) {if(i) std::cout<<','; std::cout<<a[i];} std::cout<<']';
}
struct Bridge {
    int H,N=0; LSH lsh; SparseAttentionServer sparse,dense;
    std::vector<bfloat16> keys,values; std::vector<float> norms;
    explicit Bridge(int h):H(h),keys(C*D),values(C*D),norms(C) {
        require(h==1||h==4||h==8,"unsupported GQA");
        lsh.alloc(K,L,1,H,1,1,C); sparse.alloc(1,H,1,D,1,C); dense.alloc(1,H,1,D,1,C);
    }
    void fill(SparseAttentionServer& server, int n, const std::vector<float>& k, const std::vector<float>& v,
              std::vector<bfloat16>& kb,std::vector<bfloat16>& vb,std::vector<float>& kn) {
        require(n>0&&n<=C&&k.size()==size_t(n*D)&&v.size()==k.size(),"fill dimensions");
        std::fill(kb.begin(),kb.end(),0);std::fill(vb.begin(),vb.end(),0);std::fill(kn.begin(),kn.end(),0);
        for(int t=0;t<n;++t) {double sum=0;for(int d=0;d<D;++d) {
            kb[t*D+d]=magicpig_attention_model::encode(k[t*D+d]);
            vb[t*D+d]=magicpig_attention_model::encode(v[t*D+d]);
            double x=magicpig_attention_model::decode(kb[t*D+d]);sum+=x*x;
        } kn[t]=float(std::sqrt(sum));}
        server.fill(0,0,torch::Tensor(kb.data(),{1,C,D}),torch::Tensor(vb.data(),{1,C,D}),torch::Tensor(kn.data(),{1,C}));
    }
    void load(int n,const std::vector<int>& codes,const std::vector<float>& k,const std::vector<float>& v) {
        require(n>0&&n<=C&&codes.size()==size_t(L*n),"load dimensions");N=n;
        std::vector<int16_t> sorted(L*n);std::vector<int> order(L*n);
        for(int l=0;l<L;++l) {
            std::vector<int> ids(n);std::iota(ids.begin(),ids.end(),0);
            std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return codes[l*n+a]<codes[l*n+b];});
            for(int j=0;j<n;++j){require(codes[l*n+ids[j]]>=0&&codes[l*n+ids[j]]<2,"hash domain");sorted[l*n+j]=int16_t(codes[l*n+ids[j]]);order[l*n+j]=ids[j];}
        }
        lsh.fill(0,0,torch::Tensor(sorted.data(),{1,L,n}),torch::Tensor(order.data(),{1,L,n}));
        fill(sparse,n,k,v,keys,values,norms);
    }
    void lookup(const std::vector<int>& codes,std::vector<int>& ids,std::vector<int>& nnz) {
        require(codes.size()==size_t(H*L),"query codes shape");
        for(auto c:codes) require(c>=0&&c<2,"query hash domain");
        lsh.batch_retrieve(0,torch::Tensor(const_cast<int*>(codes.data()),{H,L}),torch::Tensor(ids.data(),{H,C}),torch::Tensor(nnz.data(),{H}));
    }
    void query(int mode,int n,const std::vector<int>& codes,const std::vector<float>& q,
               const std::vector<float>& dk,const std::vector<float>& dv) {
        require(mode>=0&&mode<=2&&q.size()==size_t(H*D),"query mode/shape");
        std::vector<int> ids(H*C,0),nnz(H),dense_nnz(H,n);lookup(codes,ids,nnz);
        for(int h=0;h<H;++h) require(nnz[h]>0&&nnz[h]==nnz[0],"composition requires nonempty uniform nnz");
        if(mode==1) for(int h=0;h<H;++h) require(nnz[h]==N,"full mode requires actual full retrieval");
        std::vector<float> query=q,qn(H),sm(2*H),dm(2*H),qk(H*C),logits(H*C);
        std::vector<bfloat16> so(H*D),doo(H*D),dkey(C*D),dvalue(C*D);std::vector<float> dnorm(C);
        for(int h=0;h<H;++h) {
            double sum=0;for(int d=0;d<D;++d) sum+=double(q[h*D+d])*q[h*D+d];qn[h]=float(std::sqrt(sum));
            qk_kernel(keys.data(),ids.data()+h*C,query.data()+h*D,qk.data()+h*C,D,nnz[h]);
            std::copy_n(qk.data()+h*C,C,logits.data()+h*C);
            if(mode==0) transform_kernel(logits.data()+h*C,nnz[h],qn[h],norms.data(),K,L,std::sqrt(float(D)),ids.data()+h*C);
            else for(int j=0;j<nnz[h];++j) logits[h*C+j]/=std::sqrt(float(D));
        }
        const auto out=torch::Tensor(so.data(),{H,D}),meta=torch::Tensor(sm.data(),{2,H});
        const auto qt=torch::Tensor(query.data(),{H,D}),counts=torch::Tensor(nnz.data(),{H});
        if(mode==0) sparse.attention(0,K,L,out,meta,qt,torch::Tensor(qn.data(),{H}),torch::Tensor(ids.data(),{H,C}),counts);
        else if(mode==1) { // Explicit dense baseline: known optimized-tail defect #72 is excluded.
            auto* score=static_cast<float*>(sparse.get_score().data_ptr());
            for(int h=0;h<H;++h) {
                qk_kernel(keys.data(),ids.data()+h*C,query.data()+h*D,score+h*C,D,nnz[h]);
                softmax_kernel_full(score+h*C,nnz[h],std::sqrt(float(D)),sm.data()+h,sm.data()+H+h);
                wv_kernel(values.data(),ids.data()+h*C,score+h*C,so.data()+h*D,D,nnz[h]);
            }
        }
        else { // Deliberate mutation: actual selected QK/Softmax/WV with correction omitted.
            auto* score=static_cast<float*>(sparse.get_score().data_ptr());
            for(int h=0;h<H;++h) {
                std::copy_n(logits.data()+h*C,C,score+h*C);softmax_kernel(score+h*C,nnz[h],sm.data()+h,sm.data()+H+h);
                wv_kernel(values.data(),ids.data()+h*C,score+h*C,so.data()+h*D,D,nnz[h]);
            }
        }
        fill(dense,n,dk,dv,dkey,dvalue,dnorm);
        auto* dense_score=static_cast<float*>(dense.get_score().data_ptr());
        qk_kernel_full(dkey.data(),query.data(),dense_score,H,D,n,C);
        for(int h=0;h<H;++h) softmax_kernel_full(dense_score+h*C,n,std::sqrt(float(D)),dm.data()+h,dm.data()+H+h);
        wv_kernel_dim128_full(dvalue.data(),dense_score,doo.data(),D,C,n,H);
        std::vector<int> selected; std::vector<float> dots,corrected,weights,sp(H*D),dp(H*D),sl(H),dl(H);
        auto* score=static_cast<float*>(sparse.get_score().data_ptr());
        for(int h=0;h<H;++h) {
            for(int j=0;j<nnz[h];++j){selected.push_back(ids[h*C+j]);dots.push_back(qk[h*C+j]);corrected.push_back(logits[h*C+j]);weights.push_back(score[h*C+j]);}
            for(int d=0;d<D;++d){sp[h*D+d]=magicpig_attention_model::decode(so[h*D+d]);dp[h*D+d]=magicpig_attention_model::decode(doo[h*D+d]);}
            sl[h]=sm[H+h];dl[h]=dm[H+h];
        }
        std::cout<<"{\"counts\":";json(nnz);std::cout<<",\"selected\":";json(selected);std::cout<<",\"qk\":";json(dots);std::cout<<",\"logits\":";json(corrected);
        std::cout<<",\"weights\":";json(weights);std::cout<<",\"sparse\":";json(sp);std::cout<<",\"dense\":";json(dp);std::cout<<",\"sparse_lse\":";json(sl);std::cout<<",\"dense_lse\":";json(dl);std::cout<<"}\n"<<std::flush;
    }
    void serve(std::istream& in) {
        std::string cmd;
        while(in>>cmd) {
            if(cmd=="LOAD") {int n;require(bool(in>>n)&&n>0&&n<=C,"load count");std::vector<int> c(L*n);std::vector<float> k(n*D),v(n*D);read(in,c);read(in,k);read(in,v);load(n,c,k,v);std::cout<<"{\"loaded\":"<<n<<"}\n"<<std::flush;}
            else if(cmd=="QUERY") {int mode,n;require(bool(in>>mode>>n)&&n>0&&n<=C,"query count");std::vector<int> c(H*L);std::vector<float> q(H*D),k(n*D),v(n*D);read(in,c);read(in,q);read(in,k);read(in,v);query(mode,n,c,q,k,v);}
            else if(cmd=="CLEAR_LSH") {lsh.clear();std::cout<<"{\"cleared_lsh\":true}\n"<<std::flush;}
            else if(cmd=="CLEAR_ATTN") {
                sparse.clear();dense.clear();
                for(auto* server:{&sparse,&dense}) {
                    auto* k=static_cast<bfloat16*>(server->get_key_cache(0).data_ptr());
                    auto* v=static_cast<bfloat16*>(server->get_value_cache(0).data_ptr());
                    auto* kn=static_cast<float*>(server->get_key_norm(0).data_ptr());
                    auto* score=static_cast<float*>(server->get_score().data_ptr());
                    require(std::all_of(k,k+C*D,[](auto x){return x==0;})&&std::all_of(v,v+C*D,[](auto x){return x==0;}),"actual clear KV cache");
                    require(std::all_of(kn,kn+C,[](auto x){return x==0;})&&std::all_of(score,score+H*C,[](auto x){return x==0;}),"actual clear norms/scores");
                }
                std::cout<<"{\"cleared_attention\":true}\n"<<std::flush;}
            else if(cmd=="LOOKUP") {std::vector<int> c(H*L),ids(H*C),nnz(H);read(in,c);lookup(c,ids,nnz);std::cout<<"{\"counts\":";json(nnz);std::cout<<"}\n"<<std::flush;}
            else throw std::runtime_error("unknown bridge command");
        }
    }
};
int selftest() {
    Bridge b(4);std::vector<int> codes{0,0,1,1,0,0,1,1};std::vector<float> k(4*D),v(4*D);
    for(int t=0;t<4;++t){k[t*D]=float(t-1.5);k[t*D+1]=float(t%2)+.5f;v[t*D]=float(t)/8;}
    b.load(4,codes,k,v);std::vector<int> qcodes(8,0),ids(4*C),counts(4);b.lookup(qcodes,ids,counts);
    require(std::all_of(counts.begin(),counts.end(),[](int n){return n==2;}),"selftest retrieval");
    for(int h=0;h<4;++h) require(ids[h*C]==0&&ids[h*C+1]==1,"selftest exact set");
    std::vector<float> q(4*D);for(int h=0;h<4;++h){q[h*D]=-1;q[h*D+1]=.25f;}
    std::ostringstream capture;auto* original=std::cout.rdbuf(capture.rdbuf());
    b.query(0,4,qcodes,q,k,v);std::cout.rdbuf(original);
    require(capture.str().find("\"weights\"")!=std::string::npos,"selftest attention output");b.lsh.clear();b.sparse.clear();b.dense.clear();b.lookup(qcodes,ids,counts);
    require(std::all_of(counts.begin(),counts.end(),[](int n){return n==0;}),"selftest clear");
    std::reverse(codes.begin(),codes.end());b.load(4,codes,k,v);b.lookup(qcodes,ids,counts);
    require(ids[0]==2&&ids[1]==3,"selftest refill");
    std::cout<<"MAGICPIG-COMPOSITION-BRIDGE-CPU: PASS exact_sets=8 clear_reuse=1 native_SIMD=unexecuted\n";return 0;
}
}
int main(int argc,char** argv) {
    try {std::cout<<std::setprecision(12);if(argc==3&&std::string(argv[1])=="--composition-bridge") {composition::Bridge b(std::stoi(argv[2]));b.serve(std::cin);return 0;}return composition::selftest();}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
