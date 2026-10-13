// TRACE_TEST_ID: MAGICPIG-FASTFILL-REPAIR-CPU
// Actual maintained fill/fastfill/retrieve bodies; independent token-table oracle.
#include <array>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <utility>
namespace {
constexpr int kv_heads=2, heads=4, tables=4, layers=2, batches=2, capacity=35;
constexpr int sentinel=-1234567;
using Codes=std::vector<int>;
void require(bool condition,const std::string& message) {
    if(!condition) throw std::runtime_error(message);
}
Codes make_codes(int length,int layer,int request,int pattern,int generation) {
    Codes codes(std::max(1,kv_heads*tables*length),0);
    for(int h=0;h<kv_heads;++h) for(int t=0;t<tables;++t) for(int s=0;s<length;++s) {
        int code=0;
        if(pattern==1) {
            int hits=(s+h+2*request+layer+generation)%5;
            code=t<hits ? (h+request+generation)%4 : (h+request+generation+1+t)%4;
        } else if(pattern==2) code=(s*3+h+t*2+request+layer+generation)%4;
        codes[(h*tables+t)*length+s]=code;
    }
    return codes;
}
std::vector<int> oracle(const Codes& codes,int length,int kv,const int* query) {
    std::vector<int> result;
    for(int s=0;s<length;++s) {
        int count=0;
        for(int t=0;t<tables;++t) count+=codes[(kv*tables+t)*length+s]==query[t];
        if(count>=2) result.push_back(s);
    }
    return result;
}
void sorted_fill(LSH& owner,int layer,int request,const Codes& codes,int length) {
    std::vector<int16_t> hashes(std::max(1,kv_heads*tables*length));
    std::vector<int> ids(hashes.size());
    for(int h=0;h<kv_heads;++h) for(int t=0;t<tables;++t) {
        std::vector<std::pair<int,int>> sorted;
        for(int s=0;s<length;++s) sorted.emplace_back(codes[(h*tables+t)*length+s],s);
        std::sort(sorted.begin(),sorted.end());
        for(int s=0;s<length;++s) {
            hashes[(h*tables+t)*length+s]=static_cast<int16_t>(sorted[s].first);
            ids[(h*tables+t)*length+s]=sorted[s].second;
        }
    }
    owner.fill(layer,request,torch::Tensor(hashes.data(),{kv_heads,tables,length}),
               torch::Tensor(ids.data(),{kv_heads,tables,length}));
}
void fast_fill(LSH& owner,int layer,int request,const Codes& codes,int length) {
    // Caller-owned temporary input is changed and freed before queries.
    Codes input=codes;
    owner.fastfill(layer,request,torch::Tensor(input.data(),{kv_heads,tables,length}));
    std::fill(input.begin(),input.end(),-1);
}
std::pair<std::vector<int>,std::vector<int>> retrieve(LSH& owner,int layer,const Codes& queries) {
    std::vector<int> output(2+batches*heads*capacity,sentinel);
    std::vector<int> counts(batches*heads,-1);
    Codes q=queries;
    owner.batch_retrieve(layer,torch::Tensor(q.data(),{batches*heads,tables}),
                        torch::Tensor(output.data()+1,{batches*heads,capacity}),
                        torch::Tensor(counts.data(),{batches*heads}));
    require(output.front()==sentinel && output.back()==sentinel,"retrieval outer canaries");
    return {output,counts};
}
size_t parity_checks() {
    size_t checks=0;
    for(int length:{0,1,2,15,16,17,31,32,33,capacity}) for(int pattern=0;pattern<3;++pattern)
        for(int lifetime=0;lifetime<2;++lifetime) {
            LSH fast,sorted;
            fast.alloc(2,tables,layers,heads,kv_heads,batches,capacity);
            sorted.alloc(2,tables,layers,heads,kv_heads,batches,capacity);
            // Consecutive fastfill calls replace a shorter/empty request without
            // requiring clear. Sorted fill retains its documented clear policy.
            const std::array<int,4> lengths{length,length/2,0,length};
            for(int generation=0;generation<4;++generation) {
                int n=lengths[generation];
                sorted.clear();
                std::vector<Codes> all;
                for(int layer=0;layer<layers;++layer) for(int request=0;request<batches;++request) {
                    all.push_back(make_codes(n,layer,request,pattern,generation));
                    fast_fill(fast,layer,request,all.back(),n);
                    sorted_fill(sorted,layer,request,all.back(),n);
                }
                for(int query_kind=0;query_kind<4;++query_kind) for(int layer=0;layer<layers;++layer) {
                    Codes queries(batches*heads*tables);
                    for(int head=0;head<batches*heads;++head) for(int t=0;t<tables;++t)
                        queries[head*tables+t]=(query_kind+head+(query_kind==3 ? t : 0))%4;
                    auto a=retrieve(fast,layer,queries),b=retrieve(sorted,layer,queries);
                    for(int head=0;head<batches*heads;++head) {
                        int request=head/heads,kv=(head%heads)/(heads/kv_heads);
                        auto want=oracle(all[layer*batches+request],n,kv,queries.data()+head*tables);
                        for(const auto* state:{&a,&b}) {
                            int count=state->second[head];
                            require(count>=0 && count<=n,"candidate count within logical length");
                            const int* row=state->first.data()+1+head*capacity;
                            std::vector<int> got(row,row+count);
                            require(std::set<int>(got.begin(),got.end()).size()==got.size(),"duplicate token ID");
                            std::sort(got.begin(),got.end());
                            require(got==want,"independent distinct-table candidate set mismatch");
                            for(int s=count;s<capacity;++s) require(row[s]==sentinel,"output tail canary");
                        }
                        require(a.second[head]==b.second[head],"fill/fastfill candidate count parity");
                        ++checks;
                    }
                }
                if(generation==1) {
                    fast.clear();
                    Codes queries(batches*heads*tables,0);
                    for(int layer=0;layer<layers;++layer) {
                        auto result=retrieve(fast,layer,queries);
                        require(std::all_of(result.second.begin(),result.second.end(),[](int x){return x==0;}),
                                "clear retained buckets");
                        require(std::all_of(result.first.begin(),result.first.end(),[](int x){return x==sentinel;}),
                                "empty retrieval changed output");
                    }
                }
            }
        }
    return checks;
}
size_t rejection_checks() {
    LSH owner;
    Codes input(kv_heads*tables*4,0),queries(batches*heads*tables,0);
    size_t checks=0;
    auto rejected=[&](auto operation,const std::string& diagnostic) {
        try { operation(); }
        catch(const std::invalid_argument& error) {
            require(std::string(error.what())==diagnostic,"unexpected boundary diagnostic");
            ++checks;return;
        }
        throw std::runtime_error("invalid fastfill input accepted");
    };
    rejected([&]{owner.fastfill(0,0,torch::Tensor(input.data(),{kv_heads,tables,4}));},
             "fastfill requires an allocated LSH owner");
    owner.alloc(2,tables,layers,heads,kv_heads,batches,capacity);
    for(int layer=0;layer<layers;++layer) for(int request=0;request<batches;++request)
        fast_fill(owner,layer,request,input,4);
    std::vector<std::pair<Codes,Codes>> before;
    for(int layer=0;layer<layers;++layer) before.push_back(retrieve(owner,layer,queries));
    rejected([&]{owner.fastfill(-1,0,torch::Tensor(input.data(),{kv_heads,tables,4}));},"invalid fastfill layer");
    rejected([&]{owner.fastfill(layers,0,torch::Tensor(input.data(),{kv_heads,tables,4}));},"invalid fastfill layer");
    rejected([&]{owner.fastfill(0,-1,torch::Tensor(input.data(),{kv_heads,tables,4}));},"invalid fastfill request");
    rejected([&]{owner.fastfill(0,batches,torch::Tensor(input.data(),{kv_heads,tables,4}));},"invalid fastfill request");
    rejected([&]{owner.fastfill(0,0,torch::Tensor(input.data(),{tables,4}));},"fastfill codes must have rank 3");
    rejected([&]{owner.fastfill(0,0,torch::Tensor(input.data(),{kv_heads,tables,4},torch::kInt8));},"fastfill codes must be int32");
    rejected([&]{owner.fastfill(0,0,torch::Tensor(input.data(),{kv_heads,tables,4},torch::kInt32,false));},"fastfill codes must be contiguous CPU tensors");
    rejected([&]{owner.fastfill(0,0,torch::Tensor(input.data(),{kv_heads,tables,4},torch::kInt32,true,false));},"fastfill codes must be contiguous CPU tensors");
    rejected([&]{owner.fastfill(0,0,torch::Tensor(input.data(),{1,tables,4}));},"fastfill code head/table shape mismatch");
    rejected([&]{owner.fastfill(0,0,torch::Tensor(input.data(),{kv_heads,3,4}));},"fastfill code head/table shape mismatch");
    rejected([&]{owner.fastfill(0,0,torch::Tensor(input.data(),{kv_heads,tables,capacity+1}));},"fastfill sequence exceeds capacity");
    for(int invalid:{-1,4}) {
        // Last element of the last head/table demonstrates whole-input validation
        // before any target bucket is cleared or partially rebuilt.
        Codes invalid_input=input;invalid_input.back()=invalid;
        rejected([&]{owner.fastfill(0,0,torch::Tensor(invalid_input.data(),{kv_heads,tables,4}));},
                 "fastfill hash code outside bucket range");
    }
    for(int layer=0;layer<layers;++layer) require(retrieve(owner,layer,queries)==before[layer],
                                                "rejected input mutated valid owner state");
    return checks;
}
}
int main() {
    try {
        size_t checks=parity_checks(),rejections=rejection_checks();
        require(checks==15360 && rejections==14,"fixture count changed");
        std::cout<<"MAGICPIG-FASTFILL-REPAIR-CPU: "<<checks<<" independent batched head parity checks; "
                 <<rejections<<" pre-mutation boundary rejections; copied inputs, replacement, clear/reuse and teardown passed\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
