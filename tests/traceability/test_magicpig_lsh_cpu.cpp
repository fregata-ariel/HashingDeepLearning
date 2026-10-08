// TRACE_TEST_ID: MAGICPIG-LSH-RETRIEVAL-CPU
// TRACE_TEST_ID: MAGICPIG-FASTFILL-DEFECT-CPU
// At-least-two distinct tables, not repeated membership in a single table.
#include <array>
#include <iostream>
#include <set>
#include <string>
#include <utility>
static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
namespace {
constexpr int tables=4, kv_heads=2, heads=4, batches=2, layers=2, capacity=35;
constexpr int sentinel=-1234567;
using Codes = std::vector<int>;
Codes codes_for(int length, int layer, int request, int generation) {
    Codes codes(kv_heads*tables*length);
    // Fixed exhaustive 0/1/2/3/4 collision patterns. Different heads/requests
    // and layers rotate the pattern, preserving independent expected sets.
    for(int h=0;h<kv_heads;++h) for(int t=0;t<tables;++t) for(int s=0;s<length;++s) {
        int hits=(s+2*h+request+layer+generation)%5;
        codes[(h*tables+t)*length+s] = t<hits ? 0 : 1;
    }
    return codes;
}
std::vector<int> expected(const Codes& codes,int length,int kv,const int* query) {
    std::vector<int> accepted;
    // Independent literal table matrix oracle: examine every token once and
    // count distinct table equality; no bucket boundaries, mask or sort used.
    for(int token=0;token<length;++token) {
        int hits=0;
        for(int table=0;table<tables;++table)
            hits += codes[(kv*tables+table)*length+token]==query[table];
        if(hits>=2) accepted.push_back(token);
    }
    return accepted;
}
void populate(LSH& lsh,int layer,int request,const Codes& codes,int length) {
    std::vector<int16_t> sorted(kv_heads*tables*length);
    std::vector<int> indices(sorted.size());
    for(int h=0;h<kv_heads;++h) for(int t=0;t<tables;++t) {
        std::vector<std::pair<int,int>> pairs;
        for(int s=0;s<length;++s) pairs.emplace_back(codes[(h*tables+t)*length+s],s);
        std::sort(pairs.begin(),pairs.end());
        for(int s=0;s<length;++s) {
            sorted[(h*tables+t)*length+s]=static_cast<int16_t>(pairs[s].first);
            indices[(h*tables+t)*length+s]=pairs[s].second;
        }
    }
    lsh.fill(layer,request,torch::Tensor(sorted.data(),{kv_heads,tables,length}),
             torch::Tensor(indices.data(),{kv_heads,tables,length}));
    // Inputs are caller-owned and copied, not aliased. Mutation followed by
    // deallocation must not affect retrieval from the LSH-owned table.
    std::fill(sorted.begin(),sorted.end(),static_cast<int16_t>(-1));
    std::fill(indices.begin(),indices.end(),-1);
}
void check_results(const std::vector<int>& guarded,const std::vector<int>& nnz,
    const std::vector<Codes>& all_codes,int length,const std::vector<int>& queries,int layer) {
    require(guarded.front()==sentinel && guarded.back()==sentinel,"output boundary canary");
    for(int head=0;head<batches*heads;++head) {
        int request=head/heads, kv=(head%heads)/(heads/kv_heads);
        auto want=expected(all_codes[layer*batches+request],length,kv,queries.data()+head*tables);
        require(nnz[head]>=0 && nnz[head]<=length,"nnz bounded by real sequence");
        const int* row=guarded.data()+1+head*capacity;
        std::vector<int> got(row,row+nnz[head]);
        auto distinct=std::set<int>(got.begin(),got.end());
        require(distinct.size()==got.size(),"deduplicate 3+ hits");
        std::sort(got.begin(),got.end());
        require(got==want,"dense distinct-table oracle mismatch");
        for(int i=nnz[head];i<capacity;++i) require(row[i]==sentinel,"output row tail changed");
    }
}
}
int main(int argc, char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--fastfill-probe") {
            // Execute the unmodified incomplete path in a dedicated process.
            // Normal sanitizer success never runs this known-leaking path.
            LSH lsh; lsh.alloc(2,tables,1,1,1,1,capacity);
            std::vector<int> hashes(tables*4,0);
            lsh.fastfill(0,0,torch::Tensor(hashes.data(),{1,tables,4}));
            std::array<int,tables> query{};
            std::vector<int> output(capacity+2,sentinel);
            int count=lsh.retrieve(0,0,query.data(),output.data()+1);
            std::vector<int> got(output.data()+1,output.data()+1+count);
            require(got==std::vector<int>{0},"fastfill characterization changed: re-review defect");
            require(output.front()==sentinel && output.back()==sentinel,"fastfill guard");
            std::cout << "MAGICPIG-FASTFILL-DEFECT: expected {0,1,2,3}; actual {0}; table content remains zero; unfreed mo_ij allocation per table\n" << std::flush;
            return 0; // With LSan enabled, process exit must report leaked allocations.
        }
        require(argc==1,"unknown fixture argument");
        { LSH never_allocated; }
        size_t checks=0;
        for(int length: {1,2,15,16,17,31,32}) for(int lifetime=0;lifetime<3;++lifetime) {
            LSH lsh;
            lsh.alloc(2,tables,layers,heads,kv_heads,batches,capacity);
            for(int generation=0;generation<2;++generation) {
                std::vector<Codes> all_codes;
                for(int layer=0;layer<layers;++layer) for(int request=0;request<batches;++request) {
                    all_codes.push_back(codes_for(length,layer,request,generation));
                    populate(lsh,layer,request,all_codes.back(),length);
                }
                for(int query_kind=0;query_kind<4;++query_kind) for(int layer=0;layer<layers;++layer) {
                    std::vector<int> queries(batches*heads*tables);
                    for(int head=0;head<batches*heads;++head) for(int t=0;t<tables;++t)
                        queries[head*tables+t] = query_kind==0 ? 0 : query_kind==1 ? 3 :
                            query_kind==2 ? (t%2) : (head%2);
                    std::vector<int> guarded(2+batches*heads*capacity,sentinel);
                    std::vector<int> nnz(batches*heads,-1);
                    lsh.batch_retrieve(layer,torch::Tensor(queries.data(),{batches*heads,tables}),
                        torch::Tensor(guarded.data()+1,{batches*heads,capacity}),
                        torch::Tensor(nnz.data(),{batches*heads}));
                    check_results(guarded,nnz,all_codes,length,queries,layer);
                    checks+=batches*heads;
                    // The getter is an explicitly non-owning view. Check shape
                    // and current values while owner lives, never dereference
                    // after destruction; actual Torch ownership is untested.
                    auto mask=lsh.get_mask();
                    require(mask.size(0)==batches && mask.size(1)==heads && mask.size(2)==capacity,"mask shape");
                    auto* mask_bytes=static_cast<uint8_t*>(mask.data_ptr());
                    for(int head=0;head<batches*heads;++head) for(int s=0;s<capacity;++s) {
                        int hits=0;
                        if(s<length) for(int t=0;t<tables;++t)
                            hits+=all_codes[layer*batches+head/heads][((head%heads)/(heads/kv_heads)*tables+t)*length+s]==queries[head*tables+t];
                        require(mask_bytes[head*capacity+s]==std::min(hits,2),"mask reset/head isolation");
                    }
                    // Direct selected retrieve is also independently checked.
                    std::fill(guarded.begin(),guarded.end(),sentinel);
                    int got=lsh.retrieve(layer,0,queries.data(),guarded.data()+1);
                    auto want=expected(all_codes[layer*batches],length,0,queries.data());
                    std::vector<int> direct(guarded.data()+1,guarded.data()+1+got);
                    std::sort(direct.begin(),direct.end());
                    require(direct==want,"direct retrieve mismatch");
                    require(guarded.front()==sentinel && guarded.back()==sentinel,"direct guard");
                }
                auto retained_mask=lsh.get_mask();
                lsh.clear();
                auto mask=lsh.get_mask();
                require(mask.data_ptr()==retained_mask.data_ptr(),"clear changed live mask storage");
                for(int i=0;i<batches*heads*capacity;++i)
                    require(static_cast<uint8_t*>(retained_mask.data_ptr())[i]==0,"clear retained borrowed mask");
                std::vector<int> queries(batches*heads*tables,0), results(batches*heads*capacity,sentinel), nnz(batches*heads,-1);
                for(int layer=0;layer<layers;++layer) {
                    lsh.batch_retrieve(layer,torch::Tensor(queries.data(),{batches*heads,tables}),
                        torch::Tensor(results.data(),{batches*heads,capacity}),torch::Tensor(nnz.data(),{batches*heads}));
                    require(std::all_of(nnz.begin(),nnz.end(),[](int n){return n==0;}),"clear left old buckets");
                    require(std::all_of(results.begin(),results.end(),[](int n){return n==sentinel;}),"empty retrieval touched output");
                }
            }
        }
        // Degenerate all-colliding table: each item appears once regardless of
        // four hits. Empty buckets, zero-length and full capacity exercised.
        for(int length:{0,capacity}) {
            LSH lsh; lsh.alloc(2,tables,1,1,1,1,capacity);
            std::vector<int16_t> hashes(std::max(1,tables*length),0);
            std::vector<int> indices(std::max(1,tables*length));
            for(int t=0;t<tables;++t) for(int s=0;s<length;++s) indices[t*length+s]=s;
            lsh.fill(0,0,torch::Tensor(hashes.data(),{1,tables,length}),torch::Tensor(indices.data(),{1,tables,length}));
            std::array<int,tables> query{};
            std::vector<int> result(capacity+2,sentinel);
            int count=lsh.retrieve(0,0,query.data(),result.data()+1);
            require(count==length,"all-collision/zero-length count");
            for(int s=0;s<length;++s) require(result[s+1]==s,"all-collision membership/order");
            require(result.front()==sentinel && result.back()==sentinel,"full capacity canary");
        }
        std::cout << "MAGICPIG-LSH-RETRIEVAL-CPU: " << checks << " batched head checks; lifecycle, masks, direct retrieval, tails, empty and full capacity passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
