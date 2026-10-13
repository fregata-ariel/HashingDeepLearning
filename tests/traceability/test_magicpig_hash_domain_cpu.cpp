// TRACE_TEST_ID: MAGICPIG-HASH-DOMAIN-CPU
// Native allocation policy and literal integer-code key/query consistency.
#include <array>
#include <climits>
#include <iostream>
#include <numeric>
#include <string>
namespace {
constexpr int sentinel=-1234567;
void require(bool condition,const std::string& message) {
    if(!condition) throw std::runtime_error(message);
}
int integer_pack(int code,int k) {
    // Independent literal bits and integer powers, never floating point packing.
    int packed=0,power=1;
    for(int bit=0;bit<k;++bit) { if(code & (1<<bit)) packed+=power; power*=2; }
    return packed;
}
size_t accepted_codes() {
    size_t checks=0;
    for(int k=1;k<=11;++k) {
        const int count=1<<k;
        LSH owner;owner.alloc(k,2,1,1,1,1,count);
        std::vector<int> codes(2*count);
        for(int code=0;code<count;++code) {
            codes[code]=integer_pack(code,k);
            codes[count+code]=integer_pack(code,k);
        }
        owner.fastfill(0,0,torch::Tensor(codes.data(),{1,2,count}));
        // Discard source key hashes; the native table must own IDs.
        std::fill(codes.begin(),codes.end(),-1);
        for(int code=0;code<count;++code) {
            std::array<int,2> query{integer_pack(code,k),integer_pack(code,k)};
            std::vector<int> output(count+2,sentinel);
            int selected=owner.retrieve(0,0,query.data(),output.data()+1);
            require(selected==1 && output[1]==code,"accepted key/query integer-code mismatch");
            require(output.front()==sentinel && output.back()==sentinel,"output boundary changed");
            for(int i=2;i<count+1;++i) require(output[i]==sentinel,"retrieval tail changed");
            ++checks;
        }
    }
    return checks;
}
size_t rejected_domains() {
    size_t checks=0;
    const std::array<int,9> invalid{INT_MIN,-1,0,12,14,15,16,31,INT_MAX};
    auto reject=[&](LSH& owner,int k) {
        try { owner.alloc(k,2,1,1,1,1,4); }
        catch(const std::invalid_argument& error) {
            require(std::string(error.what())=="LSH K must be in [1,11] for exact FP16 hash packing",
                    "wrong hash-domain rejection diagnostic");
            ++checks;return;
        }
        throw std::runtime_error("unsupported K accepted");
    };
    // Invalid calls on fresh owners must not initialize state or allocate tables.
    for(int k:invalid) {
        LSH owner;reject(owner,k);
        owner.alloc(2,2,1,1,1,1,4);
        std::vector<int> hashes(8,0);
        owner.fastfill(0,0,torch::Tensor(hashes.data(),{1,2,4}));
        std::array<int,2> query{};std::array<int,6> output{};output.fill(sentinel);
        require(owner.retrieve(0,0,query.data(),output.data()+1)==4,"valid alloc after rejection failed");
    }
    // Invalid K on an existing owner must reject before clobbering dimensions,
    // pointers or populated table state; no valid reallocation is attempted.
    LSH populated;populated.alloc(2,2,1,1,1,1,4);
    std::vector<int> hashes{0,1,2,3,0,1,2,3};
    populated.fastfill(0,0,torch::Tensor(hashes.data(),{1,2,4}));
    for(int k:invalid) {
        reject(populated,k);
        for(int code=0;code<4;++code) {
            std::array<int,2> query{code,code};std::array<int,6> output{};output.fill(sentinel);
            require(populated.retrieve(0,0,query.data(),output.data()+1)==1 && output[1]==code,
                    "rejected K changed existing owner state");
            require(output.front()==sentinel && output.back()==sentinel,"post-rejection output guard");
        }
    }
    return checks;
}
}
int main() {
    try {
        size_t codes=accepted_codes(),rejections=rejected_domains();
        require(codes==4094 && rejections==18,"hash-domain fixture count changed");
        std::cout<<"MAGICPIG-HASH-DOMAIN-CPU: "<<codes<<" accepted integer key/query codes at K1..11; "
                 <<rejections<<" unsupported K rejections before owner mutation; K11/12/14/15 boundaries passed\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
