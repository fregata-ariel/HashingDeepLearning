#!/usr/bin/env python3
"""Execute two archived CUDA bodies to retain pre-repair negative controls.
No parallel CUDA claim: ordinary CPU threads execute the original hash body
with a 32-item barrier and atomic_ref; the dense index body executes serially.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from check_gslide_cpu import ROOT, check_archive, extract


def run(build):
    build.mkdir(parents=True, exist_ok=True)
    evidence = build / 'source_regressions.json'
    evidence.unlink(missing_ok=True)
    check_archive()
    bodies = []
    identities = []
    for file, symbol in [('kernel.cu', 'get_dense_acts_knl'), ('lshKnl.cu', 'init_hash_knl')]:
        source = (ROOT / 'third_party/g-slide/src' / file).read_text()
        body, line = extract(source, symbol)
        bodies.append(body)
        identities.append(dict(file='third_party/g-slide/src/'+file, symbol=symbol, line=line,
                               body_sha256=hashlib.sha256(body.encode()).hexdigest()))
    unit = build / 'original.cpp'
    unit.write_text(r'''
#include <cassert>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>
#include <barrier>
#include <atomic>
#define __global__
#define __shared__
struct Dim { int x; };
thread_local Dim threadIdx{0};
Dim blockIdx{0},blockDim{1};
struct CscActNodes { int* d_nodes; int* d_offsets; };
int smem[4096];
std::barrier syncpoint(32);
void __syncthreads(){assert(blockDim.x==32);syncpoint.arrive_and_wait();}
int atomicAdd(int* p,int x){return std::atomic_ref<int>(*p).fetch_add(x);}
using std::min;
#define FOR_IDX_ASYNC(i,b,e) for(int i=(b)+threadIdx.x;i<(e);i+=blockDim.x)
'''+'\n'.join(bodies)+r'''
int main(){
 int ids[2]={-1,-1}, offsets[3]={-71,-71,-71};
 for(int i=0;i<2;++i){blockIdx.x=i;get_dense_acts_knl({ids,offsets},1,2);}
 assert(ids[0]==0&&ids[1]==0&&offsets[0]==0&&offsets[1]==1&&offsets[2]==-71);
 int bins[4]={0,1,0,1},buckets[512],sizes[8]={};float weights[128];
 for(int n=0;n<64;++n){weights[2*n]=n%2?1:2;weights[2*n+1]=n%2?2:1;}
 std::fill_n(buckets,512,-1);blockDim.x=32;
 for(int block=0;block<2;++block){blockIdx.x=block;std::vector<std::thread> threads;
  for(int lane=0;lane<32;++lane)threads.emplace_back([&,lane]{threadIdx.x=lane;init_hash_knl(bins,weights,2,64,4,2,1,2,2,1,4,64,buckets,sizes);});
  for(auto& thread:threads)thread.join();
 }
 // Independent expected counts: 64 nodes per table, 32 at each address0/1.
 // Original emits 32 nodes/table, all at address0/table0 or address1/table1;
 // every node32..63 is absent despite a legal original-dispatch block size.
 assert(sizes[0]==32&&sizes[5]==32&&sizes[1]==0&&sizes[4]==0);
 for(int i=0;i<32;++i){assert(buckets[i]>=0&&buckets[i]<32);assert(buckets[5*64+i]>=0&&buckets[5*64+i]<32);}
 for(int i=0;i<512;++i)assert(buckets[i]<32);
 std::cout<<"ORIGINAL_CPU_REGRESSIONS dense_terminal=-71 expected=2 block32_nodes64_lanes2_each_table_count=32 expected=64 nodes32to63_absent=true gpu=false\n";
}
''')
    exe=build/'original'
    command=['g++','-std=c++20','-O0','-g','-pthread','-fno-strict-aliasing',str(unit),'-o',str(exe)]
    subprocess.run(command,check=True)
    output=subprocess.check_output([str(exe)],text=True)
    print(output,end='')
    evidence.write_text(json.dumps(dict(definitions=identities,command=command,
        original_cuda_bodies_cpu_compatible=True,hash_work_items=32,hash_grid_blocks=2,gpu_validated=False,output=output),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True)
    run(p.parse_args().build_dir.resolve())
