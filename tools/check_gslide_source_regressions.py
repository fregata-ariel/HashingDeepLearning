#!/usr/bin/env python3
"""Execute two original CUDA bodies serially to characterize translated fixes.
No parallel CUDA claim: one thread/block, barriers and atomics are serial.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from check_gslide_cpu import ROOT, PORT, extract


def run(build):
    build.mkdir(parents=True, exist_ok=True)
    evidence = build / 'source_regressions.json'
    evidence.unlink(missing_ok=True)
    bodies = []
    identities = []
    for file, symbol in [('kernel.cu', 'get_dense_acts_knl'), ('lshKnl.cu', 'init_hash_knl')]:
        source = (PORT / 'src' / file).read_text()
        body, line = extract(source, symbol)
        bodies.append(body)
        identities.append(dict(file='ports/g-slide/src/'+file, symbol=symbol, line=line,
                               body_sha256=hashlib.sha256(body.encode()).hexdigest()))
    unit = build / 'original.cpp'
    unit.write_text(r'''
#include <cassert>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <iostream>
#define __global__
#define __shared__
struct Dim { int x; };
Dim threadIdx{0}, blockIdx{0},blockDim{1};
struct CscActNodes { int* d_nodes; int* d_offsets; };
int smem[4096];
void __syncthreads(){assert(blockDim.x==1);}
int atomicAdd(int* p,int x){int old=*p;*p+=x;return old;}
using std::min;
#define FOR_IDX_ASYNC(i,b,e) for(int i=(b)+threadIdx.x;i<(e);i+=blockDim.x)
'''+'\n'.join(bodies)+r'''
int main(){
 int ids[2]={-1,-1}, offsets[3]={-71,-71,-71};
 for(int i=0;i<2;++i){blockIdx.x=i;get_dense_acts_knl({ids,offsets},1,2);}
 assert(ids[0]==0&&ids[1]==0&&offsets[0]==0&&offsets[1]==1&&offsets[2]==-71);
 int bins[4]={0,1,0,1},buckets[16],sizes[8]={};float weights[4]={2,1,1,2};
 std::fill_n(buckets,16,-1);
 for(int block=0;block<2;++block){blockIdx.x=block;init_hash_knl(bins,weights,2,2,4,2,1,2,2,1,4,2,buckets,sizes);}
 // Expected node 0 in address 0 for each table, node 1 in address 1.
 // Original lane/node staging inserts node 0 twice; node 1 is never inserted.
 assert(sizes[0]==1&&buckets[0]==0&&sizes[5]==1&&buckets[10]==0);
 bool node1=false;for(int i=0;i<16;++i)node1|=buckets[i]==1;assert(!node1);
 std::cout<<"ORIGINAL_SERIAL_REGRESSIONS dense_terminal=-71 expected=2 alternate_node1_absent=true\n";
}
''')
    exe=build/'original'
    command=['g++','-std=c++17','-O0','-g',str(unit),'-o',str(exe)]
    subprocess.run(command,check=True)
    output=subprocess.check_output([str(exe)],text=True)
    print(output,end='')
    evidence.write_text(json.dumps(dict(definitions=identities,command=command,
        original_cuda_bodies_serial=True,gpu_validated=False,output=output),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True)
    run(p.parse_args().build_dir.resolve())
