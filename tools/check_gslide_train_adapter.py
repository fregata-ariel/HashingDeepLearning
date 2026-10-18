#!/usr/bin/env python3
"""Execute exact native host caller body with explicitly mocked runtime/layers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from check_gslide_cpu import ROOT


def run(build):
    build.mkdir(parents=True,exist_ok=True)
    report=build/'native_train_adapter.json';report.unlink(missing_ok=True)
    path=ROOT/'ports/g-slide/src/Network.cu';source=path.read_text()
    start=source.index('void Network::train(');opening=source.index('{',start)
    end=opening+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    body=source[start:end]
    fixture=ROOT/'tests/gslide_sycl/native_train_adapter.cpp'
    unit=build/'native_train.cpp'
    unit.write_text('#include '+json.dumps(str(fixture))+'\n'+body+r'''
#include <iostream>
int main(){Network network;std::vector<int> i{0},max{4,4,4};std::vector<float> f{1};
 network.train(i,f,i,i,i,max,1,.001f,1,64,true,false);
 check(trace=="ILabFRRDefghijklBC");
 check(resets==2);for(auto& layer:network.relu_layers){check(layer->delta[0]==5);for(int i=1;i<5;++i)check(layer->delta[i]==0);std::fill_n(layer->delta,5,71.f);}
 trace.clear();network.train(i,f,i,i,i,max,1,.001f,1,64,false,true);
 check(trace=="ILabFRRDefghijklSC");check(resets==4);
 trace.clear();network.train(i,f,i,i,i,max,1,.001f,1,64,false,false);
 check(trace=="ILabFRRDefghijklC");check(resets==6);
 std::cout<<"GSLIDE_NATIVE_HOST_ADAPTER_PASS exact_train_body=true runtime_and_layers=mocked cuda_device_executed=false\n";
}
''')
    exe=build/'native_train_adapter';command=['g++','-std=c++17','-O0','-g','-I'+str(ROOT/'ports/g-slide/include'),str(unit),'-o',str(exe)]
    subprocess.run(command,check=True);output=subprocess.check_output([str(exe)],text=True);print(output,end='')
    report.write_text(json.dumps(dict(command=command,source=str(path.relative_to(ROOT)),body_sha256=hashlib.sha256(body.encode()).hexdigest(),
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
        original_host_body_executed=True,native_cuda_runtime_executed=False,layer_operations='explicit mocks',output=output),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build-dir',type=Path,required=True);run(p.parse_args().build_dir.resolve())
