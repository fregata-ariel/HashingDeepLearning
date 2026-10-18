#!/usr/bin/env python3
"""Build/run shared kernels on a real SYCL CPU and retain exact evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from check_gslide_cpu import ROOT, check_archive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(build):
    build.mkdir(parents=True, exist_ok=True)
    report=build/'gslide_sycl_evidence.json'
    report.unlink(missing_ok=True)
    contracts=json.loads((ROOT/'ports/g-slide/sycl/source_contracts.json').read_text())
    for definition in contracts['definitions']:
        lines=(ROOT/definition['file']).read_text().splitlines(keepends=True)
        tail=''.join(lines[definition['line']-1:]);opening=tail.index('{');end=opening+1;depth=1
        while depth:
            depth+=(tail[end]=='{')-(tail[end]=='}');end+=1
        if hashlib.sha256(tail[:end].encode()).hexdigest()!=definition['body_sha256']:
            raise ValueError('source contract changed: '+definition['signature'])
    archive=check_archive()
    from check_gslide_train_adapter import run as run_adapter
    run_adapter(build/"native_adapter")
    commands=[['cmake','-S',str(ROOT/'tests/gslide_sycl'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Debug','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
              ['cmake','--build',str(build),'--parallel','2'],
              ['ctest','--test-dir',str(build),'-V','--output-on-failure']]
    log=[]
    for command in commands:
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        print(result.stdout,end='');log.append(result.stdout)
        (build/'run.log').write_text('\n'.join(log))
        result.check_returncode()
    if any(marker not in log[-1] for marker in ('GSLIDE_SYCL_KERNELS checks=', 'GSLIDE_SYCL_PRODUCTION checks=')):
        raise RuntimeError('kernel executable did not report completed checks')
    paths=[ROOT/'ports/g-slide/sycl/kernels.hpp',ROOT/'ports/g-slide/sycl/source_contracts.json',
           ROOT/'tests/gslide_sycl/test_kernels.cpp',ROOT/'tests/gslide_sycl/test_network.cpp',ROOT/'tests/gslide_sycl/CMakeLists.txt',
           ROOT/'ports/g-slide/sycl/network.hpp', ROOT/'ports/g-slide/include/TrainingControl.h',ROOT/'ports/g-slide/src/Network.cu',ROOT/'tests/gslide_sycl/native_train_adapter.cpp',ROOT/'tools/check_gslide_train_adapter.py',
           ROOT/'ports/sycl-common/include/hashing/sycl_backend.hpp',Path(__file__).resolve()]
    report.write_text(json.dumps(dict(checkout=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        working_tree_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
        commands=commands,compiler_version=subprocess.check_output(["acpp","--version"],text=True),
        runtime_device_output=log[-1],archive_blobs_verified=archive,source_contracts=contracts,
        inputs_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        binary_sha256={name:digest(build/name) for name in ('gslide_sycl_test','gslide_sycl_training_test')},
        compiler_commands=json.loads((build/'compile_commands.json').read_text()),cmake_cache_sha256=digest(build/'CMakeCache.txt'),
        log_sha256=digest(build/'run.log'),gpu_validated=False,original_cuda_executed=False,
        arithmetic_tolerance=1e-5,training_tolerance=8e-6,native_adapter_evidence_sha256=digest(build/"native_adapter/native_train_adapter.json"),integer_contract='exact',execution='real SYCL CPU device'),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',required=True,type=Path)
    run(parser.parse_args().build_dir.resolve())
