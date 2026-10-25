#!/usr/bin/env python3
"""Run only the G-SLIDE component library on required CPU and NVIDIA devices."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gslide_sycl_suites import ROOT,require_completion,require_enrollment,sources,suites
from check_gslide_package import run as run_package


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    args=parser.parse_args()
    output=args.build_dir.resolve()
    if output.exists():parser.error('new output directory required')
    output.mkdir(parents=True)
    report: dict[str,object]={'status':'FAIL','checkout':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'profiles':[],
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources()+[Path(__file__).resolve()]}}
    failed=False
    for profile,mask,required in (('cpu','omp','OFF'),('nvidia','cuda','ON')):
        build=output/profile
        build.mkdir()
        env=dict(os.environ,ACPP_TARGETS='generic',ACPP_VISIBILITY_MASK=mask,ACPP_APPDB_DIR=str(build/'appdb'))
        stages=[]
        result={'profile':profile,'status':'FAIL','stages':stages}
        report['profiles'].append(result)
        try:
            commands=[['cmake','-S',str(ROOT/'tests/gslide_sycl'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_CXX_COMPILER=/usr/lib/llvm-18/bin/clang++','-DCMAKE_PREFIX_PATH=/opt/acpp',
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DHASHING_SYCL_REQUIRE_NVIDIA='+required],
                ['cmake','--build',str(build),'--parallel','2'],
                ['ctest','--test-dir',str(build),'-V','--no-tests=error','--timeout','120','--output-junit',str(build/'tests.xml')]]
            for index,command in enumerate(commands):
                started=time.perf_counter()
                completed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=600)
                log=completed.stdout+completed.stderr
                (build/f'{index}.log').write_text(log)
                print(log,end='',flush=True)
                stages.append({'command':command,'exit_code':completed.returncode,'wall_seconds':time.perf_counter()-started})
                completed.check_returncode()
                if index==1:require_enrollment(build)
                if index==2:
                    require_completion(log)
                    if ('is_cpu=1' if profile=='cpu' else 'is_gpu=1') not in log:
                        raise RuntimeError('required actual device missing from runtime evidence')
            result.update(status='PASS',binary_sha256={row['binary']:digest(build/row['binary']) for row in suites()},
                          compiler_commands=json.loads((build/'compile_commands.json').read_text()))
            result['package'] = run_package(build/'installed-consumer', sycl=True,
                require_nvidia=profile=='nvidia', compiler='/usr/lib/llvm-18/bin/clang++', prefix_path='/opt/acpp', env=env)
        except Exception as error:
            failed=True;result['status']='FAIL';result['error']=str(error)
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='FAIL' if failed else 'PASS'
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GSLIDE_LIBRARY_'+str(report['status']),flush=True)
    return int(failed)


if __name__=='__main__':raise SystemExit(main())
