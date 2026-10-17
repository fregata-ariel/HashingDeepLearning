#!/usr/bin/env python3
"""Verify the maintained typed later CPU component against independent raw algebra.

Actual frozen FP32 reference parity runs in a separate subprocess with a gated
74cc checkout; production imports use the unique maintained namespace. All
mutant copies/downloaded source stay outside uploaded evidence directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.check_scatterbrain_later_cpu import PIN, TREE, fetch, verify_source

FIXTURE=ROOT/'tests/traceability/test_scatterbrain_sbsmyrf_cpu.py'
ORACLE=ROOT/'tests/traceability/test_scatterbrain_later_cpu.py'
PORT=ROOT/'ports/scatterbrain-later'
MUTATIONS=('alpha_dtype','beta_dtype','weight_dtype','count_log_dtype','missing_subtraction',
    'duplicate_correction','normalizer','common_scale','output_nan','output_inf','output_-inf',
    'weights_nan','weights_inf','weights_-inf')


def require(ok: bool, message: str) -> None:
    if not ok:raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkout(path: Path) -> dict[str,str]:
    def git(*args: str) -> str:
        return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
    return {'commit':git('rev-parse','HEAD'),'tree':git('rev-parse','HEAD^{tree}'),
            'status_porcelain':git('status','--porcelain','--untracked-files=all')}


def execute(port: Path, shared: Path, source: Path | None, output: Path, label: str,
            python: Path) -> tuple[int,dict[str,object]]:
    artifact=output/(label+'.json');artifact.unlink(missing_ok=True)
    command=[str(python),'-I','-B',str(FIXTURE),'--port-root',str(port),'--shared-root',str(shared),'--emit',str(artifact)]
    if source is not None:command+=['--source',str(source)]
    process=subprocess.run(command,capture_output=True,text=True,timeout=180,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    (output/(label+'.log')).write_text(process.stdout+process.stderr)
    require(artifact.is_file(),'runtime failed without child evidence: '+label)
    data=json.loads(artifact.read_text(),parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if not isinstance(data,dict):raise ValueError('invalid child evidence')
    return process.returncode,data


def mutation(port: Path, target: Path, name: str) -> dict[str,str]:
    shutil.copytree(port,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    filename='hash_utils.py' if name in ('alpha_dtype','beta_dtype') else 'attention.py'
    path=target/'scatterbrain_later'/filename
    before=path.read_text()
    changes={
        'alpha_dtype':[('torch.normal(0, 1, (dim, n_hashes), device=device, dtype=dtype)',
                        'torch.normal(0, 1, (dim, n_hashes), device=device)')],
        'beta_dtype':[('uniform(0, r, shape=(1, n_hashes), device=device, dtype=dtype)',
                       'uniform(0, r, shape=(1, n_hashes), device=device)')],
        'weight_dtype':[('dots.new_zeros(self.n_hashes, bs, T * S)',
                         'torch.zeros(self.n_hashes, bs, T * S, device=query.device)')],
        'count_log_dtype':[('torch.log(dup_counts.to(dtype=inner.dtype))','torch.log(dup_counts.float())')],
        'missing_subtraction':[('dots = torch.exp(inner - dots_logsumexp) - dots_prime * dots_prime_scale',
                                'dots = torch.exp(inner - dots_logsumexp)')],
        'duplicate_correction':[('inner = inner - torch.log(dup_counts.to(dtype=inner.dtype))','inner = inner'),
                                ('dots_prime = dots_prime / dup_counts','dots_prime = dots_prime')],
        'normalizer':[('out_normalized = out / normalization.clamp_min(1e-6)',
                       'out_normalized = out / (normalization.clamp_min(1e-6) + 0.125)')],
        'common_scale':[('out_normalized = out / normalization.clamp_min(1e-6)',
                         'out = out * 2\n        normalization = normalization * 2\n        out_normalized = out / normalization.clamp_min(1e-6)')],
    }
    if name.startswith('output_'):
        kind=name.removeprefix('output_')
        edits=[('return out_normalized, attn',"out_normalized[-1, -1, -1, -1] = float('"+kind+"')\n        return out_normalized, attn")]
    elif name.startswith('weights_'):
        kind=name.removeprefix('weights_')
        edits=[('return out_normalized, attn',"if isinstance(attn, tuple):\n            attn[0][-1, -1, -1, 1] = float('"+kind+"')\n        return out_normalized, attn")]
    else:edits=changes[name]
    after=before
    for old,new in edits:
        require(after.count(old)==1,'mutation anchor missing/not unique: '+name)
        after=after.replace(old,new)
    require(before!=after,'mutation unchanged')
    path.write_text(after)
    return {'path':str(path.relative_to(target)),'original_sha256':hashlib.sha256(before.encode()).hexdigest(),'mutant_sha256':digest(path)}


def run(source: Path, shared: Path, output: Path, python: Path) -> dict[str,object]:
    identity=verify_source(source)
    tested_checkout,shared_checkout=checkout(ROOT),checkout(shared)
    hashes={str(p.relative_to(ROOT)):digest(p) for p in [FIXTURE,ORACLE,Path(__file__),*sorted(PORT.rglob('*.py'))]}
    shared_hashes={str(p.relative_to(shared)):digest(p) for p in sorted((shared/'src').rglob('*.py'))}
    code,reference=execute(PORT,shared,source,output,'reference',python)
    require(code==0 and reference.get('status')=='frozen_fp32_reference','reference parity subprocess failed')
    code,baseline=execute(PORT,shared,None,output,'baseline',python)
    require(code==0 and baseline.get('status')=='maintained_cpu_pass','maintained baseline failed')
    require(baseline.get('fp32_snapshot')==reference.get('snapshot'),'FP32 samples/transforms/support/projection/output/raw-weights source parity differs')
    rejected=[]
    with tempfile.TemporaryDirectory(prefix='sbsmyrf-maintained-mutants-') as temporary:
        for name in MUTATIONS:
            target=Path(temporary)/name
            change=mutation(PORT,target,name)
            code,evidence=execute(target,shared,None,output,name,python)
            expected=('hash input must match' if name=='alpha_dtype' else
                      'actual FP64 hash allocation' if name=='beta_dtype' else
                      'scatter(): Expected self.dtype' if name=='weight_dtype' else
                      'strict FP64 numerator_max_error' if name=='count_log_dtype' else
                      'output independent oracle' if name=='normalizer' else
                      'nonfinite actual value' if name.startswith(('output_','weights_')) else
                      'numerator independent oracle')
            reason=evidence.get('reason')
            require(code==2 and evidence.get('status')=='rejected' and isinstance(reason,str) and expected in reason,
                    'mutation did not reject for intended reason: '+name)
            rejected.append({'name':name,'reason':reason,**change})
    verify_source(source)
    require(all(digest(ROOT/p)==h for p,h in hashes.items()),'maintained source/fixtures changed during execution')
    require(all(digest(shared/p)==h for p,h in shared_hashes.items()),'shared dependencies changed during execution')
    cases=baseline.get('cases')
    if not isinstance(cases,list):raise ValueError('baseline cases missing')
    return {'status':'maintained_sbsmyrf_cpu_pass','original_source_pin':PIN,'original_source_tree':TREE,
        'checkout':tested_checkout,'shared_checkout':shared_checkout,
        'original_identity':identity,'maintained_sha256':hashes,'shared_runtime_sha256':shared_hashes,
        'fp32_source_parity':'exact samples,transforms,permutations,projection,outputs,weights/raw kernel',
        'mutants':rejected,'cases':len(cases),'global_default_dtype':baseline['global_default_dtype'],
        'executed_test_ids':['SCATTERBRAIN-SBSMYRF-HYBRID-CPU','SCATTERBRAIN-SBSMYRF-DTYPE-CPU'],
        'native_executed':False,'gpu_executed':False,'theorem_validated':False,
        'reference_source_failures_repaired':False,'maintained_dtype_and_precision_repair':True}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-dir',type=Path,required=True)
    parser.add_argument('--reference-source-dir',type=Path)
    parser.add_argument('--shared-root',type=Path,default=ROOT/'ports/scatterbrain')
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    args=parser.parse_args();output=args.emit_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    summary=output/'scatterbrain_sbsmyrf_cpu_evidence.json';summary.unlink(missing_ok=True)
    if args.reference_source_dir:evidence=run(args.reference_source_dir.resolve(),args.shared_root.resolve(),output,args.python)
    else:
        with tempfile.TemporaryDirectory(prefix='sbsmyrf-parity-reference-') as temporary:
            source=Path(temporary)/'upstream';fetch(source)
            evidence=run(source,args.shared_root.resolve(),output,args.python)
    summary.write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print('SBSMYRF_MAINTAINED_CPU_PASS cases='+str(evidence['cases'])+' mutants='+str(len(MUTATIONS)))


if __name__=='__main__':main()
