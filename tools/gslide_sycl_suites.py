"""Explicit G-SLIDE unit/integration enrollment shared by CPU and GPU drivers."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/'tests/gslide_sycl/suites.json'
REQUIRED={'gslide.host.control','gslide.host.batch','gslide.host.configuration','gslide.host.views','gslide.sycl.network'} | {
    'gslide.sycl.'+name for name in ('reductions','dense_ids','random','hash','candidates','operators','memory','optimizer','output','selection')}


def suites() -> list[dict[str,str]]:
    rows=json.loads(REGISTRY.read_text())
    names=[row['name'] for row in rows]
    if len(names)!=len(set(names)) or set(names)!=REQUIRED:
        raise ValueError('missing/duplicate/unexpected G-SLIDE suite enrollment')
    for row in rows:
        path=Path(row['binary'])
        if path.is_absolute() or '..' in path.parts or not row['marker']:
            raise ValueError('invalid suite descriptor')
    return rows


def require_enrollment(build: Path) -> None:
    result=subprocess.run(['ctest','--test-dir',str(build),'--show-only=json-v1'],capture_output=True,text=True,check=True,timeout=30)
    actual=json.loads(result.stdout)['tests']
    expected=suites()
    if len(actual)!=len(expected) or {row['name'] for row in actual}!={row['name'] for row in expected}:
        raise ValueError('CTest did not enroll all required G-SLIDE suites')
    by_name={row['name']:row for row in actual}
    for row in expected:
        if Path(by_name[row['name']]['command'][0]).resolve()!=(build/row['binary']).resolve():
            raise ValueError('CTest binary substitution: '+row['name'])


def require_completion(log: str) -> None:
    for row in suites():
        if row['marker'] not in log:
            raise ValueError('required suite did not complete: '+row['name'])


def sources() -> list[Path]:
    paths=[ROOT/'ports/g-slide/include/TrainingControl.h',ROOT/'ports/g-slide/library/CMakeLists.txt',ROOT/'ports/sycl-common/include/hashing/sycl_backend.hpp',REGISTRY,Path(__file__).resolve()]
    for directory in ('ports/g-slide/sycl','tests/gslide_sycl','tests/gslide_host'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.hpp','.cpp','.txt','.json'))
    return sorted(set(paths))
