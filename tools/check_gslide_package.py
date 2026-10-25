#!/usr/bin/env python3
"""Install, relocate and consume G-SLIDE without checkout include paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output: Path, *, sycl: bool = False, require_nvidia: bool = False,
        compiler: str | None = None, prefix_path: str | None = None,
        env: dict[str, str] | None = None) -> dict[str, object]:
    output = output.resolve()
    if output.exists():
        raise ValueError('fresh package evidence directory required')
    if require_nvidia and not sycl:
        raise ValueError('NVIDIA consumer requires SYCL')
    output.mkdir(parents=True)
    report: dict[str, object] = {'status': 'FAIL', 'sycl': sycl, 'require_nvidia': require_nvidia,
        'input_sha256': {}, 'stages': [], 'runtime_environment': {
            key: (os.environ if env is None else env).get(key)
            for key in ('ACPP_TARGETS', 'ACPP_VISIBILITY_MASK', 'ACPP_APPDB_DIR')}}
    inputs = [ROOT / 'ports/g-slide/include/TrainingControl.h', ROOT / 'ports/g-slide/LICENSE',
              ROOT / 'papers/g-slide-source-pin.json', Path(__file__).resolve()]
    for directory in ('ports/g-slide/library', 'ports/g-slide/sycl', 'tests/gslide_consumer'):
        inputs += [p for p in (ROOT / directory).rglob('*') if p.is_file() and p.suffix in ('.hpp', '.cpp', '.txt', '.in')]
    report['input_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(inputs))}
    destination = output / 'result.json'
    # Build outside both checkout and artifact trees in every profile, including
    # hosted jobs whose artifact directory itself lives inside the checkout.
    scratch = tempfile.TemporaryDirectory(prefix='gslide-package-')
    workspace = Path(scratch.name)
    report['build_workspace'] = str(workspace)

    def command(argv: list[str], expected_failure: bool = False) -> str:
        index = len(report['stages'])
        started = time.perf_counter()
        completed = subprocess.run(argv, cwd=output, env=env, capture_output=True, text=True, timeout=180)
        log = completed.stdout + completed.stderr
        (output / f'{index}.log').write_text(log)
        report['stages'].append({'command': argv, 'exit_code': completed.returncode,
            'expected_failure': expected_failure, 'wall_seconds': time.perf_counter() - started,
            'log_sha256': digest(output / f'{index}.log')})
        destination.write_text(json.dumps(report, indent=2) + '\n')
        print(log, end='', flush=True)
        if expected_failure:
            if not completed.returncode:
                raise RuntimeError('unsupported package component was silently accepted')
        else:
            completed.check_returncode()
        return log

    try:
        producer = workspace / 'producer'
        original = workspace / 'install-before-relocation'
        installed = workspace / 'relocated-prefix'
        consumer_source = workspace / 'consumer-source'
        consumer = workspace / 'consumer-build'
        options = ['-DCMAKE_CXX_COMPILER=' + compiler] if compiler else []
        command(['cmake', '-S', str(ROOT / 'ports/g-slide/library'), '-B', str(producer),
                 '-DCMAKE_INSTALL_PREFIX=' + str(original), '-DCMAKE_INSTALL_LIBDIR=lib'] + options)
        command(['cmake', '--build', str(producer), '--parallel', '2'])
        command(['cmake', '--install', str(producer)])
        original.rename(installed)
        assert not original.exists()
        for path in installed.rglob('*.cmake'):
            text = path.read_text()
            if str(ROOT) in text or str(original) in text or str(producer) in text:
                raise RuntimeError('installed package contains private build/source paths')
        report['installed_sha256'] = {str(p.relative_to(installed)): digest(p)
                                     for p in sorted(installed.rglob('*')) if p.is_file()}
        report['relocated_prefix'] = str(installed)
        shutil.copytree(ROOT / 'tests/gslide_consumer', consumer_source)
        config_dir = installed / 'lib/cmake/GSlide'
        extra = ['-DCMAKE_PREFIX_PATH=' + prefix_path] if prefix_path else []
        command(['cmake', '-S', str(consumer_source), '-B', str(consumer),
                 '-DGSlide_DIR=' + str(config_dir), '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                 '-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF',
                 '-DGSLIDE_CONSUMER_SYCL=' + ('ON' if sycl else 'OFF'),
                 '-DGSLIDE_CONSUMER_REQUIRE_NVIDIA=' + ('ON' if require_nvidia else 'OFF')] + options + extra)
        command(['cmake', '--build', str(consumer), '--parallel', '2'])
        commands = json.loads((consumer / 'compile_commands.json').read_text())
        if str(ROOT) in json.dumps(commands):
            raise RuntimeError('external consumer compiled with checkout paths')
        report['compiler_commands'] = commands
        enrolled = json.loads(command(['ctest', '--test-dir', str(consumer), '--show-only=json-v1']))['tests']
        expected = {'consumer.host': 'consumer_host'}
        if sycl:
            expected['consumer.sycl'] = 'consumer_sycl'
        if len(enrolled) != len(expected) or {t['name'] for t in enrolled} != set(expected):
            raise RuntimeError('external consumer tests missing or substituted')
        for test in enrolled:
            if Path(test['command'][0]).resolve() != consumer / expected[test['name']]:
                raise RuntimeError('external consumer executable substituted')
        log = command(['ctest', '--test-dir', str(consumer), '-V', '--no-tests=error',
                       '--timeout', '120', '--output-junit', str(output / 'consumer-tests.xml')])
        if 'GSLIDE_CONSUMER_HOST_PASS' not in log:
            raise RuntimeError('SDK-free installed consumer did not complete')
        if sycl:
            if 'GSLIDE_CONSUMER_SYCL_PASS' not in log or ('is_gpu=1' if require_nvidia else 'is_cpu=1') not in log:
                raise RuntimeError('required real SYCL consumer did not complete')
        report['consumer_output'] = log
        report['junit'] = (output / 'consumer-tests.xml').read_text()
        report['binary_sha256'] = {name: digest(consumer / name) for name in expected.values()}
        # A misspelled required component must fail at configure time.
        negative = workspace / 'unsupported-component'
        negative.mkdir()
        (negative / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\n'
            'project(unsupported_component LANGUAGES NONE)\n'
            'find_package(GSlide CONFIG REQUIRED COMPONENTS nonexistent)\n')
        command(['cmake', '-S', str(negative), '-B', str(negative / 'build'),
                 '-DGSlide_DIR=' + str(config_dir)], expected_failure=True)
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        try:
            snapshot = output / 'external-workspace-evidence'
            shutil.copytree(workspace, snapshot)
            report['workspace_snapshot'] = str(snapshot)
        finally:
            scratch.cleanup()
            report['scratch_workspace_cleaned'] = True
        destination.write_text(json.dumps(report, indent=2) + '\n')
    print('GSLIDE_PACKAGE_PASS', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sycl', action='store_true')
    parser.add_argument('--require-nvidia', action='store_true')
    parser.add_argument('--compiler')
    parser.add_argument('--prefix-path')
    args = parser.parse_args()
    run(args.output, sycl=args.sycl, require_nvidia=args.require_nvidia,
        compiler=args.compiler, prefix_path=args.prefix_path)
