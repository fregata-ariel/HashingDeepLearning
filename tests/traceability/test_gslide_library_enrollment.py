"""Reject missing unit registration, substituted binaries and partial completion."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('suite_contract',ROOT/'tools/gslide_sycl_suites.py')
assert spec and spec.loader
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Enrollment(unittest.TestCase):
    def test_complete_and_missing_marker(self) -> None:
        rows=module.suites()
        log='\n'.join(row['marker'] for row in rows)
        module.require_completion(log)
        with self.assertRaises(ValueError):module.require_completion(log.replace(rows[-1]['marker'],''))

    def test_required_inventory_cannot_be_dropped(self) -> None:
        rows=module.suites()[:-1]
        with patch.object(Path,'read_text',return_value=json.dumps(rows)):
            with self.assertRaises(ValueError):module.suites()

    def test_missing_or_substituted_ctest_binary(self) -> None:
        build=Path('/tmp/gslide-enrollment-probe')
        tests=[{'name':row['name'],'command':[str(build/row['binary'])]} for row in module.suites()]
        def invoke(rows: list[dict[str,object]]) -> None:
            result=subprocess.CompletedProcess([],0,stdout=json.dumps({'tests':rows}))
            with patch.object(module.subprocess,'run',return_value=result):module.require_enrollment(build)
        invoke(tests)
        with self.assertRaises(ValueError):invoke(tests[:-1])
        tests[0]['command']=['/tmp/unrelated/gslide_host_control']
        with self.assertRaises(ValueError):invoke(tests)


if __name__=='__main__':unittest.main()
