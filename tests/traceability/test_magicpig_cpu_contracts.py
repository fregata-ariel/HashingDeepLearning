"""Driver contracts for actual-body extraction, enrollment and evidence identity."""
from __future__ import annotations
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("magicpig_cpu_contract_driver", ROOT / "tools/check_magicpig_cpu.py")
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


class DriverContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for path in ("tests/traceability/magicpig_suites", "tests/traceability/magicpig_native_suites", "papers/traceability"):
            shutil.copytree(ROOT / path, self.root / path)
        for path in ("papers/traceability.json", "tests/traceability/test_magicpig_cpu_baseline.cpp", "tests/traceability/test_magicpig_bf16_cpu.cpp", "tests/traceability/magicpig_fp32_adapter.hpp"):
            destination = self.root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, destination)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_extract_masks_comment_literal_braces_and_rejects_duplicate(self) -> None:
        source = 'void demo(){const char* s="}"; /* { } */ if(true){return;} // }\n}\n'
        body, line = driver.extract_definition(source, "demo")
        self.assertEqual(body, source.rstrip())
        self.assertEqual(line, 1)
        with self.assertRaisesRegex(ValueError, "got 2"):
            driver.extract_definition(source + source, "demo")
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            driver.extract_definition("void demo(){", "demo")

    def test_bad_descriptor_and_path_escape_fail(self) -> None:
        descriptor = self.root / "tests/traceability/magicpig_suites/baseline.json"
        original = json.loads(descriptor.read_text())
        with patch.object(driver, "ROOT", self.root):
            for field, value in (("unexpected", True), ("test_file", "../outside.cpp"), ("name", "wrong"), ("definitions", {"../source.cc": ["demo"]})):
                data = dict(original)
                data[field] = value
                descriptor.write_text(json.dumps(data))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    driver.load_suites()

    def test_orphan_cpu_ledger_record_fails_enrollment(self) -> None:
        path = self.root / "papers/traceability/magicpig-baseline.json"
        data = json.loads(path.read_text())
        data["experiments"][0]["test"]["file"] = "tests/traceability/not-enrolled.cpp"
        path.write_text(json.dumps(data))
        with patch.object(driver, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "not enrolled"):
                driver.check_enrollment(driver.load_suites(), driver.load_bf16_suite())

    def test_suite_without_ledger_record_fails_enrollment(self) -> None:
        (self.root / "papers/traceability/magicpig-baseline.json").unlink()
        with patch.object(driver, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "no CPU ledger"):
                driver.check_enrollment(driver.load_suites(), driver.load_bf16_suite())

    def test_unknown_duplicate_selection_invalidates_old_evidence(self) -> None:
        build = self.root / "build"
        evidence = build / "avx512/baseline/evidence.json"
        summary = build / "magicpig_cpu_evidence.json"
        for selection in (["unknown"], ["baseline", "baseline"]):
            evidence.parent.mkdir(parents=True, exist_ok=True)
            evidence.write_text('{"status":"pass"}')
            summary.write_text('{"status":"pass"}')
            with patch.object(driver, "ROOT", self.root):
                with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
                    driver.run(build, "g++", False, False, selection, False)
            self.assertFalse(summary.exists())
            self.assertFalse(evidence.exists())

    def test_exact_nested_fixture_is_compiled_and_hashed(self) -> None:
        port = self.root / "ports/magicpig"
        shutil.copytree(ROOT / "ports/magicpig", port)
        nested = self.root / "tests/traceability/nested/test_magicpig_cpu_baseline.cpp"
        nested.parent.mkdir()
        nested.write_text('int main(){std::cout<<"EXACT_NESTED_MAGIC_TEST\\n";}\n')
        outer = self.root / "tests/traceability/test_magicpig_cpu_baseline.cpp"
        outer.write_text('int main(){std::cout<<"WRONG_BASENAME_TEST\\n";}\n')
        descriptor = self.root / "tests/traceability/magicpig_suites/baseline.json"
        data = json.loads(descriptor.read_text())
        data["test_file"] = str(nested.relative_to(self.root))
        descriptor.write_text(json.dumps(data))
        with patch.object(driver, "ROOT", self.root), patch.object(driver, "PORT", port):
            suite = driver.load_suites()["baseline"]
            evidence = driver.compile_run(suite, "portable", self.root / "build", "g++", False, False)
        self.assertIn("EXACT_NESTED_MAGIC_TEST", evidence["stdout"])
        self.assertNotIn("WRONG_BASENAME_TEST", evidence["stdout"])
        self.assertEqual(evidence["fixture_sha256"], driver.digest(nested))
        self.assertEqual(evidence["descriptor_sha256"], driver.digest(descriptor))
        self.assertEqual(evidence["mode"], "portable")

    def test_ubsan_error_is_fatal_and_cannot_publish_pass_evidence(self) -> None:
        port = self.root / "ports/magicpig"
        shutil.copytree(ROOT / "ports/magicpig", port)
        fixture = self.root / "tests/traceability/test_magicpig_cpu_baseline.cpp"
        fixture.write_text('#include <climits>\nint main(){volatile int a=INT_MAX;volatile int b=1;volatile int result=a+b;(void)result;return 0;}\n')
        with patch.object(driver, "ROOT", self.root), patch.object(driver, "PORT", port):
            suite = driver.load_suites()["baseline"]
            with self.assertRaisesRegex(RuntimeError, "signed integer overflow"):
                driver.compile_run(suite, "portable", self.root / "build", "g++", True, True)
        self.assertFalse((self.root / "build/portable/baseline/evidence.json").exists())


if __name__ == "__main__":
    unittest.main()
