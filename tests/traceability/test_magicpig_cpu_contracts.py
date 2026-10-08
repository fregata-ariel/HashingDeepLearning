"""Driver contracts for actual-body extraction, enrollment and evidence identity."""
from __future__ import annotations
import importlib.util
import hashlib
import json
import shutil
import sys
import subprocess
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
        # Isolate baseline contracts from independently enrolled task suites.
        for path in ("tests/traceability/magicpig_suites", "tests/traceability/magicpig_native_suites", "tests/traceability/magicpig_python_suites", "papers/traceability"):
            (self.root / path).mkdir(parents=True)
        for path in ("tests/traceability/magicpig_suites/baseline.json", "tests/traceability/magicpig_native_suites/bf16_tile.json", "papers/traceability/magicpig-baseline.json"):
            shutil.copyfile(ROOT / path, self.root / path)
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

    def test_constructor_destructor_tensor_and_static_inline_extraction(self) -> None:
        for text, symbol in (("LSH::LSH(){allocated=false;}", "LSH::LSH"), ("LSH::~LSH(){delete[] mask;}", "LSH::~LSH"), ("torch::Tensor LSH::get_mask(){return mask;}", "LSH::get_mask"), ("static inline void fast_memcpy_avx512(int* d){d[0]=1;}", "fast_memcpy_avx512")):
            self.assertEqual(driver.extract_definition(text, symbol), (text, 1))

    def test_python_descriptor_rejects_unapproved_source_class_blocks_and_path(self) -> None:
        fixture = self.root / "tests/traceability/test_magicpig_simhash_cpu.py"
        fixture.write_text("pass\n")
        valid = {"name": "simhash", "test_file": str(fixture.relative_to(self.root)), "source_file": "third_party/magicpig/models/attnserver.py", "class_name": "LSHSparseAttnServer", "blocks": ["packing", "centering", "fill_hash", "decode_hash", "append_centering"]}
        path = self.root / "tests/traceability/magicpig_python_suites/simhash.json"
        with patch.object(driver, "ROOT", self.root):
            path.write_text(json.dumps(valid))
            self.assertEqual(set(driver.load_python_suites()), {"simhash"})
            for key, value in (("source_file", "outside.py"), ("class_name", "Other"), ("blocks", ["packing"]), ("test_file", "../outside.py")):
                path.write_text(json.dumps({**valid, key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    driver.load_python_suites()

    def test_python_evidence_requires_source_and_actual_statement_hashes(self) -> None:
        source = self.root / "third_party/magicpig/models/attnserver.py"
        source.parent.mkdir(parents=True)
        source.write_text("a = 1\nb = 2\n")
        suite = {"name": "simhash", "source_file": str(source.relative_to(self.root)), "blocks": ["packing"]}
        data = {"source": str(source.relative_to(self.root)), "gpu_execution": False, "torch_runtime": False, "name": "simhash", "status": "passed", "source_sha256": driver.digest(source), "selected_blocks": {"packing": {"start_line": 1, "end_line": 1, "sha256": hashlib.sha256(b"a = 1").hexdigest()}}}
        with patch.object(driver, "ROOT", self.root):
            driver.validate_python_evidence(suite, data)
            data["selected_blocks"]["packing"]["sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                driver.validate_python_evidence(suite, data)
            data["source_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "enrolled source"):
                driver.validate_python_evidence(suite, data)

    def test_expected_leak_requires_exact_result_site_count_and_clean_diagnostic(self) -> None:
        output = "MAGICPIG-FASTFILL-DEFECT: expected {0,1,2,3}; actual {0}"
        diagnostic = "ERROR: LeakSanitizer: detected memory leaks\nLSH::fastfill\nSUMMARY: AddressSanitizer: 64 byte(s) leaked in 4 allocation(s)"
        good = subprocess.CompletedProcess([], 1, output, diagnostic)
        self.assertEqual(driver.validate_fastfill_probe(good, True, False), "expected_leak_reproduced")
        for result in (subprocess.CompletedProcess([], 0, output, diagnostic), subprocess.CompletedProcess([], 1, "wrong result", diagnostic), subprocess.CompletedProcess([], 1, output, diagnostic.replace("LSH::fastfill", "unrelated")), subprocess.CompletedProcess([], 1, output, diagnostic.replace("64 byte", "32 byte")), subprocess.CompletedProcess([], 1, output, diagnostic+"\nruntime error: overflow"), subprocess.CompletedProcess([], 1, output, diagnostic+"\nERROR: AddressSanitizer: heap-buffer-overflow")):
            with self.assertRaises(RuntimeError):
                driver.validate_fastfill_probe(result, True, False)
        self.assertEqual(driver.validate_fastfill_probe(subprocess.CompletedProcess([], 0, output, ""), True, True), "not_checked")

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
