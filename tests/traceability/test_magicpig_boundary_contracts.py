"""Selected-body boundary diagnostics must not accept unrelated failures."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("magicpig_boundary_driver", ROOT / "tools/check_magicpig_cpu.py")
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)


class BoundaryContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.translation = self.root / "selected_bodies.cpp"
        self.executable = self.root / "oracle"
        self.records = [{"symbol": "softmax_kernel_full", "translation_start_line": 20, "translation_end_line": 40}]
        self.contract = {"argument": "--probe-full-tail", "marker": "MAGICPIG-SOFTMAX-PROBE full-tail", "diagnostic": "asan_heap_buffer_overflow", "symbol": "softmax_kernel_full"}
        self.asan = "ERROR: AddressSanitizer: heap-buffer-overflow\n    #0 0x401234 in helper\nSUMMARY: AddressSanitizer: heap-buffer-overflow\n"
        self.resolved = f"softmax_kernel_full(float*, int, float, float*, float*)\n{self.translation}:25\n"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def validate(self, *, code: int = 1, marker: str | None = None, diagnostic: str | None = None, contract: dict[str, str] | None = None, resolved: str | None = None) -> dict[str, object]:
        result = subprocess.CompletedProcess([], code, self.contract["marker"] if marker is None else marker, self.asan if diagnostic is None else diagnostic)
        with patch.object(driver, "checked", return_value=self.resolved if resolved is None else resolved):
            return driver.validate_boundary_probe(result, self.contract if contract is None else contract, self.records, self.translation, self.executable)

    def null_contract(self) -> dict[str, str]:
        return {**self.contract, "diagnostic": "ubsan_null_load"}

    def null_diagnostic(self, line: int = 25) -> str:
        return f"{self.translation}:{line}:12: runtime error: load of null pointer of type 'float'\n"

    def test_precise_asan_and_ubsan_diagnostics_are_accepted(self) -> None:
        self.assertEqual(self.validate()["status"], "known_defect_reproduced")
        result = self.validate(contract=self.null_contract(), diagnostic=self.null_diagnostic())
        self.assertEqual(result["site_resolution"], "diagnostic_source_location")

    def test_wrong_exit_marker_and_generic_crashes_are_rejected(self) -> None:
        for options in ({"code": 0}, {"code": -11}, {"marker": "wrong"}, {"marker": self.contract["marker"] + "\nunrelated output"}, {"diagnostic": "AddressSanitizer:DEADLYSIGNAL"}, {"diagnostic": self.asan.replace("heap-buffer-overflow", "stack-buffer-overflow")}):
            with self.subTest(options=options), self.assertRaises(RuntimeError):
                self.validate(**options)

    def test_unknown_diagnostic_is_rejected_by_validator(self) -> None:
        with self.assertRaises(RuntimeError):
            self.validate(contract={**self.contract, "diagnostic": "unknown"}, diagnostic=self.null_diagnostic())

    def test_additional_sanitizer_diagnostics_are_rejected(self) -> None:
        for extra in ("ERROR: AddressSanitizer: stack-buffer-overflow\n", "ERROR: AddressSanitizer: heap-buffer-overflow\n", "runtime error: signed integer overflow\n", "LeakSanitizer: detected memory leaks\n"):
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                self.validate(diagnostic=self.asan + extra)
        with self.assertRaises(RuntimeError):
            self.validate(contract=self.null_contract(), diagnostic=self.null_diagnostic() + "unrelated.cpp:8:1: runtime error: signed integer overflow\n")

    def test_source_site_requires_paired_selected_symbol_and_location(self) -> None:
        wrong = (f"unrelated(float*)\n{self.translation}:25\n", f"softmax_kernel_full(float*)\n{self.translation}:19\n", f"softmax_kernel_full(float*)\n{self.translation}:41\n", f"softmax_kernel_full(float*)\nunrelated.cpp:25\n", f"softmax_kernel_full_decoy(float*)\n{self.translation}:25\n", f"softmax_kernel_full(float*)\nunrelated.cpp:1\nunrelated(float*)\n{self.translation}:25\n")
        for resolved in wrong:
            with self.subTest(resolved=resolved), self.assertRaises(RuntimeError):
                self.validate(resolved=resolved)

    def test_ubsan_site_must_belong_to_actual_diagnostic_line(self) -> None:
        for diagnostic in (self.null_diagnostic(19), self.null_diagnostic(41), self.null_diagnostic(19) + f"context {self.translation}:25\n"):
            with self.subTest(diagnostic=diagnostic), self.assertRaises(RuntimeError):
                self.validate(contract=self.null_contract(), diagnostic=diagnostic)

    def test_selected_symbol_is_unique(self) -> None:
        for records in ([], self.records * 2):
            with self.subTest(records=records), patch.object(self, "records", records), self.assertRaises(RuntimeError):
                self.validate()

    def test_descriptor_rejects_unknown_duplicate_and_unselected_probes(self) -> None:
        directory = self.root / "tests/traceability/magicpig_suites"
        directory.mkdir(parents=True)
        fixture = directory.parent / "test.cpp"
        fixture.write_text("int main(){return 0;}\n")
        base = {"name": "baseline", "test_file": str(fixture.relative_to(self.root)), "definitions": {driver.SOURCE: ["softmax_kernel_full"]}, "probes": [self.contract]}
        descriptor = directory / "baseline.json"
        descriptor.write_text(json.dumps(base))
        with patch.object(driver, "ROOT", self.root):
            self.assertEqual(len(driver.load_suites()["baseline"].probes), 1)
            for probes in ([{**self.contract, "diagnostic": "unknown"}], [self.contract, self.contract], [{**self.contract, "symbol": "not_selected"}], [{**self.contract, "argument": "--unapproved"}], [{**self.contract, "marker": "multi\nline"}]):
                descriptor.write_text(json.dumps({**base, "probes": probes}))
                with self.subTest(probes=probes), self.assertRaises(ValueError):
                    driver.load_suites()

    def test_failed_probe_removes_previous_success_evidence(self) -> None:
        directory = self.root / "build/portable/boundary"
        directory.mkdir(parents=True)
        evidence = directory / "evidence.json"
        evidence.write_text('{"status":"pass"}')
        descriptor = self.root / "descriptor.json"
        descriptor.write_text("{}")
        suite = driver.Suite("boundary", ROOT / "tests/traceability/test_magicpig_cpu_baseline.cpp", {driver.SOURCE: ["softmax_kernel_full"]}, descriptor, probes=(self.contract,))
        unrelated = subprocess.CompletedProcess([], 1, "wrong marker", "unrelated error")
        with patch.object(driver, "checked", return_value="ordinary fixture passed\n"), patch.object(driver.subprocess, "run", return_value=unrelated):
            with self.assertRaises(RuntimeError):
                driver.compile_run(suite, "portable", self.root / "build", "g++", True, False)
        self.assertFalse(evidence.exists())

    def test_invalid_probe_descriptor_invalidates_previous_run_evidence(self) -> None:
        directory = self.root / "tests/traceability/magicpig_suites"
        directory.mkdir(parents=True)
        fixture = directory.parent / "test.cpp"
        fixture.write_text("int main(){return 0;}\n")
        descriptor = directory / "baseline.json"
        descriptor.write_text(json.dumps({"name": "baseline", "test_file": str(fixture.relative_to(self.root)), "definitions": {driver.SOURCE: ["softmax_kernel_full"]}, "probes": [{**self.contract, "diagnostic": "unknown"}]}))
        build = self.root / "build"
        evidence = build / "portable/baseline/evidence.json"
        evidence.parent.mkdir(parents=True)
        evidence.write_text('{"status":"pass"}')
        summary = build / "magicpig_cpu_evidence.json"
        summary.write_text('{"status":"pass"}')
        with patch.object(driver, "ROOT", self.root):
            with self.assertRaises(ValueError):
                driver.run(build, "g++", True, False, [], False)
        self.assertFalse(evidence.exists())
        self.assertFalse(summary.exists())


if __name__ == "__main__":
    unittest.main()
