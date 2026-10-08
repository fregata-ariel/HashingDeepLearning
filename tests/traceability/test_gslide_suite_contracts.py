"""Guard enrollment failures that could otherwise silently drop task coverage."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gslide_driver", ROOT / "tools/check_gslide_cpu.py")
assert spec is not None and spec.loader is not None
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)
ledger_spec = importlib.util.spec_from_file_location("gslide_ledger", ROOT / "tools/validate_traceability.py")
assert ledger_spec is not None and ledger_spec.loader is not None
ledger = importlib.util.module_from_spec(ledger_spec)
ledger_spec.loader.exec_module(ledger)


class SuiteContracts(unittest.TestCase):
    def test_cpu_ledger_cannot_register_an_unexecuted_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "papers").mkdir()
            baseline = root / "tests/traceability/test_gslide_cpu_emulation.cpp"
            data = {"experiments": [{"id": "GSLIDE-ORPHAN-CPU", "paper": "g-slide-2022",
                                     "test": {"file": "tests/traceability/orphan.cpp"}}]}
            (root / "papers/traceability.json").write_text(json.dumps(data))
            suites = {"baseline": driver.Suite("baseline", baseline, driver.SELECTED)}
            with patch.object(driver, "ROOT", root), self.assertRaisesRegex(ValueError, "not an enrolled suite"):
                driver.check_enrollment(suites)

    def test_failed_selection_invalidates_a_previous_success_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            summary = build / "gslide_cpu_summary.json"
            summary.write_text('{"suites":["baseline"]}')
            with patch.object(driver, "check_archive", return_value=27), self.assertRaises(ValueError):
                driver.run(build, False, ["not-enrolled"])
            self.assertFalse(summary.exists())

    def test_nested_same_basename_includes_and_hashes_the_exact_test(self) -> None:
        # The parent basename is a real six-case test. This nested file has a
        # different executable marker; accidentally including the parent fails.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/traceability") as directory:
            nested = Path(directory) / "test_gslide_cpu_emulation.cpp"
            nested.write_text('int main() { std::cout << "EXACT_NESTED_TEST\\n"; }\n')
            build = Path(directory) / "build"
            suite = driver.Suite("nested", nested, driver.SELECTED)
            driver.run_suite(suite, build, False, 27)
            source = (build / "gslide_cpu_generated.cpp").read_text()
            self.assertIn('#include ' + json.dumps(nested.as_posix()), source)
            report = json.loads((build / "gslide_cpu_evidence.json").read_text())
            self.assertIn(str(nested.relative_to(ROOT)), report["inputs_sha256"])

    def test_extract_ignores_braces_in_comments_and_literals(self) -> None:
        source = '__global__ void example() { /* } */ const char* x="{"; if (true) { } }'
        body, line = driver.extract(source, "example")
        self.assertEqual(body, source)
        self.assertEqual(line, 1)
        with self.assertRaises(ValueError):
            driver.extract(source + source, "example")

    def test_invalid_descriptor_cannot_be_silently_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suites = root / "tests/traceability/gslide_suites"
            suites.mkdir(parents=True)
            (suites / "missing.json").write_text(json.dumps({
                "name": "missing", "test_file": "tests/traceability/absent.cpp", "definitions": {}}))
            with patch.object(driver, "ROOT", root), self.assertRaises(ValueError):
                driver.load_suites()

    def test_unknown_and_duplicate_selections_fail_before_compile(self) -> None:
        with patch.object(driver, "check_archive", return_value=27):
            with self.assertRaises(ValueError):
                driver.run(Path("unused"), False, ["not-enrolled"])
            with self.assertRaises(ValueError):
                driver.run(Path("unused"), False, ["baseline", "baseline"])


class LedgerFragments(unittest.TestCase):
    def test_duplicate_and_malformed_fragments_cannot_hide_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            papers = root / "papers"
            fragments = papers / "traceability"
            fragments.mkdir(parents=True)
            (root / "impl.cpp").write_text("void symbol() {}")
            (root / "test.cpp").write_text("// TRACE_TEST_ID: TEST")
            experiment = {"id": "TEST", "paper": "paper", "relation": "support",
                          "implementation": {"file": "impl.cpp", "symbol": "symbol"},
                          "test": {"file": "test.cpp", "id": "TEST"}}
            baseline = papers / "traceability.json"
            baseline.write_text(json.dumps({"papers": {"paper": {
                "title": "Test", "version": "fixture", "source_url": "https://example.invalid", "local_file": None}},
                "experiments": [experiment]}))
            fragment = fragments / "task.json"
            with patch.object(ledger, "ROOT", root), patch.object(ledger, "LEDGER", baseline):
                fragment.write_text(json.dumps({"schema_version": 1, "experiments": [experiment]}))
                self.assertEqual(len(ledger.load()["experiments"]), 2)
                with self.assertRaisesRegex(ValueError, "duplicate experiment id"):
                    ledger.validate(ledger.load())
                fragment.write_text(json.dumps({"schema_version": 2, "experiments": []}))
                with self.assertRaisesRegex(ValueError, "invalid experiment fragment"):
                    ledger.load()


if __name__ == "__main__":
    unittest.main()
