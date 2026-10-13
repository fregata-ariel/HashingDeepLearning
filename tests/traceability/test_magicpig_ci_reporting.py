"""Native execution notices must distinguish prospective repairs and ISA skips."""
from __future__ import annotations

import copy
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("magicpig_ci_reporter", ROOT / "tools/report_magicpig_cpu.py")
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


class ReportingContracts(unittest.TestCase):
    def evidence(self, sanitize: bool = False) -> dict:
        return {"status": "pass", "sanitize": sanitize, "capabilities": {"modes": {
            "avx512": {"eligible": False, "reason": "missing CPU/OS features: avx512f"},
            "bf16": {"eligible": False, "reason": "missing CPU/OS features: avx512_bf16"}}},
            "results": [{"suite": "baseline", "mode": "portable", "status": "pass", "sanitize": sanitize,
                         "leak_check_explicitly_disabled": False}]}

    def native(self, marker: str) -> dict:
        data = self.evidence()
        data["capabilities"]["modes"]["avx512"]["eligible"] = True
        data["results"].append({"suite": "softmax", "mode": "avx512", "status": "pass", "sanitize": False,
                                "leak_check_explicitly_disabled": False, "stdout": marker})
        return data

    def test_hardware_skips_never_become_native_pass(self) -> None:
        for sanitize in (False, True):
            phase, entries = reporter.notices(self.evidence(sanitize), "commit", True)
            self.assertEqual(phase, "sanitized" if sanitize else "plain")
            native = [text for title, text in entries if title.endswith(("avx512", "bf16"))]
            self.assertEqual(len(native), 2)
            self.assertTrue(all("SKIP" in text and "not verified" in text for text in native))
            self.assertFalse(any("ACTUALLY EXECUTED" in text for _, text in entries))

    def test_actual_enrolled_native_exp_reports_exact_marker(self) -> None:
        _, entries = reporter.notices(self.native(reporter.NATIVE_EXP_MARKER), "commit", True)
        self.assertTrue(any("ACTUALLY EXECUTED PASS" in text and reporter.NATIVE_EXP_MARKER in text for _, text in entries))

    def test_prospective_exp_repair_cannot_be_claimed_as_executed(self) -> None:
        data = self.native("historical native characterization")
        _, entries = reporter.notices(data, "commit", False)
        self.assertTrue(any("PENDING #74" in text for _, text in entries))
        self.assertFalse(any(reporter.NATIVE_EXP_MARKER in text for _, text in entries))
        with self.assertRaisesRegex(ValueError, "marker missing"):
            reporter.notices(data, "commit", True)

    def test_host_exp_marker_and_wrong_case_count_do_not_validate_native_repair(self) -> None:
        for marker in ("MAGICPIG-EXP-NATIVE-SKIP portable mode uses host exp", "MAGICPIG-EXP-NATIVE-CPU cases=28", "MAGICPIG-EXP-NATIVE-CPU cases=28210", "prefixed MAGICPIG-EXP-NATIVE-CPU cases=2821"):
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "marker missing"):
                reporter.notices(self.native(marker), "commit", True)

    def test_actual_marker_can_have_trailing_metric_fields(self) -> None:
        _, entries = reporter.notices(self.native(reporter.NATIVE_EXP_MARKER + " max_relative_error=1e-6\n"), "commit", True)
        self.assertTrue(any("ACTUALLY EXECUTED PASS" in text and reporter.NATIVE_EXP_MARKER in text for _, text in entries))

    def test_failed_sanitized_phase_remains_failure_with_readable_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plain, sanitized = root / "plain", root / "sanitized"
            plain.mkdir(); sanitized.mkdir()
            first, second = plain / "magicpig_cpu_evidence.json", sanitized / "magicpig_cpu_evidence.json"
            first.write_text(json.dumps(self.evidence()))
            (sanitized / "run.log").write_text("AddressSanitizer: SEGV\n_mm512_storeu_ps at softmax_kernel\n")
            output = io.StringIO()
            with patch.object(sys, "argv", ["reporter", "--evidence", str(first), "--evidence", str(second)]), patch.object(reporter.subprocess, "check_output", return_value="tested-commit\n"), patch.object(reporter, "native_exp_enrolled", return_value=True), redirect_stdout(output), self.assertRaises(SystemExit) as failure:
                reporter.main()
            self.assertEqual(failure.exception.code, 1)
            text = output.getvalue()
            self.assertIn("::notice title=MagicPIG CPU plain::PASS", text)
            self.assertIn("::error title=MagicPIG CPU sanitized failure::FAIL/INCOMPLETE", text)
            self.assertIn("AddressSanitizer: SEGV%0A_mm512_storeu_ps", text)
            self.assertNotIn("MagicPIG CPU sanitized::PASS", text)

    def test_compilation_without_execution_is_not_a_native_result(self) -> None:
        data = self.evidence()
        data["capabilities"]["modes"]["avx512"]["eligible"] = True
        data["compiler_command"] = ["g++", "-mavx512f"]
        with self.assertRaisesRegex(ValueError, "no execution result"):
            reporter.notices(data, "commit")

    def test_failed_suite_or_disabled_leak_detection_rejects(self) -> None:
        original = self.evidence(True)
        for field, value in (("status", "fail"), ("leak_check_explicitly_disabled", True), ("sanitize", False)):
            data = copy.deepcopy(original)
            data["results"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                reporter.notices(data, "commit")

    def test_native_execution_without_capability_confirmation_rejects(self) -> None:
        data = self.native(reporter.NATIVE_EXP_MARKER)
        data["capabilities"]["modes"]["avx512"]["eligible"] = False
        with self.assertRaisesRegex(ValueError, "confirmed capability"):
            reporter.notices(data, "commit", True)

    def test_native_exp_requirement_tracks_actual_ledger_enrollment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "papers/traceability").mkdir(parents=True)
            (root / "papers/traceability.json").write_text('{"experiments": []}')
            self.assertFalse(reporter.native_exp_enrolled(root))
            (root / "papers/traceability/exp.json").write_text(json.dumps({"experiments": [{"id": "MAGICPIG-EXP-NATIVE-CPU"}]}))
            self.assertTrue(reporter.native_exp_enrolled(root))

    def test_notice_control_characters_are_escaped(self) -> None:
        self.assertEqual(reporter.escape_notice("missing 100%\r\nfeature"), "missing 100%25%0D%0Afeature")


if __name__ == "__main__":
    unittest.main()
