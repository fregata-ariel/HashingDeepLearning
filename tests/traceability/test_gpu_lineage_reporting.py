"""CPU-only regressions: failures cannot hide later GPU profile attempts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("all_gpu_driver", ROOT / "tools/check_all_gpu.py")
assert spec and spec.loader
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Reporting(unittest.TestCase):
    def exercise(self, failed: bool, clean_diagnostics: bool = True) -> tuple[int, dict[str, object], list[list[str]]]:
        commands: list[list[str]] = []
        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            if command[0].endswith("compute-sanitizer") and clean_diagnostics:
                getattr(kwargs["stdout"], "write")("ERROR SUMMARY: 0 errors\nRACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n")
            bad = failed and "mongoose-wrapper" in command
            return subprocess.CompletedProcess(command, 1 if bad else 0)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(sys, "argv", ["check_all_gpu", "--output", str(output)]), patch.object(driver.subprocess, "run", side_effect=fake_run):
                code = driver.main()
            result = json.loads((output / "result.json").read_text())
            return code, result, commands

    def test_failure_keeps_other_lineage_attempts(self) -> None:
        code, result, commands = self.exercise(True)
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("scatterbrain" in command for command in commands))
        self.assertTrue(any("magicpig" in command for command in commands))

    def test_component_success_is_partial_research_acceptance(self) -> None:
        code, result, _ = self.exercise(False)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(result["profiles"][-1]["status"], "NOT_RUN")

    def test_stale_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "result.json").write_text('{"status":"PASS"}')
            with patch.object(sys, "argv", ["check_all_gpu", "--output", str(output)]):
                with self.assertRaises(SystemExit) as raised:
                    driver.main()
            self.assertEqual(raised.exception.code, 2)

    def test_missing_clean_summary_is_not_diagnostic_success(self) -> None:
        code, result, _ = self.exercise(False, clean_diagnostics=False)
        self.assertEqual(code, 1)
        diagnostics = [record for record in result["profiles"] if str(record["profile"]).startswith("scatterbrain-")]
        self.assertEqual(len(diagnostics), 4)
        self.assertTrue(all(record["status"] == "FAIL" for record in diagnostics))


if __name__ == "__main__":
    unittest.main()
