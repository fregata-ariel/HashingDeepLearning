"""GPU absence, missing execution and failed diagnostics must never report pass."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("gpu_driver", ROOT / "tools/check_nvidia_gpu.py")
assert spec and spec.loader
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
MARKERS = "ORIGINAL_DENSE ORIGINAL_HASH DENSE_PASS HASH_PASS STALE_PASS TRAINING_PASS GSLIDE_CUDA_PASS"


class Reporting(unittest.TestCase):
    def test_missing_tool_is_a_failed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver.shutil, "which", return_value=None):
            build=Path(tmp)/"evidence"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(driver.run(build,"75","g++-12",2),1)
            data=json.loads((build/"result.json").read_text())
            self.assertEqual(data["status"],"FAIL")
            self.assertFalse(data["gpu_validated"])

    def test_old_success_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            build=Path(tmp)
            (build/"result.json").write_text('{"status":"PASS"}')
            with self.assertRaises(FileExistsError):
                driver.run(build,"75","g++-12",2)

    def exercise(self, diagnostic_exit=0, diagnostic_summary=True, fixture=True):
        with tempfile.TemporaryDirectory() as tmp:
            build=Path(tmp)/"evidence"
            def fake_run(args, **kwargs):
                stream=kwargs["stdout"]
                stage=Path(stream.name).stem
                output="metadata\n"
                code=0
                if stage=="build":
                    native=build/"native";native.mkdir()
                    (native/"gslide_cuda_test").write_bytes(b"fixture")
                    (native/"original.cu").write_bytes(b"original")
                if stage=="native":
                    output=MARKERS if fixture else "0 tests found; exit success"
                if stage in ("memcheck","racecheck","initcheck","synccheck"):
                    output="GSLIDE_CUDA_PASS\n"
                    if diagnostic_summary:
                        output += "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" if stage=="racecheck" else "ERROR SUMMARY: 0 errors\nLEAK SUMMARY: 0 bytes leaked in 0 allocations"
                    if stage=="memcheck": code=diagnostic_exit
                stream.write(output)
                return subprocess.CompletedProcess(args,code)
            with patch.object(driver.shutil,"which",return_value="/usr/bin/true"), \
                 patch.object(driver.subprocess,"run",side_effect=fake_run), \
                 contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                code=driver.run(build,"75","g++-12",2)
            return code,json.loads((build/"result.json").read_text())

    def test_numerical_pass_cannot_hide_a_failed_diagnostic(self):
        code,data=self.exercise(diagnostic_exit=86)
        self.assertEqual(code,1)
        self.assertTrue(data["native_numerical_pass"])
        self.assertFalse(data["gpu_validated"])
        self.assertEqual(data["failed_stage"],"memcheck")
        self.assertEqual(data["stages"][-1]["exit_code"],86)

    def test_zero_exit_without_diagnostic_summary_fails(self):
        code,data=self.exercise(diagnostic_summary=False)
        self.assertEqual(code,1)
        self.assertFalse(data["gpu_validated"])
        self.assertEqual(data["stages"][-1]["status"],"FAIL")

    def test_zero_tests_is_not_gpu_execution(self):
        code,data=self.exercise(fixture=False)
        self.assertEqual(code,1)
        self.assertFalse(data["gpu_validated"])
        self.assertEqual(data["failed_stage"],"native")

    def test_all_four_diagnostics_are_required(self):
        code,data=self.exercise()
        self.assertEqual(code,0)
        self.assertTrue(data["gpu_validated"])
        self.assertEqual([s["name"] for s in data["stages"]][-5:],
                         ["memcheck","racecheck","initcheck","synccheck","gpu-after"])


if __name__=="__main__":
    unittest.main()
