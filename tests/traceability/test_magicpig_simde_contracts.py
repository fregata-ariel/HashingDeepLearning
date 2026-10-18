"""Control/provenance regressions; these mocks are not native/emulated numerics."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import check_magicpig_extension_cpu as extension
import check_magicpig_server_cpu as server
import magicpig_simde as dependency


class SimdeContracts(unittest.TestCase):
    def test_baseline_flags_and_native_gate_conflict(self) -> None:
        for bf16 in (False, True):
            flags = extension.extension_build_flags(bf16, "simde")
            self.assertIn("-DSIMDE_NO_NATIVE", flags)
            self.assertIn("-mno-avx", flags)
            self.assertIn("-mno-fma", flags)
            self.assertFalse(any(flag.startswith("-mavx") or flag == "-mfma" for flag in flags))
            self.assertEqual("-DMAGICPIG_EMULATE_BF16" in flags, bf16)
        for driver in (extension, server):
            with patch.object(sys, "argv", ["driver", "--build-dir", "/unused", "--execution-mode", "simde", "--require-native"]), redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    driver.main()
                self.assertEqual(error.exception.code, 2)

    def test_official_pin_license_and_selective_header_contract(self) -> None:
        pin = dependency.source_pin()
        self.assertEqual(pin["commit"], "2e5224981e9123b8fe89a9833a978556d636a565")
        header = (ROOT / "ports/magicpig/compat/simde_cpu.h").read_text()
        self.assertNotIn("#define SIMDE_ENABLE_NATIVE_ALIASES", header)
        self.assertLess(header.index("#include <torch/extension.h>"), header.index("#include <simde/x86/avx512.h>"))
        self.assertLess(header.index("#include <simde/x86/avx512.h>"), header.index("#define __AVX512BF16__"))
        self.assertIn("#define FloatToBfloat16_avx512 FloatToBfloat16_simd", header)
        self.assertIn("#define Bfloat16ToFloat_avx512 Bfloat16ToFloat_simd", header)
        workflow = (ROOT / ".github/workflows/traceability.yml").read_text()
        self.assertEqual(workflow.count('      - "tools/magicpig_simde.py"'), 2)
        self.assertEqual(workflow.count("--execution-mode simde --require-execution"), 2)

    def test_verified_archive_cache_reuse_and_tamper_rejection(self) -> None:
        revision = "1" * 40
        license_data = b"synthetic unit-test license bytes"
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for relative, contents in [("COPYING", license_data), ("simde/x86/avx512.h", b"synthetic header")]:
                info = tarfile.TarInfo("simde-" + revision + "/" + relative)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
        payload = stream.getvalue()
        pin = {"commit": revision, "archive_url": "https://unused.example/test", "archive_sha256": hashlib.sha256(payload).hexdigest(), "license_sha256": hashlib.sha256(license_data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory, patch.object(dependency, "source_pin", return_value=pin), patch.object(dependency, "urlopen", return_value=io.BytesIO(payload)) as download:
            cache = Path(directory)
            source = dependency.prepare_simde(cache)
            dependency.prepare_simde(cache)
            self.assertEqual(download.call_count, 1)
            (source / "simde/x86/avx512.h").write_text("changed")
            with self.assertRaisesRegex(ValueError, "modified SIMDe cache"):
                dependency.prepare_simde(cache)
            (cache / (revision + ".tar.gz")).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "archive SHA256"):
                dependency.prepare_simde(cache)
        bad = dict(pin, archive_sha256="0" * 64)
        with tempfile.TemporaryDirectory() as directory, patch.object(dependency, "source_pin", return_value=bad), patch.object(dependency, "urlopen", return_value=io.BytesIO(payload)):
            with self.assertRaisesRegex(ValueError, "download SHA256"):
                dependency.prepare_simde(Path(directory))

    def test_binary_audit_rejects_native_vector_mask_and_converter_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "fixture.so"
            binary.write_bytes(b"control fixture, not executable")
            module = ModuleType("control_fixture")
            module.__file__ = str(binary)
            legal_imports = "FloatToBfloat16_simd Bfloat16ToFloat_simd"
            for instruction in [" 10: vaddps xmm0,xmm1,xmm2", " 10: vmovups ymm0,[rax]", " 10: kmovw k1,eax"]:
                with patch.object(dependency.subprocess, "check_output", return_value=instruction):
                    with self.assertRaisesRegex(ValueError, "native AVX"):
                        dependency.verify_emulated_binary(module, False)
            with patch.object(dependency.subprocess, "check_output", side_effect=[" 10: mov eax,1", "FloatToBfloat16_avx512"]):
                with self.assertRaisesRegex(ValueError, "unconditional FBGEMM"):
                    dependency.verify_emulated_binary(module, False)
            with patch.object(dependency.subprocess, "check_output", side_effect=[" 10: mov eax,1", legal_imports]):
                self.assertFalse(dependency.verify_emulated_binary(module, True)["native_avx_instructions"])

    def test_cli_execution_identities_and_persisted_strict_failures(self) -> None:
        fake_extension = ModuleType("test_magicpig_extension_cpu")
        setattr(fake_extension, "exercise", lambda sparse, lsh, bf16: {"status": "pass", "bf16": bf16, "heads_checked": 150, "dispatch_traces": [["attention_wrapper"]], "cxx11_abi": True, "fbgemm_model_bit_mismatches": 0})
        fake_server = ModuleType("test_magicpig_server_cpu")
        setattr(fake_server, "exercise_server", lambda sparse, lsh: {"status": "pass", "head_union_checks": 192, "two_KV_GQA4_union_checks": 64, "overflow_rejections": 4, "branches": ["dense", "sampled", "baseline_full"], "full_orchestrator": True, "gpu_verified": False})
        original_read = Path.read_text
        baseline = "avx512f avx512bw avx512dq avx512vl fma"
        cases = [(extension, "", [], 0, ["skip", "skip"]), (extension, "", ["--require-native"], 1, ["skip", "skip"]), (extension, baseline, ["--require-native"], 1, ["pass", "skip"]), (extension, baseline + " avx512_bf16", ["--require-native"], 0, ["pass", "pass"]), (server, "", [], 0, ["skip"]), (server, "", ["--require-native"], 1, ["skip"]), (server, baseline, ["--require-native"], 0, ["pass"]), (extension, "", ["--execution-mode", "simde", "--require-execution"], 0, ["pass", "pass"]), (server, "", ["--execution-mode", "simde", "--require-execution"], 0, ["pass"])]
        for driver, cpu_flags, arguments, expected_exit, statuses in cases:
            with self.subTest(driver=driver.__name__, arguments=arguments), tempfile.TemporaryDirectory() as directory:
                evidence_path = Path(directory) / ("extension-evidence.json" if driver is extension else "server-evidence.json")
                output = io.StringIO()
                emit = driver.emit_native_notice
                def after_persist(title: str, message: str) -> None:
                    self.assertTrue(evidence_path.is_file())
                    emit(title, message)
                def read(path: Path, *args: object, **kwargs: object) -> str:
                    return cpu_flags if str(path) == "/proc/cpuinfo" else original_read(path)
                actual_exit = 0
                with patch.object(sys, "argv", ["driver", "--build-dir", directory, *arguments]), patch.object(Path, "read_text", read), patch.object(driver, "build_extensions", return_value=(ModuleType("sparse"), ModuleType("lsh"))), patch.object(driver, "emit_native_notice", after_persist), patch.object(driver, "verify_emulated_binary", return_value={"native_avx_instructions": False}), patch.object(driver, "source_record", return_value={"control_only": True}), patch.dict(sys.modules, {"test_magicpig_extension_cpu": fake_extension, "test_magicpig_server_cpu": fake_server}), redirect_stdout(output):
                    try:
                        driver.main()
                    except SystemExit as error:
                        actual_exit = 1 if error.code else 0
                self.assertEqual(actual_exit, expected_exit)
                evidence = json.loads(evidence_path.read_text())
                actual = [row["status"] for row in evidence["results"]] if driver is extension else [evidence["status"]]
                self.assertEqual(actual, statuses)
                notices = [line for line in output.getvalue().splitlines() if line.startswith("::notice")]
                self.assertEqual(len(notices), len(statuses))
                for notice, status in zip(notices, statuses):
                    label = "SKIPPED" if status == "skip" else "EMULATED_PASS" if "simde" in arguments else "EXECUTED_PASS"
                    self.assertIn(label, notice)
                    if "simde" in arguments:
                        self.assertNotIn("EXECUTED_PASS", notice)
        output = io.StringIO()
        with redirect_stdout(output):
            extension.emit_native_notice("control", "%\r\n::error::injected" + "x" * 10000)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertIn("%25%0D%0A", output.getvalue())
        self.assertLess(len(output.getvalue()), 3100)


if __name__ == "__main__":
    unittest.main()
