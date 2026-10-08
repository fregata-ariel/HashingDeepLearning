"""Fail-closed selection contracts; these tests never launch SIMD executables.

Copy into tests/traceability for CI. MAGICPIG_CPU_DRIVER can select an isolated
working driver without changing the eventual repository-relative import.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = Path(os.environ.get("MAGICPIG_CPU_DRIVER", str(ROOT / "tools/check_magicpig_cpu.py")))
SPEC = importlib.util.spec_from_file_location("magicpig_capability_driver", DRIVER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load MagicPIG CPU driver: {DRIVER_PATH}")
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)

ALL_FLAGS = {"avx512f", "fma", "avx512bw", "avx512_bf16"}


class CapabilitySelection(unittest.TestCase):
    def assert_modes(self, flags: set[str], machine: str, gcc_major: int | None,
                     requested: bool, expected_avx: bool, expected_bf16: bool) -> None:
        result = driver.decide_modes(flags, machine, gcc_major, requested)
        self.assertEqual(set(result), {"avx512", "bf16"})
        for name, expected in (("avx512", expected_avx), ("bf16", expected_bf16)):
            mode = result[name]
            self.assertIs(type(mode["eligible"]), bool)
            self.assertEqual(mode["eligible"], expected, msg=f"{name}: {mode}")
            self.assertIsInstance(mode["reason"], str)
            self.assertTrue(mode["reason"].strip(), msg=f"{name}: missing selection reason")

    def test_opt_out_overrides_supported_native_hardware(self) -> None:
        self.assert_modes(ALL_FLAGS, "x86_64", 13, False, False, False)

    def test_known_x86_64_platform_spellings(self) -> None:
        for machine in ("x86_64", "AMD64"):
            with self.subTest(machine=machine):
                self.assert_modes(ALL_FLAGS, machine, 11, True, True, True)

    def test_unknown_or_non_x86_platform_cannot_authorize_native(self) -> None:
        for machine in ("", "unknown", "aarch64", "arm64", "riscv64", "ppc64le"):
            with self.subTest(machine=machine):
                self.assert_modes(ALL_FLAGS, machine, 13, True, False, False)

    def test_missing_compiler_cannot_authorize_native(self) -> None:
        self.assert_modes(ALL_FLAGS, "x86_64", None, True, False, False)

    def test_missing_instruction_flags_fail_closed(self) -> None:
        cases = [
            (set(), False, False),
            ({"fma", "avx512bw", "avx512_bf16"}, False, False),
            ({"avx512f"}, False, False),
            ({"avx512f", "fma"}, True, False),
            ({"avx512f", "fma", "avx512bw"}, True, False),
        ]
        for flags, avx, bf16 in cases:
            with self.subTest(flags=sorted(flags)):
                self.assert_modes(flags, "x86_64", 13, True, avx, bf16)

    def test_bf16_actual_body_requires_word_load_capability(self) -> None:
        # qk_kernel_bf16_impl includes _mm512_loadu_epi16, not just dpbf16.
        self.assert_modes({"avx512f", "fma", "avx512_bf16"}, "x86_64", 13, True, True, False)

    def test_bf16_compiler_threshold_does_not_disable_basic_avx(self) -> None:
        for gcc_major, bf16 in ((9, False), (10, False), (11, True), (13, True)):
            with self.subTest(gcc_major=gcc_major):
                self.assert_modes(ALL_FLAGS, "x86_64", gcc_major, True, True, bf16)

    def test_bf16_is_independent_of_fma_only_avx_branch(self) -> None:
        # The BF16 TU must be compiled without -mfma; it has its own gate.
        self.assert_modes({"avx512f", "avx512bw", "avx512_bf16"}, "x86_64", 11, True, False, True)

    def test_unrelated_flags_do_not_change_eligibility(self) -> None:
        self.assert_modes(ALL_FLAGS | {"sse2", "aes", "random_unknown_feature"},
                          "x86_64", 13, True, True, True)

    def test_selection_does_not_mutate_input_flags(self) -> None:
        flags = set(ALL_FLAGS)
        before = set(flags)
        self.assert_modes(flags, "x86_64", 13, True, True, True)
        self.assertEqual(flags, before)


if __name__ == "__main__":
    unittest.main()
