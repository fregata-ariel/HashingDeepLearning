"""Exact maintained-file enrollment and strict typing/CI boundary regressions."""
from __future__ import annotations

import ast
import fnmatch
import importlib.util
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("python_annotation_audit", ROOT / "tools/audit_python_annotations.py")
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class InventoryContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "ports/magicpig").mkdir(parents=True)
        (self.root / "typings").mkdir()
        self.path = self.root / "inventory.json"
        self.data = {"version": 1, "groups": [{"name": "magicpig", "root": "ports/magicpig", "python_files": [], "stub_files": [], "mypy_path": ["ports/magicpig", "typings"]}]}
        self.save()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data))

    def load(self) -> list:
        return audit.load_inventory(self.path, self.root / "ports")

    def test_real_inventory_includes_zero_magicpig_and_all_current_sources(self) -> None:
        groups = audit.load_inventory(ROOT / "tools/maintained_python_inventory.json", ROOT / "ports")
        self.assertEqual({p for g in groups for p in g.files}, {p.resolve() for p in (ROOT / "ports").rglob("*.py")})
        self.assertEqual({p for g in groups for p in g.stubs}, {p.resolve() for p in (ROOT / "ports").rglob("*.pyi")})
        magic = next(g for g in groups if g.name == "magicpig")
        self.assertEqual(set(magic.files), {p.resolve() for p in (ROOT / "ports/magicpig").rglob("*.py")})
        self.assertIn(ROOT / "ports/magicpig", magic.mypy_path)

    def test_future_source_or_stub_requires_explicit_enrollment(self) -> None:
        for name in ("future.py", "native.pyi"):
            path = self.root / "ports/magicpig" / name
            path.write_text("def apply(value: int) -> int: ...\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unenrolled"):
                self.load()
            path.unlink()

    def test_same_file_count_cannot_hide_replacement(self) -> None:
        source = self.root / "ports/magicpig/first.py"
        source.write_text("pass\n")
        self.data["groups"][0]["python_files"] = ["first.py"]
        self.save()
        self.load()
        source.rename(source.with_name("replacement.py"))
        with self.assertRaisesRegex(ValueError, "missing file"):
            self.load()

    def test_extra_maintained_root_cannot_escape_inventory(self) -> None:
        path = self.root / "ports/new_port/source.py"
        path.parent.mkdir()
        path.write_text("pass\n")
        with self.assertRaisesRegex(ValueError, "unenrolled"):
            self.load()

    def test_unsafe_archive_path_duplicate_root_and_missing_search_path_reject(self) -> None:
        original = dict(self.data["groups"][0])
        for field, value in (("root", "../outside"), ("root", "third_party/magicpig"), ("mypy_path", ["typings"]), ("python_files", ["../other.py"])):
            self.data["groups"] = [{**original, field: value}]
            self.save()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.load()
        self.data["groups"] = [original, {**original, "name": "duplicate-root"}]
        self.save()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.load()

    def test_enrolled_magicpig_source_and_stub_share_strict_typecheck(self) -> None:
        for name in ("future.py", "native.pyi"):
            (self.root / "ports/magicpig" / name).write_text("def apply(value: int) -> int: ...\n")
        self.data["groups"][0]["python_files"] = ["future.py"]
        self.data["groups"][0]["stub_files"] = ["native.pyi"]
        self.save()
        with patch.object(audit.subprocess, "run") as run, redirect_stdout(io.StringIO()):
            audit.strict_typecheck(self.load(), self.root)
        self.assertEqual(run.call_count, 1)
        args, kwargs = run.call_args
        command = args[0]
        self.assertIn("--strict", command)
        self.assertIn("--disallow-any-explicit", command)
        self.assertEqual(set(command[-2:]), {str(self.root / "ports/magicpig/future.py"), str(self.root / "ports/magicpig/native.pyi")})
        self.assertTrue(kwargs["check"])
        self.assertEqual(kwargs["env"]["MYPYPATH"].split(audit.os.pathsep), [str(self.root / "ports/magicpig"), str(self.root / "typings")])
        with patch.object(audit.subprocess, "run", side_effect=subprocess.CalledProcessError(1, command)), self.assertRaises(subprocess.CalledProcessError):
            audit.strict_typecheck(self.load(), self.root)

    def test_empty_group_explicitly_reported_without_fake_mypy_pass(self) -> None:
        output = io.StringIO()
        with patch.object(audit.subprocess, "run") as run, redirect_stdout(output):
            audit.strict_typecheck(self.load(), self.root)
        run.assert_not_called()
        self.assertIn("PYTHON_TYPING_EMPTY group=magicpig maintained_sources=0", output.getvalue())
        self.assertNotIn("PYTHON_TYPING_PASS", output.getvalue())

    def test_unannotated_or_explicit_any_function_rejects(self) -> None:
        for source in ("def apply(value): pass", 'def apply(value: "Any") -> "Any": pass', "def apply(value: typing.Any) -> int: pass"):
            node = ast.parse(source).body[0]
            self.assertTrue(audit._audit_function(node, Path("ports/magicpig/future.py")))
        node = ast.parse("def apply(value: int) -> int: return value").body[0]
        self.assertEqual(audit._audit_function(node, Path("ports/magicpig/future.py")), [])


class WorkflowContracts(unittest.TestCase):
    def trigger_paths(self, text: str, event: str) -> list[str]:
        section = text.split(f"  {event}:\n", 1)[1].split("  workflow_dispatch:", 1)[0]
        if event == "push":
            section = section.split("  pull_request:", 1)[0]
        return re.findall(r'^      - "([^"]+)"$', section, re.M)

    def test_typing_filters_enroll_inventory_source_stubs_and_contracts(self) -> None:
        text = (ROOT / ".github/workflows/python-typing.yml").read_text()
        for event in ("push", "pull_request"):
            paths = self.trigger_paths(text, event)
            for changed in ("ports/magicpig/models/future.py", "ports/magicpig/native.pyi", "tools/maintained_python_inventory.json", "tools/audit_python_annotations.py", "tests/traceability/test_python_typing_inventory.py", "docs/MAGICPIG_INTEGRATION.md"):
                with self.subTest(event=event, changed=changed):
                    self.assertTrue(any(fnmatch.fnmatchcase(changed, p) for p in paths))
        self.assertIn("python tests/traceability/test_python_typing_inventory.py", text)
        self.assertIn("python tools/audit_python_annotations.py --root ports --mypy", text)
        config = (ROOT / "mypy.ini").read_text()
        self.assertIn("strict = True", config)
        self.assertNotIn("ignore_missing_imports", config)

    def test_traceability_cpu_lane_and_capability_skip_policy_remain(self) -> None:
        text = (ROOT / ".github/workflows/traceability.yml").read_text()
        for event in ("push", "pull_request"):
            paths = self.trigger_paths(text, event)
            for changed in ("ports/magicpig/library/lsh/lsh.cc", "third_party/magicpig/models/attnserver.py", "tests/traceability/test_magicpig_cache_merge_cpu.py", "docs/MAGICPIG_INTEGRATION.md"):
                self.assertTrue(any(fnmatch.fnmatchcase(changed, p) for p in paths))
        self.assertIn("python3 tools/check_magicpig_cpu.py --native --build-dir", text)
        self.assertIn("python3 tools/check_magicpig_cpu.py --native --sanitize --build-dir", text)
        self.assertNotIn("--disable-leak-check", text)
        self.assertNotIn("schedule:", text)
        self.assertNotIn("self-hosted", text)


if __name__ == "__main__":
    unittest.main()
