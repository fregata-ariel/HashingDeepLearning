"""Mutation checks for immutable source identity and full-tree provenance."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.check_magicpig_sources import validate  # noqa: E402


class SourceContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "papers").mkdir()
        for name in ("magicpig-source-pin.json", "magicpig-upstream-manifest.tsv"):
            shutil.copyfile(ROOT / "papers" / name, self.root / "papers" / name)
        shutil.copytree(ROOT / "third_party/magicpig", self.root / "third_party/magicpig")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_original_subset(self) -> None:
        result = validate(self.root)
        self.assertEqual(result["selected_blobs"], 41)
        self.assertEqual(result["upstream_blobs"], 7227)

    def test_changed_source_rejected(self) -> None:
        path = self.root / "third_party/magicpig/library/lsh/lsh.cc"
        path.write_bytes(path.read_bytes() + b"\n// changed\n")
        with self.assertRaisesRegex(ValueError, "blob identity"):
            validate(self.root)

    def test_missing_license_rejected(self) -> None:
        (self.root / "third_party/magicpig/library/sparse_attention/3rdparty/FBGEMM/third_party/cpuinfo/LICENSE").unlink()
        with self.assertRaisesRegex(ValueError, "file set"):
            validate(self.root)

    def test_unlisted_file_rejected(self) -> None:
        (self.root / "third_party/magicpig/unlisted.cc").write_text("extra")
        with self.assertRaisesRegex(ValueError, "file set"):
            validate(self.root)

    def test_excluded_blob_inventory_tamper_rejected(self) -> None:
        path = self.root / "papers/magicpig-upstream-manifest.tsv"
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            fields = line.split("\t")
            if fields[-1] == "excluded":
                fields[3] = "0" * 40
                lines[i] = "\t".join(fields)
                break
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(ValueError, "tree identity"):
            validate(self.root)

    def test_path_escape_rejected(self) -> None:
        path = self.root / "papers/magicpig-source-pin.json"
        pin = json.loads(path.read_text())
        pin["archive_root"] = "../outside"
        path.write_text(json.dumps(pin))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            validate(self.root)

    def test_failed_run_invalidates_old_success(self) -> None:
        (self.root / "tools").mkdir()
        script = self.root / "tools/check_magicpig_sources.py"
        shutil.copyfile(ROOT / "tools/check_magicpig_sources.py", script)
        output = self.root / "evidence"
        output.mkdir()
        evidence = output / "magicpig_source_evidence.json"
        evidence.write_text('{"status":"source_identity_pass"}')
        (self.root / "third_party/magicpig/LICENSE").unlink()
        run = subprocess.run([sys.executable, str(script), "--emit-dir", str(output)], capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse(evidence.exists())


if __name__ == "__main__":
    unittest.main()
