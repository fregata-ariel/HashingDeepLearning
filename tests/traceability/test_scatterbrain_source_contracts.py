"""Mutation contracts for source identity; no Scatterbrain numerical execution."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.check_scatterbrain_sources import validate  # noqa: E402


class SourceContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "papers").mkdir()
        self.pin_path = self.root / "papers/scatterbrain-source-pin.json"
        self.pin = json.loads((ROOT / "papers/scatterbrain-source-pin.json").read_text())
        for name in ["papers/scatterbrain-source-pin.json", self.pin["upstream_inventory"], self.pin["commit_object"],
                     *[r[key] for r in self.pin["references"] for key in ("upstream_inventory", "commit_object")],
                     *self.pin["preserved_notices"]]:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copytree(ROOT / self.pin["archive_root"], self.root / self.pin["archive_root"])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def save(self) -> None:
        self.pin_path.write_text(json.dumps(self.pin))

    def change_manifest(self, meta: dict, transform) -> None:
        path = self.root / meta["upstream_inventory"]
        path.write_text(transform(path.read_text()))
        meta["inventory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save()

    def test_original_archive_and_distinct_metadata_only_lineages(self) -> None:
        self.assertEqual(validate(self.root), {"selected_blobs": 15, "upstream_blobs": 170,
                                             "upstream_tree_objects": 38, "reference_lineages": 2,
                                             "preserved_notices": 2})
        self.assertEqual([r["upstream_blobs"] for r in self.pin["references"]], [517, 526])
        fingerprints = [r["reference_files"]["src/models/attention/sbsmyrf_attention.py"]["sha"] for r in self.pin["references"]]
        self.assertEqual(fingerprints[0], fingerprints[1])
        self.assertFalse((self.root / "third_party/scatterbrain/src/models/attention").exists())

    def test_adapted_causal_helper_preserves_performer_copyright_notice(self) -> None:
        name = "papers/notices/scatterbrain/lucidrains-performer-LICENSE"
        notice = self.pin["preserved_notices"][name]
        self.assertEqual(notice["applies_to"], ["src/models/modules/attention/scatterbrain_utils.py"])
        self.assertIn(b"Copyright (c) 2020 Phil Wang", (self.root / name).read_bytes())
        helper = self.root / self.pin["archive_root"] / notice["applies_to"][0]
        self.assertIn(b"# efficient causal linear attention, created by EPFL", helper.read_bytes())
        for mutation in ("missing", "changed"):
            original = (self.root / name).read_bytes()
            if mutation == "missing":
                (self.root / name).unlink()
            else:
                (self.root / name).write_bytes(original.replace(b"Phil Wang", b"Anonymous"))
            with self.subTest(mutation=mutation), self.assertRaises(OSError if mutation == "missing" else ValueError):
                validate(self.root)
            (self.root / name).write_bytes(original)

    def test_changed_source_rejects(self) -> None:
        path = self.root / self.pin["archive_root"] / "src/models/modules/attention/sblocal_attention.py"
        path.write_bytes(path.read_bytes() + b"\n# altered\n")
        with self.assertRaisesRegex(ValueError, "blob identity"):
            validate(self.root)

    def test_missing_archive_license_rejects(self) -> None:
        (self.root / self.pin["archive_root"] / "LICENSE").unlink()
        with self.assertRaisesRegex(ValueError, "file/directory set"):
            validate(self.root)

    def test_unlisted_file_or_empty_directory_rejects(self) -> None:
        for directory in (False, True):
            path = self.root / self.pin["archive_root"] / "unexpected"
            path.mkdir() if directory else path.write_text("extra")
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "file/directory set"):
                validate(self.root)
            path.rmdir() if directory else path.unlink()

    def test_unknown_keys_reject_at_every_metadata_boundary(self) -> None:
        original = json.dumps(self.pin)
        for where in ("pin", "selected", "reference", "reference_blob", "notice"):
            self.pin = json.loads(original)
            if where == "pin":
                target = self.pin
            elif where == "selected":
                target = self.pin["selected_blobs"]["LICENSE"]
            elif where == "reference":
                target = self.pin["references"][0]
            elif where == "reference_blob":
                target = next(iter(self.pin["references"][0]["reference_files"].values()))
            else:
                target = next(iter(self.pin["preserved_notices"].values()))
            target["arbitrary_ignored_override"] = True
            self.save()
            with self.subTest(where=where), self.assertRaisesRegex(ValueError, "unknown or missing"):
                validate(self.root)

    def test_malformed_typed_metadata_rejects(self) -> None:
        original = json.dumps(self.pin)
        for field in ("schema_version", "blob_mode", "reference_commit"):
            self.pin = json.loads(original)
            if field == "schema_version":
                self.pin[field] = True
            elif field == "blob_mode":
                self.pin["selected_blobs"]["LICENSE"]["mode"] = []
            else:
                self.pin["references"][0]["commit"] = []
            self.save()
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(self.root)

    def test_duplicate_json_keys_reject(self) -> None:
        text = self.pin_path.read_text()
        self.pin_path.write_text(text.replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,'))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            validate(self.root)

    def test_unsafe_paths_reject(self) -> None:
        original = self.pin["archive_root"]
        for name in ("../outside", "/tmp/outside", "third_party//scatterbrain", "third_party/./scatterbrain", "third_party\\scatterbrain", "third_party/scatterbrain\n"):
            self.pin["archive_root"] = name
            self.save()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe/noncanonical"):
                validate(self.root)
        self.pin["archive_root"] = original

    def test_archive_symlink_and_symlink_parent_reject(self) -> None:
        path = self.root / self.pin["archive_root"] / "README.md"
        payload = path.read_bytes()
        path.unlink(); path.symlink_to("LICENSE")
        with self.assertRaisesRegex(ValueError, "symlink"):
            validate(self.root)
        path.unlink(); path.write_bytes(payload)
        directory = self.root / "papers/source-objects/scatterbrain"
        directory.rename(directory.with_name("saved"))
        directory.symlink_to("saved", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink path traversal"):
            validate(self.root)

    def test_archive_root_symlink_rejects_even_inside_checkout(self) -> None:
        path = self.root / self.pin["archive_root"]
        path.rename(path.with_name("saved")); path.symlink_to("saved", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink path traversal"):
            validate(self.root)

    def test_commit_object_binds_advertised_commit_and_tree(self) -> None:
        self.pin["commit"] = "0" * 40
        self.save()
        with self.assertRaisesRegex(ValueError, "commit object identity"):
            validate(self.root)
        self.pin["commit"] = json.loads((ROOT / "papers/scatterbrain-source-pin.json").read_text())["commit"]
        self.pin["tree"] = "0" * 40
        self.save()
        with self.assertRaisesRegex(ValueError, "commit root tree"):
            validate(self.root)

    def test_excluded_blob_tamper_fails_full_tree_after_manifest_digest_updated(self) -> None:
        def alter(text: str) -> str:
            rows = text.splitlines()
            for i, row in enumerate(rows[1:], 1):
                fields = row.split("\t")
                if fields[-1] == "excluded" and fields[2] == "blob":
                    fields[3] = "0" * 40
                    rows[i] = "\t".join(fields)
                    break
            return "\n".join(rows) + "\n"
        self.change_manifest(self.pin, alter)
        with self.assertRaisesRegex(ValueError, "Git tree identity"):
            validate(self.root)

    def test_reference_inventory_and_fingerprint_are_checked_without_local_source(self) -> None:
        reference = self.pin["references"][0]
        next(iter(reference["reference_files"].values()))["sha"] = "0" * 40
        self.save()
        with self.assertRaisesRegex(ValueError, "reference blob fingerprint"):
            validate(self.root)
        reference["reference_files"] = {}
        self.change_manifest(reference, lambda text: text.replace("\texcluded\n", "\tselected\n", 1))
        with self.assertRaisesRegex(ValueError, "selected subset"):
            validate(self.root)

    def test_invalid_reference_role_and_duplicate_lineage_reject(self) -> None:
        self.pin["references"][0]["role"] = "numerically-validated"
        self.save()
        with self.assertRaisesRegex(ValueError, "reference role"):
            validate(self.root)
        self.pin["references"][0]["role"] = "reference-metadata-only"
        self.pin["references"][1]["commit"] = self.pin["references"][0]["commit"]
        self.save()
        with self.assertRaisesRegex(ValueError, "duplicate lineage"):
            validate(self.root)

    def test_missing_or_changed_historical_notice_rejects(self) -> None:
        path = self.root / next(iter(self.pin["preserved_notices"]))
        path.write_bytes(path.read_bytes() + b"\nchanged")
        with self.assertRaisesRegex(ValueError, "notice identity"):
            validate(self.root)
        path.unlink()
        with self.assertRaises(OSError):
            validate(self.root)

    def test_notice_and_license_declarations_cannot_silently_disappear(self) -> None:
        original = self.pin["preserved_notices"]
        self.pin["preserved_notices"] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, "missing preserved"):
            validate(self.root)
        self.pin["preserved_notices"] = original
        self.pin["licenses"] = []
        self.save()
        with self.assertRaisesRegex(ValueError, "missing archive license"):
            validate(self.root)

    def test_notice_symlink_and_extra_notice_file_reject(self) -> None:
        path = self.root / next(iter(self.pin["preserved_notices"]))
        payload = path.read_bytes(); path.unlink()
        path.symlink_to(self.root / self.pin["archive_root"] / "LICENSE")
        with self.assertRaisesRegex(ValueError, "symlink"):
            validate(self.root)
        path.unlink(); path.write_bytes(payload)
        (path.parent / "extra").write_text("unlisted")
        with self.assertRaisesRegex(ValueError, "notice file set"):
            validate(self.root)

    def test_executable_bit_and_dependency_declaration_reject(self) -> None:
        path = self.root / self.pin["archive_root"] / "README.md"
        path.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "executable mode"):
            validate(self.root)
        path.chmod(0o644)
        self.pin["dependency_source_files"].append("src/missing.py")
        self.save()
        with self.assertRaisesRegex(ValueError, "missing selected license/dependency"):
            validate(self.root)

    def test_failed_cli_invalidates_old_success_and_pass_is_source_only(self) -> None:
        (self.root / "tools").mkdir()
        script = self.root / "tools/check_scatterbrain_sources.py"
        shutil.copyfile(ROOT / "tools/check_scatterbrain_sources.py", script)
        output = self.root / "evidence"
        good = subprocess.run([sys.executable, "-B", str(script), "--emit-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(good.returncode, 0, good.stderr)
        evidence = output / "scatterbrain_source_evidence.json"
        data = json.loads(evidence.read_text())
        self.assertEqual(data["status"], "source_identity_pass")
        for field in ("build_validated", "dependencies_installed", "numerical_runtime_validated", "paper_claims_validated", "gpu_validated"):
            self.assertIs(data[field], False)
        (self.root / self.pin["archive_root"] / "LICENSE").unlink()
        bad = subprocess.run([sys.executable, "-B", str(script), "--emit-dir", str(output)], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertFalse(evidence.exists())


if __name__ == "__main__":
    unittest.main()
