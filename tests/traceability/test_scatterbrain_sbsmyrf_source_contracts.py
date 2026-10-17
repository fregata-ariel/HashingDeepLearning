"""Mutation contracts for the later archive, complete Git tree and license chain."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.check_scatterbrain_sbsmyrf_sources import (  # noqa: E402
    ARCHIVE, COMMIT, NOTICE_ROOT, PIN_PATH, validate,
)


class SBSmyrfSourceContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pin = json.loads((ROOT / PIN_PATH).read_text())
        names = [PIN_PATH, self.pin["upstream_inventory"], self.pin["commit_object"], *self.pin["preserved_notices"]]
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        shutil.copytree(ROOT / ARCHIVE, self.root / ARCHIVE)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def save(self) -> None:
        (self.root / PIN_PATH).write_text(json.dumps(self.pin))

    def manifest(self, transform: Callable[[str], str]) -> None:
        path = self.root / self.pin["upstream_inventory"]
        path.write_text(transform(path.read_text()))
        self.pin["inventory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save()

    def test_original_coherent_17_file_subset_with_all_five_notices(self) -> None:
        self.assertEqual(validate(self.root), {"selected_blobs": 17, "project_source_files": 11,
            "empty_package_initializers": 4, "upstream_blobs": 517, "upstream_tree_objects": 73,
            "preserved_notices": 5, "derivation_records": 1})
        self.assertEqual(self.pin["commit"], COMMIT)
        self.assertEqual(self.pin["component_license"], "GPL-3.0")
        self.assertEqual((self.root / ARCHIVE / "LICENSE").read_bytes(),
                         (self.root / NOTICE_ROOT / "Fly-Apache-LICENSE").read_bytes())
        self.assertIn(b"GNU GENERAL PUBLIC LICENSE", (self.root / NOTICE_ROOT / "SMYRF-GPL-3.0-LICENSE").read_bytes())

    def test_unknown_keys_reject_at_all_boundaries(self) -> None:
        original = copy.deepcopy(self.pin)
        for where in ("pin", "selected", "notice", "derivation"):
            self.pin = copy.deepcopy(original)
            target = self.pin if where == "pin" else (
                self.pin["selected_blobs"]["LICENSE"] if where == "selected" else
                next(iter(self.pin["preserved_notices"].values())) if where == "notice" else self.pin["derivation_evidence"][0])
            target["silently_ignored_override"] = True
            self.save()
            with self.subTest(where=where), self.assertRaisesRegex(ValueError, "unknown or missing"):
                validate(self.root)

    def test_duplicate_json_keys_reject(self) -> None:
        path = self.root / PIN_PATH
        path.write_text(path.read_text().replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,'))
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            validate(self.root)

    def test_bool_float_and_missing_schema_reject(self) -> None:
        for value in (True, 1.0, None):
            self.pin["schema_version"] = value
            self.save()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "schema"):
                validate(self.root)

    def test_wrong_lineage_and_component_license_reject(self) -> None:
        original = copy.deepcopy(self.pin)
        for key, value in (("commit", "6b73449a6b3e228af9e4afe4f153a384e9b537b9"),
                           ("tree", "0"*40), ("repository", "https://github.com/another/repository"),
                           ("component_license", "Apache-2.0")):
            self.pin = copy.deepcopy(original)
            self.pin[key] = value
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(self.root)

    def test_changed_archive_bytes_reject(self) -> None:
        path = self.root / ARCHIVE / "src/models/attention/hash_utils.py"
        path.write_bytes(path.read_bytes()+b"\n# edited\n")
        with self.assertRaisesRegex(ValueError, "blob identity"):
            validate(self.root)

    def test_selected_blob_metadata_cannot_relabel_edited_source(self) -> None:
        name = "src/models/attention/hash_utils.py"
        path = self.root / ARCHIVE / name
        path.write_bytes(path.read_bytes()+b"\n# edited\n")
        data = path.read_bytes()
        self.pin["selected_blobs"][name].update(size=len(data), sha256=hashlib.sha256(data).hexdigest(),
            sha=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest())
        self.save()
        with self.assertRaisesRegex(ValueError, "selected blob metadata"):
            validate(self.root)

    def test_missing_archive_license_reject(self) -> None:
        (self.root / ARCHIVE / "LICENSE").unlink()
        with self.assertRaisesRegex(ValueError, "file/directory set"):
            validate(self.root)

    def test_extra_file_and_empty_directory_reject(self) -> None:
        for directory in (False, True):
            path = self.root / ARCHIVE / "not-selected"
            path.mkdir() if directory else path.write_text("extra")
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "file/directory set"):
                validate(self.root)
            path.rmdir() if directory else path.unlink()

    def test_closure_and_dependency_declaration_cannot_shrink(self) -> None:
        original = copy.deepcopy(self.pin)
        for where in ("selected_blobs", "dependency_source_files", "licenses"):
            self.pin = copy.deepcopy(original)
            if where == "selected_blobs":
                del self.pin[where]["src/models/attention/reformer_attention.py"]
            else:
                self.pin[where].pop()
            self.save()
            with self.subTest(where=where), self.assertRaises(ValueError):
                validate(self.root)

    def test_source_modes_are_verified(self) -> None:
        (self.root / ARCHIVE / "README.md").chmod(0o755)
        with self.assertRaisesRegex(ValueError, "executable mode"):
            validate(self.root)

    def test_file_archive_and_parent_symlink_traversal_reject(self) -> None:
        path = self.root / ARCHIVE / "README.md"
        data = path.read_bytes(); path.unlink(); path.symlink_to("LICENSE")
        with self.assertRaisesRegex(ValueError, "symlink"):
            validate(self.root)
        path.unlink(); path.write_bytes(data)
        archive = self.root / ARCHIVE
        archive.rename(archive.with_name("saved")); archive.symlink_to("saved", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink path traversal"):
            validate(self.root)

    def test_metadata_parent_symlink_reject(self) -> None:
        directory = self.root / "papers/source-objects"
        directory.rename(directory.with_name("saved")); directory.symlink_to("saved", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink path traversal"):
            validate(self.root)

    def test_unsafe_metadata_paths_reject(self) -> None:
        original = self.pin["upstream_inventory"]
        for value in ("../outside", "/tmp/outside", "papers//inventory.tsv", "papers/./inventory.tsv", "papers\\inventory.tsv", "papers/inventory.tsv\n"):
            self.pin["upstream_inventory"] = value; self.save()
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "unsafe/noncanonical"):
                validate(self.root)
        self.pin["upstream_inventory"] = original

    def test_raw_commit_and_tree_binding_reject(self) -> None:
        path = self.root / self.pin["commit_object"]
        path.write_bytes(path.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "commit object identity"):
            validate(self.root)

    def test_inventory_digest_and_coordinated_tree_mutation_reject(self) -> None:
        path = self.root / self.pin["upstream_inventory"]
        path.write_bytes(path.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "inventory digest"):
            validate(self.root)
        shutil.copyfile(ROOT / self.pin["upstream_inventory"], path)
        def mutate(text: str) -> str:
            lines = text.splitlines()
            for i, line in enumerate(lines):
                fields = line.split("\t")
                if len(fields) == 6 and fields[2] == "blob" and fields[-1] == "excluded":
                    fields[3] = "0"*40; lines[i] = "\t".join(fields); break
            return "\n".join(lines)+"\n"
        self.manifest(mutate)
        with self.assertRaisesRegex(ValueError, "Git tree identity"):
            validate(self.root)

    def test_selected_inventory_status_reject(self) -> None:
        self.manifest(lambda text: text.replace("\tselected\n", "\texcluded\n", 1))
        with self.assertRaisesRegex(ValueError, "selected subset"):
            validate(self.root)

    def test_each_required_notice_cannot_be_removed_even_with_pin_update(self) -> None:
        original = copy.deepcopy(self.pin)
        for name in original["preserved_notices"]:
            self.pin = copy.deepcopy(original)
            path = self.root / name; data = path.read_bytes(); path.unlink()
            del self.pin["preserved_notices"][name]; self.save()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "required SBSmyrf notice set"):
                validate(self.root)
            path.write_bytes(data)

    def test_notice_bytes_sha_and_application_reject(self) -> None:
        name = NOTICE_ROOT + "/SMYRF-GPL-3.0-LICENSE"
        path = self.root / name; data = path.read_bytes()
        path.write_bytes(data+b"edited")
        with self.assertRaisesRegex(ValueError, "notice identity"):
            validate(self.root)
        path.write_bytes(data)
        self.pin["preserved_notices"][name]["applies_to"] = ["src/models/attention/feature_maps_sb.py"]
        self.save()
        with self.assertRaisesRegex(ValueError, "notice application"):
            validate(self.root)

    def test_notice_origin_and_license_cannot_relabel(self) -> None:
        name = NOTICE_ROOT + "/SMYRF-GPL-3.0-LICENSE"
        original = copy.deepcopy(self.pin)
        for key, value in (("license", "MIT"), ("commit", "0"*40), ("source_path", "../LICENSE")):
            self.pin = copy.deepcopy(original); self.pin["preserved_notices"][name][key] = value; self.save()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "notice origin/license"):
                validate(self.root)

    def test_notice_symlink_extra_file_and_empty_directory_reject(self) -> None:
        path = self.root / NOTICE_ROOT / "Performer-MIT-LICENSE"
        data = path.read_bytes(); path.unlink(); path.symlink_to("Reformer-MIT-LICENSE")
        with self.assertRaisesRegex(ValueError, "symlink"):
            validate(self.root)
        path.unlink(); path.write_bytes(data)
        for directory in (False, True):
            extra = self.root / NOTICE_ROOT / "extra"
            extra.mkdir() if directory else extra.write_text("extra")
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                validate(self.root)
            extra.rmdir() if directory else extra.unlink()

    def test_nonregular_pin_and_notice_reject_without_reading(self) -> None:
        for name in (PIN_PATH, NOTICE_ROOT + "/SMYRF-GPL-3.0-LICENSE"):
            path = self.root / name; data = path.read_bytes(); path.unlink(); os.mkfifo(path)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "regular file"):
                validate(self.root)
            path.unlink(); path.write_bytes(data)

    def test_derivation_cannot_disappear_or_redirect(self) -> None:
        original = copy.deepcopy(self.pin)
        for mutation in ("empty", "url", "license", "origin", "slice", "digest", "bool_count"):
            self.pin = copy.deepcopy(original)
            record = self.pin["derivation_evidence"][0]
            if mutation == "empty": self.pin["derivation_evidence"] = []
            elif mutation == "url": record["source_url"] = record["source_url"].replace(record["commit"], "master")
            elif mutation == "license": record["license_notice"] = NOTICE_ROOT + "/Fly-Apache-LICENSE"
            elif mutation == "origin": record["sha"] = "0"*40
            elif mutation == "slice": record["archive_start_line"] = 3
            elif mutation == "digest": record["matched_lines_sha256"] = "0"*64
            else: record["line_count"] = True
            self.save()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(self.root)

    def test_cli_emits_source_only_and_invalidates_stale_evidence(self) -> None:
        tools = self.root / "tools"; tools.mkdir()
        for name in ("check_scatterbrain_sources.py", "check_scatterbrain_sbsmyrf_sources.py"):
            shutil.copyfile(ROOT / "tools" / name, tools / name)
        output = self.root / "evidence"
        command = [sys.executable, "-B", str(tools / "check_scatterbrain_sbsmyrf_sources.py"), "--emit-dir", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = output / "scatterbrain_sbsmyrf_source_evidence.json"
        data = json.loads(evidence.read_text())
        self.assertEqual(data["status"], "source_identity_pass")
        for key in ("build_validated", "dependencies_installed", "numerical_runtime_validated", "paper_claims_validated", "gpu_validated"):
            self.assertIs(data[key], False)
        (self.root / ARCHIVE / "README.md").write_text("changed")
        failure = subprocess.run(command, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(failure.returncode, 0)
        self.assertFalse(evidence.exists())
        self.assertNotIn("SOURCE_PASS", failure.stdout)


if __name__ == "__main__":
    unittest.main()
