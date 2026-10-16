#!/usr/bin/env python3
"""Run frozen later SBSmyrf CPU composition with independent mutation-tested oracles.

Fetches an exact upstream commit ephemerally; does not vendor adapted GPL source,
validate a maintained port, install dependencies, or execute native/GPU kernels.
Requires CPU Torch, NumPy, einops and tqdm in the selected Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PIN = "74cc6b14347b4edcf5c49098e80a9ef47c0b21bf"
TREE = "1821d435ef810ce88038ef81965c65cd0ac8c6bc"
REPOSITORY = "https://github.com/HazyResearch/fly.git"
ENTRY = "src/models/attention/sbsmyrf_attention.py"
CLOSURE = {'src/models/attention/batching_utils.py': {'sha': '5c73f9e114233c1b630862ccdcc9c1d9102848a1',
                                            'sha256': '44498a101debbeeccd43d2d9985203db1c79c99b41aded1bf0c982507d664fa4',
                                            'size': 557},
 'src/models/attention/feature_maps_sb.py': {'sha': '7c6fe0c52dd806ef34424a027415121850a357f2',
                                             'sha256': '9de3386dd31fdc9818e0d1e1202a027099595528c04bb7212244a1bcefdfb164',
                                             'size': 7504},
 'src/models/attention/hash_utils.py': {'sha': '47224c645c07457042bd4ef4cca3844d6242dd17',
                                        'sha256': '91012c5b7c92665517da4e98bdec386fa8211c2cd294dd6b96c326d08c061b70',
                                        'size': 10317},
 'src/models/attention/mask_utils.py': {'sha': '011443cfcac494484ba5f1b67b9ef2a05f97d52b',
                                        'sha256': '07bdff72a0793ad51142ee13abb380f9358e15cf592777c7839d210cfc609bcc',
                                        'size': 755},
 'src/models/attention/projection_utils.py': {'sha': '5f103649a7a664a8303108611b6261965a550a0a',
                                              'sha256': 'ec2ebfae61ea226e78bf9da9c54d0bfd9d005c57320d6671ae6fafcbf77ee10b',
                                              'size': 1255},
 'src/models/attention/reformer_attention.py': {'sha': 'c15b54dd5d993ffb314fc6e0288a9118a465e601',
                                                'sha256': 'c6a9dad003279b00ec1f062287e2cade903882e86dc834cebc6c7fa10e0d7b93',
                                                'size': 15307},
 'src/models/attention/sbsmyrf_attention.py': {'sha': '882c3e5e8410279ec858a2f0b8a057fa4da7dbe0',
                                               'sha256': '709614aa5b2f0363591d23f068ceefc9a1a42b227c879ff636450f631e78681f',
                                               'size': 14896},
 'src/models/attention/scatterbrain_utils.py': {'sha': '61f022b9d5a482669fe4eeacec4521ebf6490ac4',
                                                'sha256': '0dbca5f571b63c91cd1f7920717cf68d5cb3226884306f00beb31c1db030b33a',
                                                'size': 1923},
 'src/models/modules/masking.py': {'sha': '784e4b10670849a227cedcec8f65881e405fe24c',
                                   'sha256': 'f20f3c7004015752260050281f9dc09d214566a54d5b45c43af86c66a8d21755',
                                   'size': 7524},
 'src/ops/permutation.py': {'sha': 'c09e9cbb1c955cbb50c23767444225617699e0e1',
                            'sha256': 'd97893e27f11767d319cde52516ebb0c4f4486974222a2e686dce17cdf7ea21a',
                            'size': 1057},
 'src/utils/padding.py': {'sha': '4889c845b0c5335b2c68573ef9d83ae4a55b57b4',
                          'sha256': '142fd3351f4bafb010afec02be17739154278527d270cb75604da67572826030',
                          'size': 905}}
# These four imported package initializers are empty, separately hash-guarded.
PACKAGES = ["src/__init__.py", "src/models/__init__.py", "src/models/modules/__init__.py", "src/utils/__init__.py"]
FIXTURE = ROOT / "tests/traceability/test_scatterbrain_later_cpu.py"
MUTATIONS = ("missing_subtraction", "duplicate_correction", "normalizer", "common_scale",
             "output_nan", "output_inf", "output_-inf", "weights_nan", "weights_inf", "weights_-inf")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def verify_source(source: Path) -> dict[str, object]:
    require(not source.is_symlink() and source.is_dir(), "invalid source directory")
    require(git(source, "rev-parse", "HEAD") == PIN, "upstream commit differs")
    require(git(source, "rev-parse", "HEAD^{tree}") == TREE, "upstream tree differs")
    require(not git(source, "status", "--porcelain", "--untracked-files=all"), "source checkout is dirty")
    for name, record in CLOSURE.items():
        path = source / name
        require(all(not parent.is_symlink() for parent in [path, *path.parents] if parent.is_relative_to(source)), "source symlink forbidden")
        require(path.is_file() and path.stat().st_size == record["size"] and sha256(path) == record["sha256"], "source closure SHA256 differs: " + name)
        require(git(source, "rev-parse", "HEAD:" + name) == record["sha"], "source Git blob differs: " + name)
    for name in PACKAGES:
        require((source / name).is_file() and not (source / name).is_symlink() and (source / name).read_bytes() == b"", "package initializer changed: " + name)
    return {"commit": PIN, "tree": TREE, "repository": REPOSITORY,
            "core_closure": CLOSURE, "empty_package_initializers": PACKAGES}


def fetch(source: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "core.hooksPath=/dev/null", "fetch", "--quiet", "--depth=1", REPOSITORY, PIN], check=True, timeout=120)
    subprocess.run(["git", "-C", str(source), "-c", "core.hooksPath=/dev/null", "checkout", "--quiet", "--detach", PIN], check=True)


def mutate(source: Path, target: Path, name: str) -> dict[str, str]:
    # Copies exist only in the temporary runtime directory and are removed on exit.
    for relative in [*CLOSURE, *PACKAGES]:
        dest = target / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, dest)
    path = target / ENTRY; before = path.read_text()
    edits = {
        "missing_subtraction": [("dots = torch.exp(inner - dots_logsumexp) - dots_prime * dots_prime_scale", "dots = torch.exp(inner - dots_logsumexp)")],
        "duplicate_correction": [("inner = inner - torch.log(dup_counts.float())", "inner = inner"), ("dots_prime = dots_prime / dup_counts", "dots_prime = dots_prime")],
        "normalizer": [("out_normalized = out / normalization.clamp_min(1e-6)", "out_normalized = out / (normalization.clamp_min(1e-6) + 0.125)")],
        # This preserves the output ratio. The intended rejection must come
        # from the raw numerator gate, before a later returned-weight check.
        "common_scale": [("out_normalized = out / normalization.clamp_min(1e-6)", "out = out * 2\n        normalization = normalization * 2\n        out_normalized = out / normalization.clamp_min(1e-6)")],
    }
    if name.startswith("output_"):
        kind = name.removeprefix("output_")
        anchor = "return out_normalized, attn"
        replacements = [(anchor, "out_normalized[-1, -1, -1, -1] = float('" + kind + "')\n        " + anchor)]
    elif name.startswith("weights_"):
        kind = name.removeprefix("weights_")
        anchor = "attn = attn_unnormalized / normalization.clamp_min(1e-6)"
        replacements = [(anchor, anchor + "\n            attn[-1, q_seqlen_og - 1, 1] = float('" + kind + "')")]
    else:
        replacements = edits[name]
    after = before
    for old, new in replacements:
        require(after.count(old) == 1, "mutation anchor not unique: " + name)
        after = after.replace(old, new)
    require(before != after, "mutation did not change source")
    path.write_text(after)
    return {"path": ENTRY, "baseline_sha256": hashlib.sha256(before.encode()).hexdigest(), "mutant_sha256": sha256(path)}


def execute(source: Path, output: Path, python: Path, label: str) -> tuple[int, dict[str, object]]:
    artifact = output / (label + ".json"); artifact.unlink(missing_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [str(python), "-I", "-B", str(FIXTURE), "--source", str(source), "--emit", str(artifact)]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
    (output / (label + ".log")).write_text(result.stdout + result.stderr)
    require(artifact.is_file(), "runtime failed without evidence: " + label + "; see " + label + ".log")
    data = json.loads(artifact.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))
    require(isinstance(data, dict), "invalid child evidence")
    return result.returncode, data


def run(source: Path, output: Path, python: Path) -> dict[str, object]:
    fixture_hash, driver_hash = sha256(FIXTURE), sha256(Path(__file__))
    checkout = {"commit": git(ROOT, "rev-parse", "HEAD"), "tree": git(ROOT, "rev-parse", "HEAD^{tree}"),
                "status_porcelain": git(ROOT, "status", "--porcelain", "--untracked-files=all")}
    identity = verify_source(source)
    code, baseline = execute(source, output, python, "baseline")
    require(code == 0 and baseline.get("status") == "supported_cases_passed_with_separate_source_failure_observations", "baseline rejected")
    require(baseline.get("source_sha256") == {name: record["sha256"] for name, record in CLOSURE.items()}, "runtime source hashes differ")
    require(baseline.get("source_pin") == PIN and baseline.get("source_tree") == TREE, "runtime source identity differs")
    mutations = []
    with tempfile.TemporaryDirectory(prefix="scatterbrain-mutants-") as temporary:
        for name in MUTATIONS:
            mutant = Path(temporary) / name
            change = mutate(source, mutant, name)
            code, evidence = execute(mutant, output, python, name)
            reason = evidence.get("reason", "")
            expected = "nonfinite actual value" if name.startswith(("output_", "weights_")) else ("output independent oracle" if name == "normalizer" else "numerator independent oracle")
            require(code == 2 and evidence.get("status") == "oracle_rejected" and isinstance(reason, str) and expected in reason, "mutation did not reject for intended reason: " + name)
            mutations.append({"name": name, "status": "rejected", "reason": reason, **change})
    verify_source(source)
    require(sha256(FIXTURE) == fixture_hash and sha256(Path(__file__)) == driver_hash, "fixture/driver changed during execution")
    cases = baseline.get("cases")
    if not isinstance(cases, list):
        raise ValueError("baseline cases missing")
    return {"status": "frozen_reference_cpu_pass", "source_identity": identity, "checkout": checkout, "fixture_sha256": fixture_hash,
            "driver_sha256": driver_hash, "baseline_evidence_sha256": sha256(output / "baseline.json"),
            "cases": len(cases), "mutations": mutations,
            "boundary": {"actual_whole_module": "noncausal SBSmyrfAttention.forward", "actual_hash": "CPU XBOXPLUS/E2LSH rank clustering", "oracle": "independent scalar fixed-feature hybrid and permutation-derived support", "hash_numeric_correctness_independently_validated": False, "gradient_scope": "conditional support/projection, six QKV coordinates per one/three rounds", "denominator_scope": "unclamped positive rows and zero masked rows", "upstream_source_repaired": False, "maintained_port_validated": False, "general_archive_ci_maintenance_claim": False, "native_extension_executed": False, "gpu_executed": False, "paper_theorem_or_statistical_quality_validated": False}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="optional existing clean exact-pin checkout; default ephemeral pinned fetch")
    parser.add_argument("--emit-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    output = args.emit_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    summary = output / "scatterbrain_later_cpu_evidence.json"
    summary.unlink(missing_ok=True)
    if args.source_dir:
        evidence = run(args.source_dir.resolve(), output, args.python)
    else:
        with tempfile.TemporaryDirectory(prefix="scatterbrain-source-") as temporary:
            source = Path(temporary) / "upstream"; fetch(source)
            evidence = run(source, output, args.python)
    summary.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print("SCATTERBRAIN_LATER_CPU_PASS cases=" + str(evidence["cases"]) + " mutations=" + str(len(MUTATIONS)) + " native=False gpu=False")


if __name__ == "__main__":
    main()
