"""Runtime/syntax contracts for both maintained SLIDE Python example trees."""
from __future__ import annotations

import importlib.util
import py_compile
import tempfile
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np

# TRACE_TEST_ID: SLIDE-PYTHON-EXAMPLE-CONTRACTS

ROOT = Path(__file__).resolve().parents[2]
TREES = (
    ROOT / "ports" / "slide-original" / "python_examples",
    ROOT / "ports" / "slide-optimized-avx512" / "python_examples",
)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exercise_tree(tree: Path, tag: str) -> None:
    config_module = _load_module(
        tree / "config.py", f"{tag}_config"
    )
    util = _load_module(tree / "util.py", f"{tag}_util")

    config = config_module.config
    assert isinstance(config.feature_dim, int)
    assert isinstance(config.n_classes, int)
    assert isinstance(config.lr, float)
    assert isinstance(config.data_path_train, str)

    with tempfile.TemporaryDirectory() as tmp:
        dataset = Path(tmp) / "tiny.txt"
        dataset.write_text(
            "4 6 4\n"
            "0,2 1:1.0 4:0.5\n"
            "1 2:0.25 5:2.0\n"
            "2,3 0:1.5 3:0.75\n"
            "0 1:0.1 2:0.2\n",
            encoding="utf-8",
        )
        files = [str(dataset)]

        dense_gen = util.data_generator(
            files, batch_size=2, n_classes=4
        )
        idxs, vals, targets = next(dense_gen)
        assert idxs == [(0, 1), (0, 4), (1, 2), (1, 5)]
        assert vals == [1.0, 0.5, 0.25, 2.0]
        assert targets.dtype == np.float32
        assert targets.shape == (2, 4)
        assert np.isclose(targets[0, 0], 0.5)
        assert np.isclose(targets[0, 2], 0.5)
        assert np.isclose(targets[1, 1], 1.0)

        sampled_gen = util.data_generator_ss(
            files,
            batch_size=2,
            n_classes=4,
            max_label=2,
        )
        sidxs, svals, sample_labels = next(sampled_gen)
        assert sidxs == idxs
        assert svals == vals
        assert sample_labels[0] == [0, 2]
        assert sample_labels[1] == [1, 4]

        test_gen = util.data_generator_tst(files, batch_size=2)
        tidxs, tvals, labels = next(test_gen)
        assert tidxs == idxs
        assert tvals == vals
        assert labels == [[0, 2], [1]]

    for script in (
        "config.py",
        "util.py",
        "example_full_softmax.py",
        "example_sampled_softmax.py",
    ):
        py_compile.compile(str(tree / script), doraise=True)


def test_both_slide_example_trees() -> None:
    for index, tree in enumerate(TREES):
        _exercise_tree(tree, f"slide_examples_{index}")


if __name__ == "__main__":
    test_both_slide_example_trees()
    print("TRACE_TEST_PASS SLIDE-PYTHON-EXAMPLE-CONTRACTS")
