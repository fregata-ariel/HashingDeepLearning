#!/usr/bin/env python3
"""Projection lifecycle regressions; sampled features use a scalar FP64 oracle.

Preimage mode executes a separately selected frozen initial/current source.
TRACE_TEST_ID: SCATTERBRAIN-FEATURE-STATE-CPU
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import io
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import torch


def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)


def raises(action: Callable[[], object], text: str) -> None:
    try:
        action()
    except (ValueError, RuntimeError, OverflowError) as error:
        require(text in str(error), 'unexpected exception: '+str(error))
        return
    raise AssertionError('missing rejection: '+text)


def snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    stream.seek(0)
    return torch.load(stream, weights_only=True)


def identical(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def raw_oracle(x: torch.Tensor, matrix: torch.Tensor, temperature: float, cosh: bool) -> torch.Tensor:
    inputs, rows = x.double().tolist(), matrix.to(x.dtype).double().tolist()
    result = []
    for batch in inputs:
        values = []
        for token in batch:
            norm = sum(v*v for v in token)
            dots = [math.sqrt(temperature)*sum(a*b for a, b in zip(token, row)) for row in rows]
            if cosh:
                dots += [-v for v in dots]
            values.append([math.exp(dot-temperature*norm/2)/math.sqrt(len(dots)) for dot in dots])
        result.append(values)
    return torch.tensor(result, dtype=torch.float64)


def regression(features: ModuleType, projection: ModuleType) -> dict[str, int]:
    cls = features.SBPerformerFeatures
    x = torch.tensor([[[.2, -.3, .1], [.4, .1, -.2]]], dtype=torch.float64)
    counts = {'schedules': 0, 'feature_oracles': 0, 'checkpoints': 0, 'rejections': 0}
    # Expected draw requests are fixed examples, not the implementation's modulo.
    for period, draws in ((1, [True, True, True, True]), (2.0, [True, False, True, False, True]),
                          (3, [True, False, False, True, False, False, True]),
                          (math.inf, [True, False, False, False])):
        torch.manual_seed(106)
        start = torch.random.get_rng_state().clone()
        expected = []
        matrix = None
        for draw in draws:
            if draw:
                matrix = projection.gaussian_orthogonal_random_matrix(4, 3, device='cpu', dtype=torch.float64)
            expected.append((matrix.clone(), torch.random.get_rng_state().clone()))
        torch.random.set_rng_state(start)
        module = cls(3, 8, redraw=period, dtype=torch.float32).to(dtype=torch.float64, device='cpu')
        require(module.projection_matrix.shape == (4, 3) and module._calls == -1, 'cold valid placeholder')
        require(torch.equal(start, torch.random.get_rng_state()), 'constructor consumed RNG')
        require(module._projection_initialized.dtype == torch.bool and module._feature_map_calls.dtype == torch.int64 and module._redraw_period.dtype == torch.int64, 'module.to changed lifecycle metadata dtype')
        for i, (matrix, rng) in enumerate(expected):
            module.new_feature_map('cpu')
            require(module._calls == i and torch.equal(module.projection_matrix, matrix), 'draw request sequence')
            require(module.projection_matrix.dtype == torch.float64, 'redraw ignored current buffer dtype')
            require(torch.equal(rng, torch.random.get_rng_state()), 'sample RNG continuation')
            for encode in (module.forward_queries, module.forward_keys):
                value, scale = encode(x)
                require(torch.allclose(value.double()*scale.double().exp(), raw_oracle(x, matrix, module.softmax_temp, True), atol=2e-14, rtol=2e-13), 'scalar raw feature oracle')
                counts['feature_oracles'] += 1
            require(module._calls == i and torch.equal(rng, torch.random.get_rng_state()), 'q/key encoding advanced cadence')
        counts['schedules'] += 1
    # Cold direct encode initializes even deterministic eval; warm eval freezes.
    torch.manual_seed(207)
    cold = cls(3, 8, redraw=2, deterministic_eval=True, dtype=torch.float64).eval()
    before = torch.random.get_rng_state().clone()
    cold.forward_keys(x)
    require(cold._calls == 0 and not torch.equal(before, torch.random.get_rng_state()), 'cold eval did not initialize')
    first, rng = cold.projection_matrix.clone(), torch.random.get_rng_state().clone()
    cold.forward_queries(x)
    for _ in range(3): cold.new_feature_map('cpu')
    require(cold._calls == 0 and torch.equal(first, cold.projection_matrix) and torch.equal(rng, torch.random.get_rng_state()), 'warm eval advances cadence')
    cold.train(); cold.new_feature_map('cpu')
    require(cold._calls == 1 and torch.equal(first, cold.projection_matrix), 'eval/train cadence transition')
    saved, rng = snapshot(cold), torch.random.get_rng_state().clone()
    cold.new_feature_map('cpu')
    expected, next_rng = cold.projection_matrix.clone(), torch.random.get_rng_state().clone()
    restored = cls(3, 8, redraw=1, deterministic_eval=True, dtype=torch.float64)
    before = torch.random.get_rng_state().clone()
    restored.load_state_dict(saved)
    require(restored.training and restored.redraw == 2 and restored._calls == 1, 'checkpoint cadence or training policy')
    require(torch.equal(before, torch.random.get_rng_state()), 'warm load drew projection')
    torch.random.set_rng_state(rng)
    restored.new_feature_map('cpu')
    require(torch.equal(expected, restored.projection_matrix) and torch.equal(next_rng, torch.random.get_rng_state()), 'warm resume matrix/RNG continuation')
    counts['checkpoints'] += 1
    # A warm due redraw after .to also uses buffer dtype, not constructor kwargs.
    typed = cls(3, 8, redraw=1, dtype=torch.float32)
    typed.new_feature_map('cpu'); typed.to(dtype=torch.float64)
    rng = torch.random.get_rng_state().clone()
    expected = projection.gaussian_orthogonal_random_matrix(4, 3, device='cpu', dtype=torch.float64)
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng); typed.new_feature_map(None)
    require(typed.projection_matrix.dtype == torch.float64 and torch.equal(expected, typed.projection_matrix) and torch.equal(expected_rng, torch.random.get_rng_state()), 'warm dtype migration')
    # Cold serialization and nested-module loading never sample.
    parent = torch.nn.Module(); parent.add_module('features', cls(3, 8, redraw=3, dtype=torch.float64))
    saved = snapshot(parent)
    target = torch.nn.Module(); target.add_module('features', cls(3, 8, redraw=1, dtype=torch.float64))
    rng = torch.random.get_rng_state().clone(); target.load_state_dict(saved)
    child = target.features
    require(child._calls == -1 and child.redraw == 3 and torch.equal(rng, torch.random.get_rng_state()), 'nested cold load sampled')
    cold_state = snapshot(child)
    torch.random.set_rng_state(rng); child.forward_queries(x)
    expected, expected_rng = child.projection_matrix.clone(), torch.random.get_rng_state().clone()
    other = cls(3, 8, redraw=1, dtype=torch.float64); other.load_state_dict(cold_state)
    torch.random.set_rng_state(rng); other.forward_queries(x)
    require(torch.equal(expected, other.projection_matrix) and torch.equal(expected_rng, torch.random.get_rng_state()), 'cold checkpoint sampling continuation')
    counts['checkpoints'] += 2
    # Legacy projection-only states: warm shape resumes at0; any all-zero legacy
    # matrix is explicitly cold, including an ambiguous shape-correct matrix.
    for matrix, initialized in ((expected, True), (torch.zeros(3, 8), False), (torch.zeros(4, 3), False)):
        module = cls(3, 8, redraw=3, dtype=torch.float64)
        rng = torch.random.get_rng_state().clone(); module.load_state_dict({'projection_matrix': matrix})
        require(module._initialized == initialized and module._calls == (0 if initialized else -1) and module.redraw == 3, 'legacy migration policy')
        require(module.projection_matrix.shape == (4, 3) and torch.equal(rng, torch.random.get_rng_state()), 'legacy load sampled')
        counts['checkpoints'] += 1
    # New metadata deliberately distinguishes an initialized zero projection.
    zero = snapshot(other); zero['projection_matrix'].zero_(); zero['_redraw_period'].zero_()
    warmzero = cls(3, 8, deterministic_eval=True, dtype=torch.float64).eval(); warmzero.load_state_dict(zero)
    rng = torch.random.get_rng_state().clone(); value, scale = warmzero.forward_queries(x)
    require(warmzero._initialized and torch.equal(rng, torch.random.get_rng_state()), 'explicit warm-zero checkpoint resampled')
    require(torch.allclose(value*scale.exp(), raw_oracle(x, zero['projection_matrix'], warmzero.softmax_temp, True), atol=2e-14, rtol=2e-13), 'warm-zero raw features')
    for bad in (False, True, 0, -1, math.nan, -math.inf, math.pi, 1e20, 2**63, '2', b'2'):
        rng = torch.random.get_rng_state().clone(); raises(lambda: cls(3, 8, redraw=bad), 'redraw')
        require(torch.equal(rng, torch.random.get_rng_state()), 'invalid constructor consumed RNG')
        counts['rejections'] += 1
        before = snapshot(warmzero)
        raises(lambda: setattr(warmzero, 'redraw', bad), 'redraw')
        require(identical(before, snapshot(warmzero)) and torch.equal(rng, torch.random.get_rng_state()), 'invalid setter mutated lifecycle/RNG')
        counts['rejections'] += 1
    # A public period change keeps its place in the stream; the next due draw
    # must consume exactly the original sampler's RNG sequence.
    switch = cls(3, 8, redraw=math.inf, dtype=torch.float64)
    switch.new_feature_map(None); switch.new_feature_map(None); switch.redraw = 2
    require(switch._calls == 1 and switch.redraw == 2, 'period setter reset cadence')
    rng = torch.random.get_rng_state().clone()
    expected = projection.gaussian_orthogonal_random_matrix(4, 3, device='cpu', dtype=torch.float64)
    expected_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng); switch.new_feature_map(None)
    require(torch.equal(expected, switch.projection_matrix) and torch.equal(expected_rng, torch.random.get_rng_state()), 'period setter next sample/RNG')
    boundary = cls(3, 8, redraw=2**63-1, dtype=torch.float64)
    require(boundary.redraw == 2**63-1, 'largest supported period')
    maximum = snapshot(other); maximum['_feature_map_calls'].fill_(2**63-1)
    boundary.load_state_dict(maximum)
    before, rng = snapshot(boundary), torch.random.get_rng_state().clone()
    raises(lambda: boundary.new_feature_map(None), 'counter exceeds int64')
    require(identical(before, snapshot(boundary)) and torch.equal(rng, torch.random.get_rng_state()), 'counter overflow mutated state/RNG')
    counts['rejections'] += 1
    valid = snapshot(other)
    for fault in ('partial', 'dtype', 'rank', 'period', 'counter', 'cold_nonzero', 'nonfinite', 'shape'):
        invalid = {key: value.clone() for key, value in valid.items()}
        if fault == 'partial': invalid.pop('_feature_map_calls')
        if fault == 'dtype': invalid['_feature_map_calls'] = torch.tensor(0.0)
        if fault == 'rank': invalid['_projection_initialized'] = torch.tensor([True])
        if fault == 'period': invalid['_redraw_period'] = torch.tensor(-1)
        if fault == 'counter': invalid['_feature_map_calls'] = torch.tensor(-1)
        if fault == 'cold_nonzero': invalid['_projection_initialized'] = torch.tensor(False); invalid['_feature_map_calls'] = torch.tensor(-1)
        if fault == 'nonfinite': invalid['projection_matrix'][0, 0] = math.nan
        if fault == 'shape': invalid['projection_matrix'] = torch.zeros(3, 8)
        before, rng = snapshot(other), torch.random.get_rng_state().clone()
        raises(lambda: other.load_state_dict(invalid), 'Error(s) in loading state_dict')
        require(identical(before, snapshot(other)) and torch.equal(rng, torch.random.get_rng_state()), 'rejected state mutated live buffers/RNG')
        counts['rejections'] += 1
    return counts


def preimage(features: ModuleType) -> dict[str, bool]:
    cls = features.SBPerformerFeatures
    cold = cls(3, 8, deterministic_eval=True).eval(); cold.new_feature_map('cpu')
    require(cold.projection_matrix.shape == (3, 8), 'preimage placeholder changed')
    raises(lambda: cold.forward_queries(torch.zeros(1, 2, 3)), 'subscript d has size')
    warm = cls(3, 8, redraw=1, dtype=torch.float32); warm.new_feature_map('cpu')
    first = warm.projection_matrix.clone(); rng = torch.random.get_rng_state().clone(); warm.new_feature_map('cpu')
    require(warm.redraw == math.pi and torch.equal(first, warm.projection_matrix) and torch.equal(rng, torch.random.get_rng_state()), 'preimage redraw changed')
    saved = snapshot(warm); raises(lambda: cls(3, 8).load_state_dict(saved), 'size mismatch')
    moved = cls(3, 8, dtype=torch.float32).to(dtype=torch.float64); moved.new_feature_map('cpu')
    require(moved.projection_matrix.dtype == torch.float32, 'preimage stale dtype changed')
    return {'cold_eval_shape_failure': True, 'requested_redraw_ignored': True, 'fresh_warm_load_failure': True, 'stale_dtype': True}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--port-root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True); parser.add_argument('--preimage', action='store_true')
    args = parser.parse_args(); args.evidence.unlink(missing_ok=True)
    sys.dont_write_bytecode = True
    root = args.port_root.resolve(); sys.path.insert(0, str(root)); torch.set_num_threads(1)
    features = importlib.import_module('src.models.modules.attention.feature_maps_sb')
    projection = importlib.import_module('src.models.modules.attention.projection_utils')
    source = Path(features.__file__).resolve()
    require(source.is_relative_to(root), 'source outside requested root')
    result = preimage(features) if args.preimage else regression(features, projection)
    mutations = []
    if not args.preimage:
        edits = {
            'cold_eval_skip': ('if self._initialized and self.deterministic_eval and not self.training:', 'if self.deterministic_eval and not self.training:'),
            'ignore_period': ('period = self._validate_redraw(redraw)', 'period = self._validate_redraw(math.inf)'),
            'stale_dtype': ('dtype=self.projection_matrix.dtype)', "dtype=self.factory_kwargs['dtype'])"),
            'reset_resume_counter': ('        super()._load_from_state_dict(local, prefix, local_metadata, strict,', '        local[metadata_keys[1]] = self._feature_map_calls.new_tensor(0)\n        super()._load_from_state_dict(local, prefix, local_metadata, strict,'),
            'redraw_on_encoding': ('        if not self._initialized:\n            self.new_feature_map(x.device)', '        if True:\n            self.new_feature_map(x.device)'),
        }
        text = source.read_text()
        for name, (old, new) in edits.items():
            require(old in text, 'mutation target missing: '+name)
            namespace = dict(features.__dict__); exec(compile(text.replace(old, new), str(source)+'::'+name, 'exec'), namespace)
            mutant = ModuleType(name); mutant.SBPerformerFeatures = namespace['SBPerformerFeatures']
            try: regression(mutant, projection)
            except AssertionError: mutations.append(name)
            else: raise AssertionError('surviving lifecycle mutant: '+name)
    evidence = {'status': 'preimage_reproduced' if args.preimage else 'passed', 'checks': result, 'mutants_rejected': mutations,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'runtime': {'torch': str(torch.__version__), 'device': 'cpu'}, 'global_rng_serialized': False, 'gpu_validated': False}
    args.evidence.parent.mkdir(parents=True, exist_ok=True); args.evidence.write_text(json.dumps(evidence, indent=2)+'\n')
    print('SCATTERBRAIN_FEATURE_STATE_CPU '+json.dumps(evidence, sort_keys=True))


if __name__ == '__main__': main()
