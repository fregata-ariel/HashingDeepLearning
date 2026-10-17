from __future__ import annotations

from typing import Callable, Protocol, TypeAlias, TypedDict, cast
from collections.abc import Mapping

# Adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/feature_maps/fourier_features.py
import math
import torch

from einops import rearrange

from src.models.modules.attention.projection_utils import gaussian_orthogonal_random_matrix


FeatureOutput: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class _FactoryKwargs(TypedDict):
    device: torch.device | str | None
    dtype: torch.dtype | None


class _FeatureConstructor(Protocol):
    def __call__(self, query_dims: int, *args: object, **kwargs: object) -> FeatureMap: ...


# from fast_transformers.feature_maps.base import FeatureMap
# For compatibility with our old code (using pytorch_fast_transformers 0.3.0), we copy the
# FeatureMap class here.
# Will be removed once we migrate to our new code
class FeatureMap(torch.nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims: int) -> None:
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device: torch.device | str | None) -> None:
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x: torch.Tensor) -> FeatureOutput:
        """Encode the queries `x` using this feature map."""
        return cast(FeatureOutput, self(x))

    def forward_keys(self, x: torch.Tensor) -> FeatureOutput:
        """Encode the keys `x` using this feature map."""
        return cast(FeatureOutput, self(x))

    def forward(self, x: torch.Tensor) -> FeatureOutput:
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args: object, **kwargs: object) -> Callable[[int], FeatureMap]:
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.
        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims: int) -> FeatureMap:
            # The original factory forwards a subclass-specific constructor;
            # nn.Module does not expose that variadic signature to the checker.
            return cast(_FeatureConstructor, cls)(query_dims, *args, **kwargs)
        return inner


def softmax_kernel(data: torch.Tensor, *, projection_matrix: torch.Tensor,
                   is_query: bool, softmax_temp: float | None = None, eps: float = 0.0,
                   cosh: bool = True, return_log: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """For key, we expect shape (..., S, D) where S is the sequence dimension
    return_log: return the log of the features (i.e. don't apply exp at the end).
    """
    if return_log and eps != 0:
        raise NotImplementedError('return_log is not compatible with nonzero eps')
    d = data.shape[-1]
    m = projection_matrix.shape[0] if not cosh else 2 * projection_matrix.shape[0]
    if softmax_temp is None:
        softmax_temp = 1 / math.sqrt(d)
    data_normalizer = math.sqrt(softmax_temp)
    projection_matrix = projection_matrix.type_as(data)
    data_dash = torch.einsum('...id,jd->...ij', data, data_normalizer * projection_matrix)
    diag_data = data.square().sum(dim=-1, keepdim=True) / 2 * (data_normalizer ** 2)
    if cosh:
        # We use the cosh feature map from the Performer paper, which effectively means
        # concatenating data_dash and -data_dash
        data_dash = torch.cat([data_dash, -data_dash], dim=-1)
    if is_query:
        log_scale = -diag_data + torch.amax(data_dash, dim=-1, keepdim=True) - math.log(m) / 2
        # TD: The correct scaling is torch.exp(data_dash - diag_data)
        data_dash_log = data_dash - torch.amax(data_dash, dim=-1, keepdim=True)
        if not return_log:
            data_dash = torch.exp(data_dash_log) + eps / math.sqrt(m)
    else:
        data_dash_m_diag = data_dash - diag_data - math.log(m) / 2
        log_scale = torch.amax(data_dash_m_diag, dim=(-1, -2), keepdim=True)
        data_dash_log = data_dash_m_diag - log_scale
        if not return_log:
            data_dash = torch.exp(data_dash_log) + eps / math.sqrt(m)
    return (data_dash if not return_log else data_dash_log).type_as(data), log_scale


class SBPerformerFeatures(FeatureMap):
    """Random Fourier Features for the RBF kernel according to [1].
    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.
    Arguments
    ---------
        query_dims: int, The input query dimensions in order to sample
                          the noise matrix
        n_features: int, The size of the feature map (should be divisible by 2)
                (default: int(query_dims * log(query_dims)))
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dims))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        redraw: positive integer/integral float request period, or +inf to
                sample once and freeze (default: 1; int64 range)
        deterministic_eval: bool, Only redraw the random matrix during training
                            (default: False)
    """
    projection_matrix: torch.Tensor
    _projection_initialized: torch.Tensor
    _feature_map_calls: torch.Tensor
    _redraw_period: torch.Tensor
    _version = 2

    def __init__(self, query_dims: int, n_features: int | None = None, ortho_scaling: int = 0,
                 softmax_temp: float | None = None, orthogonal: bool = False, cosh: bool = True,
                 redraw: int | float = 1, deterministic_eval: bool = False, eps: float = 0.0,
                 device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        period = self._validate_redraw(redraw)
        if query_dims <= 0:
            raise ValueError('query_dims must be positive')
        factory_kwargs: _FactoryKwargs = {'device': device, 'dtype': dtype}
        super().__init__(query_dims)
        self.n_features = n_features or int(query_dims * math.log(query_dims))
        rows = self.n_features // 2 if cosh else self.n_features
        if rows <= 0:
            raise ValueError('n_features must provide at least one projection row')
        self.ortho_scaling = ortho_scaling
        # The historical orthogonal option remains unused; sampling is unchanged.
        self.orthogonal = orthogonal
        self.softmax_temp = 1 / math.sqrt(query_dims) if softmax_temp is None else softmax_temp
        self.cosh = cosh
        self.deterministic_eval = deterministic_eval
        self.eps = eps
        # Constructor/load do not sample. The correctly shaped placeholder and
        # lifecycle metadata follow module.to(); integer/bool metadata keeps dtype.
        self.register_buffer('projection_matrix', torch.zeros(rows, self.query_dims, **factory_kwargs))
        self.register_buffer('_projection_initialized', torch.tensor(False, device=device))
        self.register_buffer('_feature_map_calls', torch.tensor(-1, dtype=torch.int64, device=device))
        self.register_buffer('_redraw_period', torch.tensor(period, dtype=torch.int64, device=device))
        self.factory_kwargs = factory_kwargs  # Retained configuration; never the redraw dtype source.

    @staticmethod
    def _validate_redraw(redraw: int | float) -> int:
        if isinstance(redraw, bool) or not isinstance(redraw, (int, float)):
            raise ValueError('redraw must be a positive integer or positive infinity')
        if redraw == math.inf:
            return 0  # Persistent integer sentinel for draw-once/freeze.
        if isinstance(redraw, float) and (not math.isfinite(redraw) or not redraw.is_integer()):
            raise ValueError('redraw must be a positive integer or positive infinity')
        period = int(redraw)
        if period <= 0 or period > torch.iinfo(torch.int64).max:
            raise ValueError('redraw must fit a positive int64 or be positive infinity')
        return period

    @property
    def redraw(self) -> int | float:
        period = int(self._redraw_period.item())
        return period if period != 0 else math.inf

    @redraw.setter
    def redraw(self, value: int | float) -> None:
        self._redraw_period.fill_(self._validate_redraw(value))

    @property
    def _calls(self) -> int:
        return int(self._feature_map_calls.item())

    @property
    def _initialized(self) -> bool:
        return bool(self._projection_initialized.item())

    def new_feature_map(self, device: torch.device | str | None) -> None:
        """Initialize once, including cold eval; explicit warm requests set cadence.

        First draw has counter0. Subsequent eligible requests increment it and
        draw on multiples of redraw. Warm deterministic eval neither draws nor
        advances it; query/key encoding itself does not advance the cadence.
        TRACE_TEST_ID: SCATTERBRAIN-FEATURE-STATE-CPU.
        """
        if self._initialized and self.deterministic_eval and not self.training:
            return
        calls = self._calls + 1
        if calls > torch.iinfo(torch.int64).max:
            raise OverflowError('feature-map request counter exceeds int64')
        period = int(self._redraw_period.item())
        draw = not self._initialized or (period != 0 and calls % period == 0)
        if draw:
            rows = self.n_features // 2 if self.cosh else self.n_features
            projection_matrix = gaussian_orthogonal_random_matrix(
                nrows=rows, ncols=self.query_dims, scaling=self.ortho_scaling,
                device=self.projection_matrix.device if device is None else device,
                dtype=self.projection_matrix.dtype)
            self.register_buffer('projection_matrix', projection_matrix)
            self._projection_initialized.fill_(True)
        self._feature_map_calls.fill_(calls)

    def _load_from_state_dict(self, state_dict: Mapping[str, torch.Tensor], prefix: str,
                              local_metadata: dict[str, object], strict: bool,
                              missing_keys: list[str], unexpected_keys: list[str],
                              error_msgs: list[str]) -> None:
        # Copy the loader's local mapping: compatibility injection must not edit
        # a caller's checkpoint. Torch still owns dtype/device copy/assign policy.
        local = dict(state_dict)
        projection_key = prefix + 'projection_matrix'
        metadata_keys = [prefix + name for name in
                         ('_projection_initialized', '_feature_map_calls', '_redraw_period')]
        present = [key in local for key in metadata_keys]
        matrix = local.get(projection_key)
        if matrix is None:
            error_msgs.append(prefix + 'missing projection_matrix')
            return
        expected = tuple(self.projection_matrix.shape)
        finite = matrix.device.type != 'meta' and matrix.is_floating_point() and bool(torch.isfinite(matrix).all())
        if not finite:
            error_msgs.append(prefix + 'projection_matrix must be finite floating-point storage')
            return
        if not any(present):
            # Old archives saved only projection_matrix. All-zero legacy states
            # are explicitly treated as cold, even when shape is ambiguous;
            # intentional warm-zero states need new-format metadata.
            cold = bool((matrix == 0).all())
            old_placeholder = tuple(matrix.shape) == (self.query_dims, self.n_features)
            if tuple(matrix.shape) != expected and not (cold and old_placeholder):
                error_msgs.append(prefix + 'invalid legacy projection_matrix shape')
                return
            if cold:
                local[projection_key] = torch.zeros_like(self.projection_matrix)
            local[metadata_keys[0]] = self._projection_initialized.new_tensor(not cold)
            local[metadata_keys[1]] = self._feature_map_calls.new_tensor(-1 if cold else 0)
            local[metadata_keys[2]] = self._redraw_period.detach().clone()
        else:
            if not all(present) or tuple(matrix.shape) != expected:
                error_msgs.append(prefix + 'incomplete lifecycle metadata or invalid projection shape')
                return
            flag, count, period = [local[key] for key in metadata_keys]
            if any(value.device.type == 'meta' or value.ndim != 0 for value in (flag, count, period)) or (
                    flag.dtype != torch.bool or count.dtype != torch.int64 or period.dtype != torch.int64):
                error_msgs.append(prefix + 'invalid lifecycle metadata dtype/shape')
                return
            initialized, calls, cadence = bool(flag.item()), int(count.item()), int(period.item())
            if cadence < 0 or (initialized and calls < 0) or (not initialized and (calls != -1 or bool((matrix != 0).any()))):
                error_msgs.append(prefix + 'inconsistent feature-map lifecycle metadata')
                return
        super()._load_from_state_dict(local, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward_queries(self, x: torch.Tensor, return_log: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._initialized:
            self.new_feature_map(x.device)
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=True,
                              softmax_temp=self.softmax_temp, eps=self.eps, cosh=self.cosh,
                              return_log=return_log)

    def forward_keys(self, x: torch.Tensor, return_log: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._initialized:
            self.new_feature_map(x.device)
        return softmax_kernel(x, projection_matrix=self.projection_matrix, is_query=False,
                              softmax_temp=self.softmax_temp, eps=self.eps, cosh=self.cosh,
                              return_log=return_log)
