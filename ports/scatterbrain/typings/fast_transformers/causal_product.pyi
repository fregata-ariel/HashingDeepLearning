from torch import Tensor

# Retained lazy import only; CPU causal execution is outside the tested port.
def causal_dot_product(queries: Tensor, keys: Tensor, values: Tensor) -> Tensor: ...
