from collections.abc import Callable
from torch import Tensor

# Type contract of the retained optional decorator, not an Apex runtime model.
def float_function(function: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Callable[[Tensor, Tensor, Tensor], Tensor]: ...
