from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import NamedTuple, Protocol, cast


class Stream(NamedTuple):
    ptr: int


class _Device(Protocol):
    compute_capability: str | int


class _DeviceFactory(Protocol):
    def __call__(self) -> _Device: ...


class _KernelFunction(Protocol):
    def __call__(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[int],
        *,
        stream: Stream,
    ) -> None: ...


class _CudaModule(Protocol):
    def load(self, ptx: bytes) -> None: ...
    def get_function(self, name: str) -> _KernelFunction: ...


class _CudaModuleFactory(Protocol):
    def __call__(self) -> _CudaModule: ...


class _NvrtcProgram(Protocol):
    def compile(self, options: list[str]) -> str: ...


class _NvrtcProgramFactory(Protocol):
    def __call__(self, source: str, title: str) -> _NvrtcProgram: ...


def _load_cuda_factories() -> tuple[
    _DeviceFactory, _CudaModuleFactory, _NvrtcProgramFactory
]:
    """Resolve optional CUDA compiler/runtime dependencies on demand."""
    device_module = importlib.import_module("cupy.cuda.device")
    function_module = importlib.import_module("cupy.cuda.function")
    compiler_module = importlib.import_module("pynvrtc.compiler")
    return (
        cast(_DeviceFactory, getattr(device_module, "Device")),
        cast(_CudaModuleFactory, getattr(function_module, "Module")),
        cast(_NvrtcProgramFactory, getattr(compiler_module, "Program")),
    )


class cupyKernel:
    """Lazy CUDA kernel compiler/launcher used by SimHash.

    grid/block are 3-tuples, args is the raw integer device-pointer/scalar
    argument list, and strm is a CUDA stream pointer. Optional CuPy/PyNVRTC
    dependencies are imported only when compilation is requested.
    """

    def __init__(self, kernel: str, func_name: str) -> None:
        self.kernel = kernel
        self.title = func_name + ".cu"
        self.func_name = func_name
        self.compiled = False
        self.func: _KernelFunction | None = None

    @staticmethod
    def get_compute_arch() -> str:
        device_factory, _module_factory, _program_factory = (
            _load_cuda_factories()
        )
        capability = device_factory().compute_capability
        return f"compute_{capability}"

    def compile(self) -> None:
        device_factory, module_factory, program_factory = (
            _load_cuda_factories()
        )
        del device_factory
        program = program_factory(self.kernel, self.title)
        arch = f"-arch={self.get_compute_arch()}"
        ptx = program.compile([arch])
        module = module_factory()
        module.load(ptx.encode())
        self.func = module.get_function(self.func_name)
        self.compiled = True

    def __call__(
        self,
        *,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list[int],
        strm: int,
    ) -> None:
        if not self.compiled:
            self.compile()
        if self.func is None:
            raise RuntimeError("CUDA kernel did not compile")
        self.func(
            grid,
            block,
            args,
            stream=Stream(ptr=strm),
        )
