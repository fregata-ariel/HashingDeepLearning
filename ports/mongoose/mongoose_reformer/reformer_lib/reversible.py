from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, cast

import torch
import torch.nn as nn
from torch.autograd.function import Function, FunctionCtx
from torch.utils.checkpoint import get_device_states, set_device_states

RouteKwargs: TypeAlias = dict[str, object]
BlockKwargs: TypeAlias = dict[str, RouteKwargs]


class Deterministic(nn.Module):
    """Replay an nn.Module with the RNG state captured during forward."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.cpu_state: torch.Tensor | None = None
        self.cuda_in_fwd = False
        self.gpu_devices: list[int] = []
        self.gpu_states: list[torch.Tensor] = []

    def record_rng(self, *args: torch.Tensor) -> None:
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(
        self,
        *args: torch.Tensor,
        record_rng: bool = False,
        set_rng: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return cast(torch.Tensor, self.net(*args, **kwargs))

        if self.cpu_state is None:
            raise RuntimeError("RNG replay requested before RNG state was recorded")

        with torch.random.fork_rng(
            devices=self.gpu_devices if self.cuda_in_fwd else [],
            enabled=True,
        ):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return cast(torch.Tensor, self.net(*args, **kwargs))


class ReversibleBlock(nn.Module):
    """One additive reversible pair f/g."""

    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        depth: int | None = None,
        send_signal: bool = False,
    ) -> None:
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.depth = depth
        self.send_signal = send_signal

    def forward(
        self,
        x: torch.Tensor,
        f_args: RouteKwargs | None = None,
        g_args: RouteKwargs | None = None,
    ) -> torch.Tensor:
        f_kwargs = {} if f_args is None else dict(f_args)
        g_kwargs = {} if g_args is None else dict(g_args)
        x1, x2 = torch.chunk(x, 2, dim=2)

        if self.send_signal:
            f_kwargs["_reverse"] = g_kwargs["_reverse"] = False
            f_kwargs["_depth"] = g_kwargs["_depth"] = self.depth

        with torch.no_grad():
            y1 = x1 + self.f(
                x2, record_rng=self.training, **f_kwargs
            )
            y2 = x2 + self.g(
                y1, record_rng=self.training, **g_kwargs
            )
        return torch.cat([y1, y2], dim=2)

    def backward_pass(
        self,
        y: torch.Tensor,
        dy: torch.Tensor,
        f_args: RouteKwargs | None = None,
        g_args: RouteKwargs | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_kwargs = {} if f_args is None else dict(f_args)
        g_kwargs = {} if g_args is None else dict(g_args)
        y1, y2 = torch.chunk(y, 2, dim=2)
        dy1, dy2 = torch.chunk(dy, 2, dim=2)

        if self.send_signal:
            f_kwargs["_reverse"] = g_kwargs["_reverse"] = True
            f_kwargs["_depth"] = g_kwargs["_depth"] = self.depth

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_kwargs)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            y1_grad = y1.grad
            if y1_grad is None:
                raise RuntimeError("missing reversible gradient for y1")
            dx1 = dy1 + y1_grad
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_kwargs)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            x2_grad = x2.grad
            if x2_grad is None:
                raise RuntimeError("missing reversible gradient for x2")
            dx2 = dy2 + x2_grad
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, dx


class IrreversibleBlock(nn.Module):
    def __init__(self, f: nn.Module, g: nn.Module) -> None:
        super().__init__()
        self.f = f
        self.g = g

    def forward(
        self,
        x: torch.Tensor,
        f_args: RouteKwargs,
        g_args: RouteKwargs,
    ) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1 = x1 + cast(torch.Tensor, self.f(x2, **f_args))
        y2 = x2 + cast(torch.Tensor, self.g(y1, **g_args))
        return torch.cat([y1, y2], dim=2)


class _ReversibleFunction(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        blocks: Sequence[ReversibleBlock],
        kwargs: BlockKwargs,
    ) -> torch.Tensor:
        ctx.kwargs = kwargs
        ctx.blocks = list(blocks)
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        return x

    @staticmethod
    def backward(
        ctx: FunctionCtx, dy: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        y = cast(torch.Tensor, ctx.y)
        kwargs = cast(BlockKwargs, ctx.kwargs)
        blocks = cast(list[ReversibleBlock], ctx.blocks)
        for block in blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    """Route kwargs through reversible or ordinary residual blocks."""

    def __init__(
        self,
        blocks: Sequence[Sequence[nn.Module]],
        layer_dropout: float = 0.0,
        reverse_thres: int = 0,
        send_signal: bool = False,
    ) -> None:
        super().__init__()
        self.layer_dropout = layer_dropout
        self.reverse_thres = reverse_thres
        pairs = [(pair[0], pair[1]) for pair in blocks]
        self.blocks = nn.ModuleList(
            [
                ReversibleBlock(f, g, depth, send_signal)
                for depth, (f, g) in enumerate(pairs)
            ]
        )
        self.irrev_blocks = nn.ModuleList(
            [IrreversibleBlock(f=f, g=g) for f, g in pairs]
        )

    def forward(
        self,
        x: torch.Tensor,
        arg_route: tuple[bool, bool] = (True, True),
        **kwargs: object,
    ) -> torch.Tensor:
        reverse = x.shape[1] > self.reverse_thres
        selected: list[nn.Module] = list(
            self.blocks if reverse else self.irrev_blocks
        )

        if self.training and self.layer_dropout > 0:
            to_drop = (
                torch.empty(len(self.blocks)).uniform_(0, 1)
                < self.layer_dropout
            )
            selected = [
                block
                for block, drop in zip(self.blocks, to_drop)
                if not bool(drop)
            ]
            if not selected:
                selected = [self.blocks[0]]

        f_args: RouteKwargs = dict(kwargs) if arg_route[0] else {}
        g_args: RouteKwargs = dict(kwargs) if arg_route[1] else {}
        block_kwargs: BlockKwargs = {
            "f_args": f_args,
            "g_args": g_args,
        }

        if not reverse:
            for block in selected:
                x = cast(torch.Tensor, block(x, **block_kwargs))
            return x

        reversible_blocks = [
            cast(ReversibleBlock, block) for block in selected
        ]
        return cast(
            torch.Tensor,
            _ReversibleFunction.apply(
                x, reversible_blocks, block_kwargs
            ),
        )
