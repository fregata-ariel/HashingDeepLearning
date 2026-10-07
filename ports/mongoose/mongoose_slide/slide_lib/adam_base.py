from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import TypedDict, cast

import torch
from torch.optim import Optimizer


class AdamGroup(TypedDict):
    params: list[torch.Tensor]
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    amsgrad: bool


class AdamState(TypedDict, total=False):
    step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    max_exp_avg_sq: torch.Tensor


class Adam(Optimizer):
    """Adam variant supporting both dense and sparse gradients.

    Optimizer state is explicit: one integer step and dense first/second
    moment tensors per parameter, plus max_exp_avg_sq when AMSGrad is enabled.
    Sparse updates only touch indices present in the coalesced sparse gradient.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 0: {betas[0]}"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 1: {betas[1]}"
            )
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, object]) -> None:
        super().__setstate__(state)
        groups = cast(list[AdamGroup], self.param_groups)
        for group in groups:
            group.setdefault("amsgrad", False)

    @staticmethod
    def _state_for(
        optimizer: Adam,
        parameter: torch.Tensor,
        amsgrad: bool,
    ) -> AdamState:
        state = cast(AdamState, optimizer.state[parameter])
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(parameter.data)
            state["exp_avg_sq"] = torch.zeros_like(parameter.data)
            if amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(
                    parameter.data
                )
        return state

    def dense(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        group: AdamGroup,
    ) -> None:
        amsgrad = group["amsgrad"]
        state = self._state_for(self, parameter, amsgrad)
        state["step"] += 1

        if group["weight_decay"] != 0:
            grad = grad.add(
                parameter.data, alpha=group["weight_decay"]
            )

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            grad, grad, value=1 - beta2
        )

        if amsgrad:
            max_exp_avg_sq = state["max_exp_avg_sq"]
            torch.maximum(
                max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq
            )
            denom = max_exp_avg_sq.sqrt().add_(group["eps"])
        else:
            denom = exp_avg_sq.sqrt().add_(group["eps"])

        step = state["step"]
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = (
            group["lr"]
            * math.sqrt(bias_correction2)
            / bias_correction1
        )
        parameter.data.addcdiv_(
            exp_avg, denom, value=-step_size
        )

    def sparse(
        self,
        parameter: torch.Tensor,
        grad: torch.Tensor,
        group: AdamGroup,
    ) -> None:
        if group["weight_decay"] != 0:
            raise RuntimeError(
                "weight_decay is not supported for sparse Adam updates"
            )
        if group["amsgrad"]:
            raise RuntimeError(
                "AMSGrad is not supported for sparse Adam updates"
            )

        state = self._state_for(self, parameter, False)
        state["step"] += 1

        coalesced = grad.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        size = coalesced.size()
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        beta1, beta2 = group["betas"]

        def make_sparse(update_values: torch.Tensor) -> torch.Tensor:
            if indices.dim() == 0 or update_values.dim() == 0:
                return torch.zeros_like(coalesced)
            return torch.sparse_coo_tensor(
                indices,
                update_values,
                size=size,
                dtype=coalesced.dtype,
                device=coalesced.device,
            )

        old_avg = exp_avg.sparse_mask(coalesced).values()
        avg_update = values.sub(old_avg).mul_(1 - beta1)
        exp_avg.add_(make_sparse(avg_update))

        old_avg_sq = exp_avg_sq.sparse_mask(coalesced).values()
        avg_sq_update = (
            values.pow(2).sub_(old_avg_sq).mul_(1 - beta2)
        )
        exp_avg_sq.add_(make_sparse(avg_sq_update))

        numer = avg_update.add_(old_avg)
        avg_sq_update.add_(old_avg_sq)
        denom = avg_sq_update.sqrt_().add_(group["eps"])

        step = state["step"]
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = (
            group["lr"]
            * math.sqrt(bias_correction2)
            / bias_correction1
        )
        parameter.data.add_(
            make_sparse(-step_size * numer.div_(denom))
        )

    def step(
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        """Perform one optimizer step and return an optional closure loss."""
        loss = closure() if closure is not None else None
        groups = cast(list[AdamGroup], self.param_groups)
        for group in groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach()
                if grad.is_sparse:
                    self.sparse(parameter, grad, group)
                else:
                    self.dense(parameter, grad, group)
        return loss
