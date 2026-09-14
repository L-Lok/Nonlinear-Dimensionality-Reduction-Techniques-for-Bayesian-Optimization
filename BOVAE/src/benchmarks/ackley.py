"""Ackley minimization benchmark with the BO internal-target convention."""

from __future__ import annotations

import torch

from .base import BaseTestFunction, check_dimension


class Ackley(BaseTestFunction):
    def __init__(
        self,
        dim: int,
        bounds: torch.Tensor | None = None,
        a: float = 20.0,
        b: float = 0.2,
        c: float = 2.0 * torch.pi,
    ) -> None:
        super().__init__(dim, bounds)
        self.a, self.b, self.c = a, b, c
        self.optimal_input = torch.zeros((1, dim))
        self.optimal_value = 0.0
        if self.bounds is None:
            self.bounds = torch.tensor([[-30.0, 30.0]] * dim)
        self.name = f"Ackley_{dim}d_{self.bounds_suffix()}"

    @check_dimension
    def func(self, x: torch.Tensor) -> torch.Tensor:
        sum_squares = torch.sum(x.square(), dim=1)
        sum_cosines = torch.sum(torch.cos(self.c * x), dim=1)
        original = (
            -self.a * torch.exp(-self.b * torch.sqrt(sum_squares / self.dim))
            - torch.exp(sum_cosines / self.dim)
            + self.a
            + torch.exp(torch.ones((), device=x.device, dtype=x.dtype))
        )
        return -original.unsqueeze(1)
