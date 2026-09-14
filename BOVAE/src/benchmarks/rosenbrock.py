"""Rosenbrock minimization benchmark with the BO internal-target convention."""

from __future__ import annotations

import torch

from .base import BaseTestFunction, check_dimension


class Rosenbrock(BaseTestFunction):
    def __init__(
        self,
        dim: int,
        bounds: torch.Tensor | None = None,
        a: float = 1.0,
        b: float = 100.0,
    ) -> None:
        super().__init__(dim, bounds)
        self.a, self.b = a, b
        self.optimal_input = torch.ones((1, dim))
        self.optimal_value = 0.0
        if self.bounds is None:
            self.bounds = torch.tensor([[-30.0, 30.0]] * dim)
        self.name = f"Rosenbrock_{dim}d_{self.bounds_suffix()}"

    @check_dimension
    def func(self, x: torch.Tensor) -> torch.Tensor:
        original = torch.sum(
            (self.a - x[:, :-1]).square()
            + self.b * (x[:, 1:] - x[:, :-1].square()).square(),
            dim=1,
            keepdim=True,
        )
        return -original
