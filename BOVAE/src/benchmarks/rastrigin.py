"""Rastrigin minimization benchmark with the BO internal-target convention."""

from __future__ import annotations

import torch

from .base import BaseTestFunction, check_dimension


class Rastrigin(BaseTestFunction):
    def __init__(self, dim: int, bounds: torch.Tensor | None = None) -> None:
        super().__init__(dim, bounds)
        self.optimal_input = torch.zeros((1, dim))
        self.optimal_value = 0.0
        if self.bounds is None:
            self.bounds = torch.tensor([[-5.12, 5.12]] * dim)
        self.name = f"Rastrigin_{dim}d_{self.bounds_suffix()}"

    @check_dimension
    def func(self, x: torch.Tensor) -> torch.Tensor:
        original = (
            torch.sum(x.square(), dim=1)
            - torch.sum(10.0 * torch.cos(2.0 * torch.pi * x), dim=1)
            + 10.0 * self.dim
        )
        return -original.unsqueeze(1)
