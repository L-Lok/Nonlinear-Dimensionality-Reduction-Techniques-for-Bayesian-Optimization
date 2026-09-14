"""Canonical full-rank Styblinski--Tang benchmark."""

from __future__ import annotations

import torch

from .base import BaseTestFunction, check_dimension


class CanonicalStyblinskiTang(BaseTestFunction):
    OPTIMAL_COORDINATE = -2.9035340277711783
    OPTIMAL_VALUE_PER_DIMENSION = -39.16616570377141

    def __init__(self, dim: int, bounds: torch.Tensor | None = None) -> None:
        super().__init__(dim, bounds)
        self.optimal_input = torch.full(
            (1, dim), self.OPTIMAL_COORDINATE, dtype=torch.float64
        )
        self.optimal_value = self.OPTIMAL_VALUE_PER_DIMENSION * dim
        if self.bounds is None:
            self.bounds = torch.tensor([[-5.0, 5.0]] * dim, dtype=torch.float64)
        self.name = f"CanonicalStyblinskiTang_{dim}d_{self.bounds_suffix()}"

    @check_dimension
    def original_objective(self, x: torch.Tensor) -> torch.Tensor:
        return (0.5 * (x.pow(4) - 16.0 * x.square() + 5.0 * x)).sum(
            dim=1, keepdim=True
        )

    def func(self, x: torch.Tensor) -> torch.Tensor:
        return -self.original_objective(x)
