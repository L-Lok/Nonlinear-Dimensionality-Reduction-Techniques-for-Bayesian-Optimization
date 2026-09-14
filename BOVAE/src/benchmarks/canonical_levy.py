"""Canonical Levy benchmark from the SFU test-function collection."""

from __future__ import annotations

import torch

from .base import BaseTestFunction, check_dimension


class CanonicalLevy(BaseTestFunction):
    def __init__(self, dim: int, bounds: torch.Tensor | None = None) -> None:
        super().__init__(dim, bounds)
        self.optimal_input = torch.ones((1, dim), dtype=torch.float64)
        self.optimal_value = 0.0
        if self.bounds is None:
            self.bounds = torch.tensor([[-10.0, 10.0]] * dim, dtype=torch.float64)
        self.name = f"CanonicalLevy_{dim}d_{self.bounds_suffix()}"

    @check_dimension
    def original_objective(self, x: torch.Tensor) -> torch.Tensor:
        w = 1.0 + (x - 1.0) / 4.0
        first = torch.sin(torch.pi * w[:, 0]).square()
        last = (w[:, -1] - 1.0).square() * (
            1.0 + torch.sin(2.0 * torch.pi * w[:, -1]).square()
        )
        middle = (
            torch.zeros_like(first)
            if self.dim == 1
            else torch.sum(
                (w[:, :-1] - 1.0).square()
                * (1.0 + 10.0 * torch.sin(torch.pi * w[:, :-1] + 1.0).square()),
                dim=1,
            )
        )
        return (first + middle + last).unsqueeze(1)

    def func(self, x: torch.Tensor) -> torch.Tensor:
        return -self.original_objective(x)
