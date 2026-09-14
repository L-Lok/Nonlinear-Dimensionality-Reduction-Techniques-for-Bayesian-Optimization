"""Shared contracts for manuscript benchmark objectives."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

import numpy as np
import torch


def check_dimension(func: Callable) -> Callable:
    """Validate the conventional ``(n, dimension)`` tensor shape."""

    @wraps(func)
    def wrapper(self: "BaseTestFunction", x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(
                f"input must have shape (n, {self.dim}), got {tuple(x.shape)}"
            )
        return func(self, x)

    return wrapper


class BaseTestFunction:
    """Base class for BO objectives.

    Subclasses return an internal target from :meth:`func`. The pipeline
    maximizes that target, so manuscript minimization objectives return
    ``-original_objective``.
    """

    def __init__(self, dim: int, bounds: torch.Tensor | None = None) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.dim = dim
        self.optimal_input: torch.Tensor | None = None
        self.optimal_value: float | None = None
        if bounds is not None:
            if bounds.ndim != 2 or bounds.shape != (dim, 2):
                raise ValueError(f"bounds must have shape ({dim}, 2)")
        self.bounds = bounds
        self.name = type(self).__name__

    def bounds_suffix(self) -> str:
        if self.bounds is None:
            return "bounds_unspecified"
        if torch.all(self.bounds == self.bounds[0]):
            lower, upper = self.bounds[0].detach().cpu().tolist()
            return f"bounds_{lower}_{upper}"
        values = "".join(
            f"_{lower}_{upper}"
            for lower, upper in self.bounds.detach().cpu().tolist()
        )
        return f"bounds_{values}"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)

    @check_dimension
    def func(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def sample_uniform_design(dim: int, size: int, seed: int) -> np.ndarray:
    """Sample a deterministic design from the manuscript ambient box."""

    if dim <= 0 or size <= 0:
        raise ValueError("dim and size must be positive")
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(size, dim))
