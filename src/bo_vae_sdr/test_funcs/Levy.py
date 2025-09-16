"""
Levy function
"""

import numpy as np
import torch

from .base import BaseTestFunction


class Levy(BaseTestFunction):
    def __init__(self, dim, bounds: None | torch.Tensor = None) -> None:
        """
        Levy Function
        """
        super().__init__(dim, bounds)
        assert dim > 0 and isinstance(
            dim, int), "Dimension must be a positive integer."
        self.dim = dim

        self.optimal_input = torch.tensor(
            [1.0]*self.dim
        )
        self.optimal_value = 0

        if bounds is None:
            self.bounds = torch.tensor([[-5.0] * self.dim, [5.0] * self.dim]).T

        self.name = f"Levy_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Levy and Montalvo 2 function. Global minimum is f=0 at x = np.ones((n,))
        :param x: Input vector, length n
        :param n: Dimension
        :return: Float.
        """
        xs = x.cpu().detach().numpy()
        f = []
        for x_ in xs:
            f_ = np.sin(3.0 * np.pi * x_[0]) ** 2
            f_ += np.sum((x_[:-1] - 1.0) ** 2 *
                         (1.0 + np.sin(3.0 * np.pi * x_[1:]) ** 2))
            f_ += (x_[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x_[-1]))
            f_ *= 0.1
            f.append(f_)

        return torch.from_numpy(-np.array(f)).to(x).unsqueeze(-1)
