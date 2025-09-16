"""
Hartmann6 function
"""

import numpy as np
import torch

from .base import BaseTestFunction


class Hartmann6(BaseTestFunction):
    def __init__(self, dim, bounds: None | torch.Tensor = None) -> None:
        """
        Hartmann6 Function
        """
        super().__init__(dim, bounds)
        assert dim > 0 and isinstance(
            dim, int), "Dimension must be a positive integer."
        self.dim = dim
        assert self.dim == 6, "Dimension must be 6"

        self.optimal_input = torch.tensor(
            [[0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301]]
        )
        self.optimal_value = -3.322368

        if bounds is None:
            self.bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim]).T

        self.name = f"Hatmann6_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """

        Hartman 6 function. Global minimum is f=-3.322368 at x = np.array([0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301])
        :param x: Input vector, length 6
        :return: Float.

        """
        xs = x.cpu().detach().numpy()
        c = np.array([1.0, 1.2, 3.0, 3.2])
        a1 = np.array([10.0, 3.0, 17.0, 3.5, 1.7, 8.0])
        a2 = np.array([0.05, 10.0, 17.0, 0.1, 8.0, 14.0])
        a3 = np.array([3.0, 3.5, 1.7, 10.0, 17.0, 8.0])
        a4 = np.array([17.0, 8.0, 0.05, 10.0, 0.1, 14.0])
        A = [a1, a2, a3, a4]
        p1 = np.array([0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886])
        p2 = np.array([0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991])
        p3 = np.array([0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665])
        p4 = np.array([0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381])
        # The rest is the same for both Hartman functions
        p = [p1, p2, p3, p4]
        f = []
        for x in xs:
            inner_terms = np.array(
                [- np.dot(A[i], (x - p[i]) ** 2) for i in range(4)])
            f_ = -np.dot(c, np.exp(inner_terms))
            f.append(f_)

        return torch.from_numpy(-np.array(f)).to(x).unsqueeze(-1)
