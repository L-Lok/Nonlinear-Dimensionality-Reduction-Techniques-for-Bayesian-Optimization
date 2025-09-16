"""
Price's Transistor Monitoring function
"""

import numpy as np
import torch

from .base import BaseTestFunction


class PriceTM(BaseTestFunction):
    def __init__(self, dim, bounds: None | torch.Tensor = None) -> None:
        """
        PriceTM Function
        """
        super().__init__(dim, bounds)
        assert dim > 0 and isinstance(
            dim, int), "Dimension must be a positive integer."
        self.dim = dim
        assert self.dim == 9, "Dimension must be 9"

        self.optimal_input = torch.tensor(
            [[0.9, 0.45, 1.0, 2.0, 8.0, 8.0, 5.0, 1.0, 2.0]]
        )
        self.optimal_value = 0

        if bounds is None:
            self.bounds = torch.tensor(
                [[-10.0] * self.dim, [10.0] * self.dim]).T

        self.name = f"Hartmann6_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Price's Transistor Monitoring function. Global minimum is f=0 at x = np.array([0.9,0.45,1.0,2.0,8.0,8.0,5.0,1.0,2.0])
        :param x: Input vector, length 9
        :return: Float.
        """
        xs = x.cpu().detach().numpy()
        g1 = np.array([0.485, 0.369, 5.2095, 23.3037, 28.5132])
        g2 = np.array([0.752, 1.254, 10.0677, 101.779, 111.8467])
        g3 = np.array([0.869, 0.703, 22.9274, 111.461, 134.3884])
        g4 = np.array([0.982, 1.455, 20.2153, 191.267, 211.4823])
        g = [g1, g2, g3, g4]
        f = []
        for x in xs:
            alpha = np.array(
                [
                    (1.0 - x[0] * x[1])
                    * x[2]
                    * (
                        np.exp(
                            x[4] * (gk[0] - 1e-3 * gk[2] *
                                    x[6] - 1e-3 * gk[4] * x[7])
                        )
                        - 1.0
                    )
                    - gk[4]
                    + gk[3] * x[1]
                    for gk in g
                ]
            )
            beta = np.array(
                [
                    (1.0 - x[0] * x[1])
                    * x[3]
                    * (
                        np.exp(
                            x[5]
                            * (
                                gk[0]
                                - gk[1]
                                - 1e-3 * gk[2] * x[6]
                                + 1e-3 * gk[3] * x[8]
                            )
                        )
                        - 1.0
                    )
                    - gk[4] * x[0]
                    + gk[3]
                    for gk in g
                ]
            )
            gamma = x[0] * x[2] - x[1] * x[3]
            f_ = gamma**2 + np.sum(alpha**2 + beta**2)
            f.append(f_)

        return torch.from_numpy(-np.array(f)).to(x).unsqueeze(-1)
