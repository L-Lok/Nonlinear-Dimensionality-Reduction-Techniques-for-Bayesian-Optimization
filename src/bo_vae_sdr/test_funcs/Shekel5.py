"""
Shekel5 function
"""
import numpy as np
import torch

from .base import BaseTestFunction


class Shekel5(BaseTestFunction):
    def __init__(self, dim, bounds: None | torch.Tensor = None) -> None:
        """
        Shekel5 Function
        """
        super().__init__(dim, bounds)
        assert dim > 0 and isinstance(
            dim, int), "Dimension must be a positive integer."
        self.dim = dim
        assert self.dim == 4, 'Dimension must be 4'

        self.optimal_input = torch.tensor([[4.0] * self.dim])
        self.optimal_value = -10.1499

        if bounds is None:
            self.bounds = torch.tensor([[-10.] * self.dim, [10.] * self.dim]).T

        self.name = f"Shekel_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shekel5 function implementation in PyTorch.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the Shekel5 function.
        """
        assert x.shape[1] == self.dim, "Input tensor has incorrect dimension."

        x = x.cpu().detach().numpy()
        a1 = np.array([4.0, 4.0, 4.0, 4.0])
        a2 = np.array([1.0, 1.0, 1.0, 1.0])
        a3 = np.array([8.0, 8.0, 8.0, 8.0])
        a4 = np.array([6.0, 6.0, 6.0, 6.0])
        a5 = np.array([3.0, 7.0, 3.0, 7.0])
        a = [a1, a2, a3, a4, a5]
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4])
        f = []
        for x_ in x:
            f_ = -np.sum(
                np.array([1.0 / (np.sum((x_ - a[i]) ** 2) + c[i])
                         for i in range(5)])
            )
            f.append(f_)

        return torch.from_numpy(-np.array(f)).to(x).unsqueeze(-1)
