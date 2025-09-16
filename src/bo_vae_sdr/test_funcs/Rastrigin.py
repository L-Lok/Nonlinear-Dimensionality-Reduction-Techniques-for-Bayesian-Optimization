"""
Rastrigin function
"""
import torch

from .base import BaseTestFunction


class Rastrigin(BaseTestFunction):
    """
    Rastrigin function.
    """

    def __init__(self, dim, bounds: None | torch.Tensor = None) -> None:
        """
        .
        """
        super().__init__(dim, bounds)

        self.optimal_input = torch.tensor([[0.] * self.dim])
        self.optimal_value = 0.

        if self.bounds is None:
            self.bounds = torch.tensor(
                [[-5.12] * self.dim, [5.12] * self.dim]).T

        self.name = f"Rastrigin_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rastrigin function implementation in PyTorch.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the function.
        """
        assert x.shape[1] == self.dim, "Input tensor has incorrect dimension."

        sum1 = torch.sum(x**2, dim=1)
        sum2 = torch.sum(10*torch.cos(2*torch.pi*x), dim=1)
        sum3 = 10.*self.dim
        result = (sum1 - sum2 + sum3)
        return -result.unsqueeze(1)  # for minimisation
