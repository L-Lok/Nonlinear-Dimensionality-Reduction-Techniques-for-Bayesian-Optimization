"""
Styblinski-Tang function
"""
import torch

from .base import BaseTestFunction


class StyblinskiTang(BaseTestFunction):
    """
    Styblinski-Tang function.
    """

    def __init__(self, dim, bounds: None | torch.Tensor = None) -> None:
        """
        .
        """
        super().__init__(dim, bounds)

        self.optimal_input = torch.tensor([[-2.903534] * self.dim])
        self.optimal_value = -39.16599*self.dim

        if self.bounds is None:
            self.bounds = torch.tensor([[-5.] * self.dim, [5.] * self.dim]).T

        self.name = f"Styblinski-Tang_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Styblinski-Tang function implementation in PyTorch.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the function.
        """
        assert x.shape[1] == self.dim, "Input tensor has incorrect dimension."

        sum1 = torch.sum(x**4, dim=1)
        sum2 = torch.sum(16*x**2, dim=1)
        sum3 = torch.sum(5*x, dim=1)
        result = (sum1 - sum2 + sum3)/2
        return -result.unsqueeze(1)
