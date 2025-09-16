"""
Trid function
"""
import torch

from .base import BaseTestFunction


class Trid(BaseTestFunction):
    """
    Trid function.
    https://www.sfu.ca/~ssurjano/trid.html
    """
    def __init__(self, dim, bounds: None | torch.Tensor = None) -> None:
        """
        .
        """
        super().__init__(dim, bounds)

        self.optimal_input = torch.tensor([i*(self.dim + 1 - i) for i in range(self.dim)])
        self.optimal_value = - self.dim*(self.dim + 4)*(self.dim - 1)/6

        if self.bounds is None:
            self.bounds = torch.tensor([[-self.dim**2] * self.dim, [self.dim**2] * self.dim]).T

        self.name = f"Trid_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Trid function implementation in PyTorch.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the Trid function.
        """
        assert x.shape[1] == self.dim, "Input tensor has incorrect dimension."

        # Compute (xx - 1)^2 for each element and sum along the rows
        sum1 = torch.sum((x - 1) ** 2, dim=1, keepdim=True)

        # Compute xi * x(i-1) for each pair and sum along the rows
        sum2 = torch.sum(x[:, 1:] * x[:, :-1], dim=1, keepdim=True)

        # Calculate the result
        f = sum1 - sum2

        return -f
