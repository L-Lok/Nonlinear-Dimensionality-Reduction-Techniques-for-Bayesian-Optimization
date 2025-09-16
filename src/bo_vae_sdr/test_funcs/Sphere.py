"""
Standard function of sum of squares implementation in PyTorch.
"""
import torch

from .base import BaseTestFunction


class Sphere(BaseTestFunction):
    def __init__(self, dim, bounds: None | torch.Tensor = None) -> None:
        """
        Sphere function implementation in PyTorch.
        Just sum of squares and return negative value for maximization.

        Parameters:
            dim (int): Dimension of the test function.
            bounds (tensor): Bounds of the test function. in dim x 2 shape.
        """
        super().__init__(dim, bounds)
        assert dim > 0 and isinstance(
            dim, int), "Dimension must be a positive integer."
        self.dim = dim

        self.optimal_input = torch.tensor([[0.0] * self.dim])
        self.optimal_value = 0.0

        if bounds is None:
            self.bounds = torch.tensor([[-30.] * self.dim, [30.] * self.dim]).T

        self.name = f"Sphere_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sphere function implementation in PyTorch.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the Sphere squares function.
        """
        assert x.shape[1] == self.dim, "Input tensor has incorrect dimension."

        f = torch.sum(x ** 2, dim=1).unsqueeze(1)
        return -f
