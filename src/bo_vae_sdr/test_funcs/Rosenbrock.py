"""
Rosenbrock function
"""
import torch

from .base import BaseTestFunction


class Rosenbrock(BaseTestFunction):
    def __init__(self, dim, bounds: None | torch.Tensor = None, a=1, b=100) -> None:
        """
        Rosenbrock function implementation in PyTorch.

        Parameters:
            dim (int): Dimension of the test function.
            bounds (tensor): Bounds of the test function. in dim x 2 shape.
            a (float): Parameter a, default is 1.
            b (float): Parameter b, default is 100.
        """
        super().__init__(dim, bounds)
        assert dim > 0 and isinstance(
            dim, int), "Dimension must be a positive integer."
        self.dim = dim
        self.a = a
        self.b = b

        self.optimal_input = torch.tensor([[1.0] * self.dim])
        self.optimal_value = 0.0

        if bounds is None:
            self.bounds = torch.tensor([[-30.] * self.dim, [30.] * self.dim]).T


        self.name = f"Rosenbrock_{self.dim}d_{self.bounds_suffix()}"


    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rosenbrock function implementation in PyTorch.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the Rosenbrock function.
        """
        assert x.shape[1] == self.dim, "Input tensor has incorrect dimension."

        f = torch.sum(
            (self.a - x[:, :-1]) ** 2
            + self.b * (x[:, 1:] - x[:, :-1] ** 2) ** 2, dim=1, keepdim=True)
        return -f
