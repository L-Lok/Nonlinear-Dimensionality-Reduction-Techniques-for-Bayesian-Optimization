"""
BeckerAndLago function

Generalized function in multi dimensions

min f(x) = sum(|x_i| - a)
"""
import torch

from .base import BaseTestFunction
from itertools import product


class BeckerAndLago(BaseTestFunction):
    def __init__(self, dim, a=5, bounds: None | torch.Tensor = None) -> None:
        """
        BeckerAndLago function implementation in PyTorch.

        Generalized function in multi dimensions

        min f(x) = sum(|x_i| - a)

        Parameters:
            dim (int): Dimension of the test function.
            a (float): Parameter a, default is 5.
            bounds (tensor): Bounds of the test function. in dim x 2 shape.
        """
        super().__init__(dim, bounds)
        assert dim > 0 and isinstance(
            dim, int), "Dimension must be a positive integer."
        self.dim = dim
        self.a = a

        # any permutation of a and -a in each dimension
        self.optimal_input = torch.tensor(list(product([a, -a], repeat=dim)))
        self.optimal_value = 0.0

        if bounds is None:
            self.bounds = torch.tensor([[-10.] * self.dim, [10.] * self.dim]).T

        self.name = f"BeckerAndLago_{self.dim}d_{self.bounds_suffix()}"


    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        BeckerAndLago function implementation in PyTorch.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the BeckerAndLago function.
        """
        assert x.shape[1] == self.dim, "Input tensor has incorrect dimension."

        sum_term = torch.sum(torch.abs(x) - self.a, dim=1)
        return -sum_term.unsqueeze(1)
