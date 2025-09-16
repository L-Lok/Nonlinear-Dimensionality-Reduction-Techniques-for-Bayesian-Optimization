"""
Ackley function
"""
import torch

from .base import BaseTestFunction


class Ackley(BaseTestFunction):
    """
    Ackley function.
    """
    def __init__(self, dim, bounds: None | torch.Tensor = None, a=20, b=0.2, c=2 * torch.pi) -> None:
        """
        Ackley function.

        Parameters:
            dim (int): Dimension of the test function.
            a (float): Parameter a, default is 20.
            b (float): Parameter b, default is 0.2.
            c (float): Parameter c, default is 2*pi.
            bounds (tensor): Bounds of the test function. in dim x 2 shape.
        """
        super().__init__(dim, bounds)
        self.a = a
        self.b = b
        self.c = c

        self.optimal_input = torch.tensor([[0.0] * self.dim])
        self.optimal_value = 0.0

        if self.bounds is None:
            self.bounds = torch.tensor([[-30.] * self.dim, [30.] * self.dim]).T

        self.name = f"Ackley_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ackley function implementation in PyTorch.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the Ackley function.
        """
        assert x.shape[1] == self.dim, "Input tensor has incorrect dimension."

        sum1 = torch.sum(x**2, dim=1)
        sum2 = torch.sum(torch.cos(self.c * x), dim=1)
        term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum1 / self.dim))
        term2 = -torch.exp(sum2 / self.dim)
        result = term1 + term2 + self.a + torch.exp(torch.tensor(1.0))
        return -result.unsqueeze(1)


