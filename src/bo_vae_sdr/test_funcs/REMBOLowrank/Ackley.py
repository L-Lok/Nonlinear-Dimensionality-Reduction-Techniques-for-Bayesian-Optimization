import numpy as np
import torch

from ..base import BaseTestFunction
from .rotation_mat import get_Q_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


Q = get_Q_matrix(r=4, n=100)


def ackley_4d(
    x: torch.Tensor,
    dim=4,
    a=20,
    b=0.2,
    c=2 * torch.pi,
    emb_domain=torch.tensor([[-5.0, 5.0]] * 4, device=device),
) -> torch.Tensor:
    """
    Ackley function implementation in PyTorch.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The result of the Ackley function.
    """
    assert x.shape[1] == 4, "Input tensor has incorrect dimension."

    scale = (emb_domain[:, 1] - emb_domain[:, 0]) / 2
    shift = (emb_domain[:, 1] + emb_domain[:, 0]) / 2

    x = x * scale + shift

    sum1 = torch.sum(x**2, dim=1)
    sum2 = torch.sum(torch.cos(c * x), dim=1)
    term1 = -a * torch.exp(-b * torch.sqrt(sum1 / dim))
    term2 = -torch.exp(sum2 / dim)
    result = term1 + term2 + a + torch.exp(torch.tensor(1.0))
    return -result.unsqueeze(1)  # for minimisation


def ackley_HD(xx):
    """
    xx: clipped upto the unit cube box constraint
    """
    xx = xx @ Q.T

    x = torch.clip(
        xx,
        min=torch.tensor([[-1.0] * 4], device=device),
        max=torch.tensor([[1.0] * 4], device=device),
    )

    return ackley_4d(x)


class AckleyRank(BaseTestFunction):
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
            self.bounds = torch.tensor([[-1.] * self.dim, [1.] * self.dim]).T

        self.name = f"AckleyRank_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ackley function implementation in PyTorch.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The result of the Ackley function.
        """
        return ackley_HD(x)
