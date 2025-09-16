from ..base import BaseTestFunction
import torch
import numpy as np
from .rotation_mat import get_Q_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_dtype(torch.float64)


Q = get_Q_matrix(r=4, n=100)


def StyTang_4d(
    x: torch.Tensor, emb_domain=torch.tensor([[-5.0, 5.0]] * 4, device=device)
) -> torch.Tensor:
    """
    Styblinski-Tang function implementation in PyTorch.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The result of the function.
    """
    assert x.shape[1] == 4, "Input tensor has incorrect dimension."

    scale = (emb_domain[:, 1] - emb_domain[:, 0]) / 2
    shift = (emb_domain[:, 1] + emb_domain[:, 0]) / 2

    x = x * scale + shift

    sum1 = torch.sum(x**4, dim=1)
    sum2 = torch.sum(16 * x**2, dim=1)
    sum3 = torch.sum(5 * x, dim=1)
    result = (sum1 - sum2 + sum3) / 2
    return -result.unsqueeze(1)  # for minimisation


def StyTang_HD(xx):
    """
    xx: clipped upto the unit cube box constraint
    """

    xx = xx @ Q.T

    x = torch.clip(
        xx,
        min=torch.tensor([[-1.0] * 4], device=device),
        max=torch.tensor([[1.0] * 4], device=device),
    )

    return StyTang_4d(x)


class StyTangRank(BaseTestFunction):

    def __init__(self, dim, bounds: None | torch.Tensor = None) -> None:
        super().__init__(dim, bounds)

        self.optimal_input = torch.tensor([[0.0] * self.dim])
        self.optimal_value = 0.0

        if self.bounds is None:
            self.bounds = torch.tensor([[-1.] * self.dim, [1.] * self.dim]).T

        self.name = f"StyTangRank_{self.dim}d_{self.bounds_suffix()}"

    def func(self, x: torch.Tensor) -> torch.Tensor:
        return StyTang_HD(x).unsqueeze(-1)
