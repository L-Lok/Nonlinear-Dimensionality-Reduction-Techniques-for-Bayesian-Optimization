import torch
import numpy as np
from REMBO.oneHD_low_rank_functions.rotation_mat import get_Q_matrix

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
    return -result  # for minimisation


def ackley_HD(xx):
    """
    xx: clipped upto the unit cube box constraint
    """
    xx = (Q @ xx).T

    x = torch.clip(
        xx,
        min=torch.tensor([[-1.0] * 4], device=device),
        max=torch.tensor([[1.0] * 4], device=device),
    )

    return ackley_4d(x)
