import torch
import numpy as np
from REMBO.oneHD_low_rank_functions.rotation_mat import get_Q_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


Q = get_Q_matrix(r=4, n=100)


def Rosenbrock_4d(
    x, a=1, b=100, emb_domain=torch.tensor([[-5.0, 10.0]] * 4, device=device)
):
    """
    x: a n-dim vector, x = [[x_1, x_2, ..., x_n],.....] NOTICE INPUT


    When n = 3, i.e., 3-dim, the global minimum is at (1, , 1, 1)

    When 4 <= n <= 7, the global minimum is at (1, 1, ..., 1)
    and a local minimum near (-1, 1, ..., 1)
    """

    scale = (emb_domain[:, 1] - emb_domain[:, 0]) / 2
    shift = (emb_domain[:, 1] + emb_domain[:, 0]) / 2

    x = x * scale + shift

    f = torch.sum((a - x[:, :-1]) ** 2 + b * (x[:, 1:] - x[:, :-1] ** 2) ** 2, dim=1)

    return -f


def Rosenbrock_HD(xx):
    xx = (Q @ xx).T

    x = torch.clip(
        xx,
        min=torch.tensor([[-1.0] * 4], device=device),
        max=torch.tensor([[1.0] * 4], device=device),
    )
    return Rosenbrock_4d(x)
