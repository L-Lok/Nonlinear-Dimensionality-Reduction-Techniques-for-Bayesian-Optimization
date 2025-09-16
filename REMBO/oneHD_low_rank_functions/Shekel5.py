import torch
import numpy as np
from REMBO.oneHD_low_rank_functions.rotation_mat import get_Q_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


Q = get_Q_matrix(r=4, n=100)


def shekel5(x, emb_domain=torch.tensor([[0.0, 10.0]] * 4, device=device)):
    """
    Scale domain from [-1, 1] to emb_domain
    Shekel 5 function. Global minimum is f=-10.1499 at x = np.array([4.0, 4.0, 4.0, 4.0])
    :param x: Input vector, number x length 4
    :return: Float.
    """

    scale = (emb_domain[:, 1] - emb_domain[:, 0]) / 2
    shift = (emb_domain[:, 1] + emb_domain[:, 0]) / 2

    if x.shape[1] != 4:
        raise ValueError(f"Input x must have 4 features, but got {x.shape[1]} features.")
    
    x = x * scale + shift

    x = x.detach().cpu().numpy()

    a1 = np.array([4.0, 4.0, 4.0, 4.0])
    a2 = np.array([1.0, 1.0, 1.0, 1.0])
    a3 = np.array([8.0, 8.0, 8.0, 8.0])
    a4 = np.array([6.0, 6.0, 6.0, 6.0])
    a5 = np.array([3.0, 7.0, 3.0, 7.0])
    a = [a1, a2, a3, a4, a5]
    c = np.array([0.1, 0.2, 0.2, 0.4, 0.4])
    f = []
    for x_ in x:
        f_ = -np.sum(
            np.array([1.0 / (np.sum((x_ - a[i]) ** 2) + c[i]) for i in range(5)])
        )
        f.append(f_)
    return torch.from_numpy(-np.array(f)).to(device)


def shekel5_HD(xx):
    """
    xx: R^D, clipped upto the unit cube box constraint
    """
    xx = (Q @ xx).T

    x = torch.clip(
        xx,
        min=torch.tensor([[-1.0] * 4], device=device),
        max=torch.tensor([[1.0] * 4], device=device),
    )
    return shekel5(x)
