import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def get_Q_matrix(r, n):
    """
    Sample random orthogonal matrix Q of size r x n

    For use in generating low-rank problems

    :param n: desired ambient dimension
    :param r: the problem rank
    :returns: Q
    """

    M = np.random.randn(n, r)
    Q_T_temp, R = np.linalg.qr(M, mode="reduced")
    signs = np.sign(np.diag(R))
    Q_T = Q_T_temp * signs

    Q = torch.from_numpy(Q_T.T).to(device)
    
    return Q
