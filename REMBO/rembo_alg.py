from src.bo_vae_sdr.gp.fit_gp_model import train_gp_plain
from src.bo_vae_sdr.gp.gp_model import gp_model_unit_cube
from src.bo_vae_sdr.gp.optimize_acqf import optimize_acqf_with_warmup
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def rembo(d_e, d_orig, n_init, n_iter, high_dim_low_rank_obj_f):
    """
    d_e:    the effective dim such that the embedding space is d = d_e + 1
    d_orig: the ambient space dim
    n_init: the initial sampling size, 2*(problem dim)
    n_iter: the number of epochs
    high_dim_low_rank_obj_f: the synthetic high-dimensional rotated
                            low-rank function
    """

    # d-bound
    d = d_e + 1
    init_bounds = torch.tensor(
        [
            [
                -2.2 * torch.sqrt(torch.tensor(d, device=device)),
                2.2 * torch.sqrt(torch.tensor(d, device=device)),
            ]
        ]
        * (d),
        device=device,
    )

    # init_y samples
    init_y = (
        torch.rand(n_init, d, device=device) * (init_bounds[:, 1] - init_bounds[:, 0])
        + init_bounds[:, 0]
    )

    A = (
        torch.distributions.Normal(loc=0.0, scale=1 / (d_orig))
        .sample((d_orig, d))
        .to(device)
    )
    A, _ = torch.linalg.qr(A)

    init_x = A @ (init_y.T)

    init_x = torch.clip(
        init_x,
        min=torch.tensor([[-1.0]] * d_orig, device=device),
        max=torch.tensor([[1.0]] * d_orig, device=device),
    )

    init_f = high_dim_low_rank_obj_f(init_x)

    if init_f.dim() != 2:
        init_f = init_f.unsqueeze(-1)

    f0 = -init_f.max()

    best_f = torch.empty(0).to(device)

    for i in range(n_iter):
        gp_model = gp_model_unit_cube(train_x=init_y, train_y=init_f)
        train_gp_plain(gp_model=gp_model)
        new_y, _ = optimize_acqf_with_warmup(
            gp_model=gp_model,
            best_f=init_f.max(),
            current_bounds=init_bounds,
            n_warmup=3000,
            obj_f=None,
        )
        new_x = A @ (new_y.T)

        new_x = torch.clip(
            new_x,
            min=torch.tensor([[-1.0]] * d_orig, device=device),
            max=torch.tensor([[1.0]] * d_orig, device=device),
        )

        new_f = high_dim_low_rank_obj_f(new_x)
        if new_f.dim() != 2:
            new_f = new_f.reshape(-1, 1)

        init_y = torch.cat([init_y, new_y])
        init_f = torch.cat([init_f, new_f])

        best_f = torch.cat([best_f, init_f.max().reshape(-1, 1)])

    return f0.cpu().detach().numpy(), (-1) * best_f.cpu().detach().numpy()
