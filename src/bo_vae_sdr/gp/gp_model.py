import gpytorch
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize


NOISE_FREE_CONST = 1e-6


def gp_model_base(train_x: torch.Tensor, train_y: torch.Tensor, covar_module=None):

    # define the likelihood
    g_likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # define the model
    gp_model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=torch.full_like(train_y, NOISE_FREE_CONST),
        likelihood=g_likelihood,
        covar_module=covar_module
    ).to(train_x)
    return gp_model


def gp_model_unit_cube(train_x: torch.Tensor, train_y: torch.Tensor, train_y_var=NOISE_FREE_CONST, covar_module=None):

    # define the likelihood
    g_likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # define the model
    gp_model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=torch.full_like(train_y, train_y_var),
        likelihood=g_likelihood,
        covar_module=covar_module,
        # input/output transformation ref: https://github.com/pytorch/botorch/issues/1150#issuecomment-1086394146
        # input transformation to unit cube
        input_transform=Normalize(d=train_x.shape[1]),
        # one dimensional functional output
        outcome_transform=Standardize(m=1),
    ).to(train_x)
    return gp_model