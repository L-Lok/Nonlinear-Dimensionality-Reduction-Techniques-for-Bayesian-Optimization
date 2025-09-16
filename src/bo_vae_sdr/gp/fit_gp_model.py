"""
Fit a SingleTaskGP model using different optimization methods.
"""
import gpytorch
import torch
from botorch.fit import (fit_gpytorch_mll, fit_gpytorch_mll_scipy,
                         fit_gpytorch_mll_torch)
from botorch.models import SingleTaskGP
from botorch.optim.stopping import ExpMAStoppingCriterion
from gpytorch import settings as gpt_settings
from gpytorch.mlls import ExactMarginalLogLikelihood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# double precision to avoid fit mll nan error
torch.set_default_dtype(torch.float64)


def train_gp_plain(gp_model: SingleTaskGP) -> SingleTaskGP:
    """
    Train a GP model using the built-in fit_gpytorch_mll.

    Args:
        train_x (torch.Tensor): training inputs with shape (n, d)
        train_y (torch.Tensor): training targets with shape (n, 1)

    Returns:
        SingleTaskGP: trained GP model    
    """

    # define the marginal log likelihood
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(device)

    # train the model
    gp_model.train()
    fit_gpytorch_mll(mll)

    # set the model to eval mode
    gp_model.eval()
    # return gp_model


def train_gp_scipy(gp_model: SingleTaskGP) -> SingleTaskGP:
    """
    Train a GP model using the scipy optimizer.

    Args:
        train_x (torch.Tensor): training inputs with shape (n, d)
        train_y (torch.Tensor): training targets with shape (n, 1)

    Returns:
        SingleTaskGP: trained GP model    
    """
    # define the marginal log likelihood
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(device)

    # train the model
    gp_model.train()
    fit_gpytorch_mll_scipy(mll)

    # set the model to eval mode
    gp_model.eval()
    return gp_model


def train_gp_torch(gp_model: SingleTaskGP) -> SingleTaskGP:
    """
    Train a GP model using the torch optimizer.

    Args:
        train_x (torch.Tensor): training inputs with shape (n, d)
        train_y (torch.Tensor): training targets with shape (n, 1)

    Returns:
        SingleTaskGP: trained GP model
    """
    # define the marginal log likelihood
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model).to(device)

    # train the model
    gp_model.train()
    fit_gpytorch_mll_torch(mll)

    # set the model to eval mode
    gp_model.eval()
    return gp_model


def train_gp_custom_implementation(train_x, train_y, options=None):
    """
    Train a GP model using a customized torch optimizer.

    Args:
        train_x (torch.Tensor): training inputs with shape (n, d)
        train_y (torch.Tensor): training targets with shape (n, 1)
        options (dict): optimization options

    Returns:
        SingleTaskGP: trained GP model
    """
    # define the likelihood
    gLikelihood = gpytorch.likelihoods.GaussianLikelihood()

    # define the model
    model = SingleTaskGP(train_X=train_x, train_Y=train_y,
                         likelihood=gLikelihood).to(train_x)

    # define the marginal log likelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_x)

    # train the model
    model.train()
    gLikelihood.train()

    optim_options = {"maxiter": 200, "minimize": True,
                     "rel_tol": 0.0005, **options}

    stopping_criterion = ExpMAStoppingCriterion(**optim_options)
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    i = 0
    stop = False
    while not stop:
        optimizer.zero_grad()

        # Evaluate the negative marginal log likelihood to be the loss function
        with gpt_settings.fast_computations(log_prob=True):
            output = mll.model(*train_inputs)
            loss = - mll.forward(output, train_targets).sum()
            loss.backward(retain_graph=True)
        optimizer.step()

        i += 1
        # print(f"[gp] Training GP model @ {i} iterations loss: {loss:.8e}...")
        stop = stopping_criterion.evaluate(fvals=loss.detach())

    print(f"[gp] Finished training GP model in {i} iterations...")

    # set the model to eval mode
    model.eval()
    gLikelihood.eval()
    return model
