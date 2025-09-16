
import logging
import time

import gpytorch
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_acqf_with_warmup(
    gp_model: SingleTaskGP,
    best_f,
    current_bounds,
    n_warmup: int = 10000 # better adopt a GO solver to avoid local optima
):
    """

    Warmup Ref:
    https://github.com/bayesian-optimization/BayesianOptimization/blob/48e4816dfbfa62fc7fa7b2e9404c106622ba31e3/bayes_opt/util.py



    """
    current_bounds = current_bounds.to(device)
    # Can use LogEI for better performance
    # NumericsWarning: qExpectedImprovement has known numerical issues
    # that lead to suboptimal optimization performance. It is strongly recommended to simply replace
    # qExpectedImprovement    -->     qLogExpectedImprovement
    # instead, which fixes the issues and has the same API. See https://arxiv.org/abs/2310.20708 for details.
    acquisition_function = ExpectedImprovement(
        gp_model,
        best_f=best_f,
        # sampler=sampler
    ).to(device)

    # do warmup
    # Warm up with random points
    # Idea comment: see BayesOpt package for reference about how to do warmup when incorporating search domain update
    x_tries = torch.rand(n_warmup, current_bounds.shape[0]).to(
        device) * (current_bounds[:, 1] - current_bounds[:, 0]) + current_bounds[:, 0]

    # unsqueeze to add batch dimension
    acq_warm_up_values = acquisition_function.forward(
        x_tries.unsqueeze(1))

    acq_warm_up_values_max_id = acq_warm_up_values.argmax()
    x_max = x_tries[acq_warm_up_values_max_id]
    max_acq = acq_warm_up_values[acq_warm_up_values_max_id]

    
    start_time = time.time()
    new_point_x, acq_val_ = optimize_acqf(
        acq_function=acquisition_function,
        # acq_function = MC_UCB,
        bounds=current_bounds.T,
        q=1,
        num_restarts=1024,
        # options={
        #     "x0": x_max.unsqueeze(0).cpu().detach().numpy(),
        # },
        options={
            "sample_around_best": True,
        },
        raw_samples=10000,
        # sequential=True,
        timeout_sec=1,
        return_best_only=True
    )
    logging.info(f"bo optimize_acqf takes {time.time()- start_time}")
    # end = time.time()
    # logging.info('opti_acqf',end - start)
    # logging.info("max_acq", max_acq, "acq_val_", acq_val_)
    if max_acq > acq_val_:
        logging.info("USE WARMUP")
        return x_max.unsqueeze(0), max_acq
    else:
        return new_point_x, acq_val_
