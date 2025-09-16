import logging
import time

import torch

from ..gp.fit_gp_model import train_gp_plain
from ..gp.gp_model import gp_model_unit_cube, NOISE_FREE_CONST
from ..gp.optimize_acqf import optimize_acqf_with_warmup
from ..test_funcs.base import BaseTestFunction
from ..utils.initialize_points import initialize_points_uniformly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


def bo_warmup(
    obj_f,
    init_train_x,
    init_train_y,
    original_bounds,
    n_trials,
    train_y_var=NOISE_FREE_CONST
):
    """
    Bayesian Optimization with Warmup and Sequential Domain Reduction
    """

    train_x = init_train_x
    train_y = init_train_y
    acq_val_trace = torch.empty(0).to(init_train_x)
    best_value_f_trace = torch.empty(0).to(init_train_x)
    current_bounds = original_bounds.clone()

    gp_model = None

    for j in range(n_trials):
        logging.info("Trial %s", j)

        best_value = train_y.max()

        gp_model = gp_model_unit_cube(
            train_x,
            train_y,
            train_y_var=train_y_var
            #   we guide the gp model with previous trained covariance
            # covar_module=gp_model.covar_module if gp_model is not None else None
        )
        start_time = time.time()
        train_gp_plain(gp_model)
        print("train_gp_plain", time.time() - start_time)
        start_time = time.time()

        new_point_x, acq_val_ = optimize_acqf_with_warmup(
            gp_model=gp_model,
            best_f=best_value,
            current_bounds=current_bounds,
        )
        print("optimize_acqf_with_warmup", time.time() - start_time)
        new_point_y = obj_f(new_point_x)

        # update training sets
        train_x = torch.cat([train_x, new_point_x])
        train_y = torch.cat([train_y, new_point_y])

        # keep track of acq_val and best f found so far
        acq_val_trace = torch.cat([acq_val_trace, acq_val_.detach().view(1)])
        best_value_f_trace = torch.cat(
            [best_value_f_trace, best_value.detach().view(1)])
        j += 1
 
        best_value_x = train_x[train_y.argmax()]
        best_value = train_y.max()

        logging.info("Best value found so far: %s @ %s",
                     best_value, best_value_x)

    return best_value, acq_val_trace, best_value_f_trace, train_x[-n_trials:], train_y[-n_trials:]


def bo_warmup_pipeline(
    test_func: BaseTestFunction,
    n_repeats: int = 25,
    n_initial_points: int = 50,
    n_trails: int = 500,
    train_y_var=NOISE_FREE_CONST
) -> tuple[list, dict]:
    results = []
    best_value_f_for_all_repeats = []
    for i in range(n_repeats):
        logging.info("Repeat %s @ %s", test_func.name, i)
        train_x = initialize_points_uniformly(
            n_initial_points,
            test_func.dim,
            test_func.bounds
        )
        train_y = test_func.func(train_x)

        best_value, acq_val_trace, best_value_f_trace, train_x_trace, train_y_trace = bo_warmup(
            obj_f=test_func.func,
            init_train_x=train_x.to(device),
            init_train_y=train_y.to(device),
            original_bounds=test_func.bounds.to(device),
            n_trials=n_trails,
            train_y_var=train_y_var
        )

        best_value_f_for_all_repeats.append(best_value)

        results.append({
            "init_train_x": train_x.detach().cpu(),
            "init_train_y": train_y.detach().cpu(),
            "query_train_x": train_x_trace.detach().cpu(),
            "query_train_y": train_y_trace.detach().cpu(),
            "acq_val_trace": acq_val_trace.detach().cpu(),
            "best_value_f_trace": best_value_f_trace.detach().cpu(),
        })

    return results
