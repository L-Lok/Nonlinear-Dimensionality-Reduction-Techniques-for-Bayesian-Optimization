"""
Run all Bayesian Optimization (BO) pipelines
"""
import logging
from pathlib import Path

import numpy as np
import pytest
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from src.bo_vae_sdr.pipeline.bo import bo_warmup_pipeline
from src.bo_vae_sdr.pipeline.bo_sdr import bo_warmup_sdr_pipeline
from src.bo_vae_sdr.pipeline.vae_bo_sdr import vae_bo_sdr_pipeline
from src.bo_vae_sdr.pipeline.retrain_vae_bo_sdr import retrain_vae_bo_sdr_pipeline
from src.bo_vae_sdr.pipeline.retrain_vae_dml_bo import retrain_vae_dml_bo_pipeline
from src.bo_vae_sdr.pipeline.vae_bo import vae_bo_pipeline
from src.bo_vae_sdr.test_funcs import Ackley, Rosenbrock
from src.bo_vae_sdr.test_funcs.base import NoisyTestFunction

logging.basicConfig(level=logging.INFO)


plt.style.use("src/bo_sdr.mplstyle")


@pytest.mark.gpu
def test_gen_data_10d():
    """
    Run all Bayesian Optimization (BO) pipelines
    """

    Path("output/appendix/bo_warmup_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/bo_warmup_sdr_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/vae_bo_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/vae_bo_sdr_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/retrain_vae_bo_sdr_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/retrain_vae_dml_bo_pipeline").mkdir(parents=True, exist_ok=True)

    test_funcs = [
        NoisyTestFunction(Ackley(
            dim=10, bounds=torch.tensor([[-30.] * 10, [30.] * 10]).T), noise_sigma_y=1e-2),
        NoisyTestFunction(Rosenbrock(dim=10, bounds=torch.tensor(
            [[-5.] * 10, [10.] * 10]).T), noise_sigma_y=1e-2),
    ]

    for test_func in test_funcs:

        test_func_name = test_func.name
        if not Path(f"output/appendix/bo_warmup_pipeline/{test_func_name}_results.pth").exists():
            bo_results = bo_warmup_pipeline(
                test_func, n_repeats=5, n_trails=500, n_initial_points=2*test_func.dim,
                train_y_var=1e-2)
            torch.save(
                bo_results,
                Path(
                    f"output/appendix/bo_warmup_pipeline/{test_func_name}_results.pth")
            )

        if not Path(f"output/appendix/bo_warmup_sdr_pipeline/{test_func_name}_results.pth").exists():
            bo_sdr_results = bo_warmup_sdr_pipeline(
                test_func, n_repeats=5, n_trails=500, n_initial_points=2*test_func.dim,train_y_var=1e-2)
            torch.save(
                bo_sdr_results,
                Path(
                    f"output/appendix/bo_warmup_sdr_pipeline/{test_func_name}_results.pth")
            )

        if not Path(f"output/appendix/vae_bo_pipeline/{test_func_name}_results.pth").exists():
            vae_bo_results = vae_bo_pipeline(
                test_func,
                encoder_layers=[10, 5, 2],
                n_repeats=5,
                n_initial_points=2*test_func.dim,
                n_trails=500,
                train_y_var=1e-2
            )
            torch.save(
                vae_bo_results,
                Path(
                    f"output/appendix/vae_bo_pipeline/{test_func_name}_results.pth")
            )

        if not Path(f"output/appendix/vae_bo_sdr_pipeline/{test_func_name}_results.pth").exists():
            vae_bo_sdr_results = vae_bo_sdr_pipeline(
                test_func,
                encoder_layers=[10, 5, 2],
                n_repeats=5,
                n_initial_points=2*test_func.dim,
                n_trails=500,
                train_y_var=1e-2
            )
            torch.save(
                vae_bo_sdr_results,
                Path(
                    f"output/appendix/vae_bo_sdr_pipeline/{test_func_name}_results.pth")
            )
        if not Path(f"output/appendix/retrain_vae_bo_sdr_pipeline/{test_func_name}_results.pth").exists():
            retrain_vae_bo_sdr_results = retrain_vae_bo_sdr_pipeline(
                test_func,
                encoder_layers=[10, 5, 2],
                n_repeats=5,
                n_initial_points=2*test_func.dim,
                n_trails=500,
                train_y_var=1e-2
            )
            torch.save(
                retrain_vae_bo_sdr_results,
                Path(
                    f"output/appendix/retrain_vae_bo_sdr_pipeline/{test_func_name}_results.pth")
            )
        if not Path(f"output/appendix/retrain_vae_dml_bo_pipeline/{test_func_name}_results.pth").exists():
            retrain_vae_dml_bo_results = retrain_vae_dml_bo_pipeline(
                test_func,
                encoder_layers=[10, 5, 2],
                n_repeats=5,
                n_initial_points=2*test_func.dim,
                n_trails=500,
                train_y_var=1e-2
            )
            torch.save(
                retrain_vae_dml_bo_results,
                Path(
                    f"output/appendix/retrain_vae_dml_bo_pipeline/{test_func_name}_results.pth")
            )


@pytest.mark.gpu
def test_gen_data_100d():
    """
    Run all Bayesian Optimization (BO) pipelines
    """

    Path("output/appendix/bo_warmup_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/bo_warmup_sdr_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/vae_bo_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/vae_bo_sdr_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/retrain_vae_bo_sdr_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/appendix/retrain_vae_dml_bo_pipeline").mkdir(parents=True, exist_ok=True)

    test_funcs = [
        NoisyTestFunction(Ackley(
            dim=100, bounds=torch.tensor([[-30.] * 100, [30.] * 100]).T), noise_sigma_y=1e-2),
        NoisyTestFunction(Rosenbrock(dim=100, bounds=torch.tensor(
            [[-5.] * 100, [10.] * 100]).T), noise_sigma_y=1e-2),
    ]

    for test_func in test_funcs:

        test_func_name = test_func.name

        if not Path(f"output/appendix/vae_bo_pipeline/{test_func_name}_results.pth").exists():
            vae_bo_results = vae_bo_pipeline(
                test_func,
                encoder_layers=[100, 30, 2],
                n_repeats=5,
                n_initial_points=2*test_func.dim,
                n_trails=500,
                train_y_var=1e-2
            )
            torch.save(
                vae_bo_results,
                Path(
                    f"output/appendix/vae_bo_pipeline/{test_func_name}_results.pth")
            )

        if not Path(f"output/appendix/vae_bo_sdr_pipeline/{test_func_name}_results.pth").exists():
            vae_bo_sdr_results = vae_bo_sdr_pipeline(
                test_func,
                encoder_layers=[100, 30, 2],
                n_repeats=5,
                n_initial_points=2*test_func.dim,
                n_trails=500,
                train_y_var=1e-2
            )
            torch.save(
                vae_bo_sdr_results,
                Path(
                    f"output/appendix/vae_bo_sdr_pipeline/{test_func_name}_results.pth")
            )
        if not Path(f"output/appendix/retrain_vae_bo_sdr_pipeline/{test_func_name}_results.pth").exists():
            retrain_vae_bo_sdr_results = retrain_vae_bo_sdr_pipeline(
                test_func,
                encoder_layers=[100, 30, 2],
                n_repeats=5,
                n_initial_points=2*test_func.dim,
                n_trails=500,
                train_y_var=1e-2
            )
            torch.save(
                retrain_vae_bo_sdr_results,
                Path(
                    f"output/appendix/retrain_vae_bo_sdr_pipeline/{test_func_name}_results.pth")
            )
        if not Path(f"output/appendix/retrain_vae_dml_bo_pipeline/{test_func_name}_results.pth").exists():
            retrain_vae_dml_bo_results = retrain_vae_dml_bo_pipeline(
                test_func,
                encoder_layers=[100, 30, 2],
                n_repeats=5,
                n_initial_points=2*test_func.dim,
                n_trails=500,
                train_y_var=1e-2
            )
            torch.save(
                retrain_vae_dml_bo_results,
                Path(
                    f"output/appendix/retrain_vae_dml_bo_pipeline/{test_func_name}_results.pth")
            )
