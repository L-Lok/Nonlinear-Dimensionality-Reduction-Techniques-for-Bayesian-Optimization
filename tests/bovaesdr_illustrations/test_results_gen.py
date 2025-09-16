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

from src.bo_vae_sdr.pipeline.retrain_vae_bo_sdr import \
    retrain_vae_bo_sdr_pipeline
from src.bo_vae_sdr.pipeline.retrain_vae_dml_bo import \
    retrain_vae_dml_bo_pipeline

from src.bo_vae_sdr.pipeline.vae_bo import vae_bo_pipeline
from src.bo_vae_sdr.pipeline.vae_bo_sdr import vae_bo_sdr_pipeline
from src.bo_vae_sdr.test_funcs import Ackley, Rosenbrock
from src.bo_vae_sdr.test_funcs.base import ShiftedTestFunction

logging.basicConfig(level=logging.INFO)


plt.style.use("src/bo_sdr.mplstyle")


@pytest.mark.gpu
def test_compare_vae_bo():
    """
    Run all VAE Bayesian Optimization (BO) pipelines
    """

    Path("output/Three_bovaesdr_algs_illustrations/vae_bo_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/Three_bovaesdr_algs_illustrations/vae_bo_sdr_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/Three_bovaesdr_algs_illustrations/retrain_vae_bo_sdr_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/Three_bovaesdr_algs_illustrations/retrain_vae_dml_bo_pipeline").mkdir(parents=True, exist_ok=True)

    test_funcs = [
        # ShiftedTestFunction(
        #     Ackley(dim=10, bounds=torch.tensor([[-30.] * 10, [30.] * 10]).T)
            
        #     ),

        Ackley(dim=10, bounds=torch.tensor([[-30.] * 10, [30.] * 10]).T),
        Rosenbrock(dim=10, bounds=torch.tensor([[-5.] * 10, [10.] * 10]).T),
        Ackley(dim=100, bounds=torch.tensor([[-30.] * 100, [30.] * 100]).T),
        Rosenbrock(dim=100, bounds=torch.tensor([[-5.] * 100, [10.] * 100]).T),
    ]

    for test_func in test_funcs:
        test_func_name = test_func.name

        if test_func.dim == 10:
            all_encoder_layers = [
                [10, 5],
                [10, 5, 2]
            ]
        elif test_func.dim == 100:
            all_encoder_layers = [
                [100, 30, 2],
                [100, 32, 10],
                [100, 50],
            ]
        for _encoder_layers in all_encoder_layers:
            vae_bo_path = Path(
                f"output/Three_bovaesdr_algs_illustrations/vae_bo_pipeline/encoders_{'_'.join([str(x) for x in _encoder_layers])}/{test_func_name}_results.pth")
            vae_bo_path.parent.mkdir(exist_ok=True, parents=True)

            if not vae_bo_path.exists():
                vae_bo_results = vae_bo_pipeline(
                    test_func,
                    n_repeats=5,
                    n_trails=350,
                    n_initial_points=2*test_func.dim,
                    encoder_layers=_encoder_layers
                )
                torch.save(
                    vae_bo_results,
                    vae_bo_path
                )
            else:
                vae_bo_results = torch.load(
                    vae_bo_path
                )

            vae_bo_sdr_path = Path(
                f"output/Three_bovaesdr_algs_illustrations/vae_bo_sdr_pipeline/encoders_{'_'.join([str(x) for x in _encoder_layers])}/{test_func_name}_results.pth")
            vae_bo_sdr_path.parent.mkdir(exist_ok=True, parents=True)

            if not vae_bo_sdr_path.exists():
                vae_bo_sdr_results = vae_bo_sdr_pipeline(
                    test_func,
                    n_repeats=5,
                    n_trails=350,
                    n_initial_points=2*test_func.dim,
                    encoder_layers=_encoder_layers
                )
                torch.save(
                    vae_bo_sdr_results,
                    vae_bo_sdr_path
                )
            else:
                vae_bo_sdr_results = torch.load(
                    vae_bo_sdr_path
                )

            retrain_vae_bo_sdr_path = Path(
                f"output/Three_bovaesdr_algs_illustrations/retrain_vae_bo_sdr_pipeline/encoders_{'_'.join([str(x) for x in _encoder_layers])}/{test_func_name}_results.pth")
            retrain_vae_bo_sdr_path.parent.mkdir(exist_ok=True, parents=True)

            if not retrain_vae_bo_sdr_path.exists():
                retrain_vae_bo_sdr_results = retrain_vae_bo_sdr_pipeline(
                    test_func,
                    n_repeats=5,
                    n_trails=350,
                    n_initial_points=500,
                    encoder_layers=_encoder_layers
                )
                torch.save(
                    retrain_vae_bo_sdr_results,
                    retrain_vae_bo_sdr_path
                )
            else:
                retrain_vae_bo_sdr_results = torch.load(
                    retrain_vae_bo_sdr_path
                )

            retrain_vae_dml_path = Path(
                f"output/Three_bovaesdr_algs_illustrations/retrain_vae_dml_bo_pipeline/encoders_{'_'.join([str(x) for x in _encoder_layers])}/{test_func_name}_results.pth")
            retrain_vae_dml_path.parent.mkdir(exist_ok=True, parents=True)

            if not retrain_vae_dml_path.exists():
                retrain_vae_dml_results = retrain_vae_dml_bo_pipeline(
                    test_func,
                    n_repeats=5,
                    n_trails=350,
                    n_initial_points=500,
                    encoder_layers=_encoder_layers
                )
                torch.save(
                    retrain_vae_dml_results,
                    retrain_vae_dml_path
                )
            else:
                retrain_vae_dml_results = torch.load(
                    retrain_vae_dml_path
                )