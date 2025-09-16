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
from src.bo_vae_sdr.test_funcs import Ackley, Rosenbrock

logging.basicConfig(level=logging.INFO)


plt.style.use("src/bo_sdr.mplstyle")


@pytest.mark.gpu
def test_compare_bo_and_bo_sdr():
    """
    Run all Bayesian Optimization (BO) pipelines
    """

    Path("output/standard_bo_sdr_compare/bo_warmup_pipeline").mkdir(parents=True, exist_ok=True)
    Path("output/standard_bo_sdr_compare/bo_warmup_sdr_pipeline").mkdir(parents=True, exist_ok=True)

    test_funcs = [
        Ackley(dim=10, bounds=torch.tensor([[-30.] * 10, [30.] * 10]).T),
        Rosenbrock(dim=10, bounds=torch.tensor([[-5.] * 10, [10.] * 10]).T),
    ]

    for test_func in test_funcs:

        test_func_name = test_func.name
        if not Path(f"output/standard_bo_sdr_compare/bo_warmup_pipeline/{test_func_name}_results.pth").exists():
            bo_results = bo_warmup_pipeline(
                test_func, n_repeats=5, n_trails=350, n_initial_points=2*test_func.dim)
            torch.save(
                bo_results,
                Path(
                    f"output/standard_bo_sdr_compare/bo_warmup_pipeline/{test_func_name}_results.pth")
            )

        if not Path(f"output/standard_bo_sdr_compare/bo_warmup_sdr_pipeline/{test_func_name}_results.pth").exists():
            bo_sdr_results = bo_warmup_sdr_pipeline(
                test_func, n_repeats=5, n_trails=350, n_initial_points=2*test_func.dim)
            torch.save(
                bo_sdr_results,
                Path(
                    f"output/standard_bo_sdr_compare/bo_warmup_sdr_pipeline/{test_func_name}_results.pth")
            )

        bo_results = torch.load(
            Path(
                f"output/standard_bo_sdr_compare/bo_warmup_pipeline/{test_func_name}_results.pth")
        )
        bo_f_min_matrix = np.abs(
            [
                np.array(bo_result["best_value_f_trace"]) - test_func.optimal_value for bo_result in bo_results
            ]
        )
        bo_sdr_results = torch.load(
            Path(
                f"output/standard_bo_sdr_compare/bo_warmup_sdr_pipeline/{test_func_name}_results.pth")
        )
        bo_sdr_f_min_matrix = np.abs(
            [
                np.array(bo_result["best_value_f_trace"]) - test_func.optimal_value for bo_result in bo_sdr_results
            ]
        )

        colors = sns.color_palette("Set1", n_colors=8)

        # we now calculate the mean and std of the best values for each epoch in bo_loop_no_SDR
        # first we find the absolute difference between the best value in our algorithm and the known analytic global minimum
        mean_no_SDR = np.mean(bo_f_min_matrix, axis=0)

        std_no_SDR = np.std(bo_f_min_matrix, axis=0)

        plt.figure()
        plt.plot(mean_no_SDR, label='BO without SDR', color=colors[0])
        plt.fill_between(range(len(mean_no_SDR)), mean_no_SDR -
                         std_no_SDR, mean_no_SDR + std_no_SDR, alpha=0.2, color=colors[0])

        # we now calculate the mean and std of the best values for each epoch in bo_loop_w_SDR

        mean_w_SDR = np.mean(bo_sdr_f_min_matrix, axis=0)

        std_w_SDR = np.std(bo_sdr_f_min_matrix, axis=0)
        plt.plot(mean_w_SDR, label='BO with SDR', color=colors[1])
        plt.fill_between(range(len(mean_w_SDR)), mean_w_SDR -
                         std_w_SDR, mean_w_SDR + std_w_SDR, alpha=0.2, color=colors[1])

        plt.xlabel('Number of epochs')
        plt.ylabel('Absolute difference from global minimum')
        plt.legend()
        plt.yscale('log')
        plt.savefig(Path(f"output/standard_bo_sdr_compare/{test_func_name}_compare.pdf"))
