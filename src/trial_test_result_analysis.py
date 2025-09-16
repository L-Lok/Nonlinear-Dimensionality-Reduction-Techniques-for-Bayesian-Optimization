

import logging
from pathlib import Path

import torch

from bo_vae_sdr.pipeline.bo_sdr import bo_warmup_sdr_pipeline
from bo_vae_sdr.results.solver_comparision_plotting import (
    create_plots, get_solved_times_for_file, test_funcs_to_plots)
from bo_vae_sdr.test_funcs import Ackley, Rosenbrock, Sphere

logging.basicConfig(level=logging.INFO)


def bo_sdr_post_processing():
    """

    Reference file format:
    https://lindonroberts.github.io/opt/resources.html#methodology-for-comparing-solvers

    """
    test_funcs = [
        # Sphere(dim=5, bounds=torch.tensor([[-30.] * 5, [30.] * 5]).T),
        # Sphere(dim=10, bounds=torch.tensor([[-30.] * 10, [30.] * 10]).T),
        # Sphere(dim=25, bounds=torch.tensor([[-30.] * 25, [30.] * 25]).T),
        # Ackley(dim=5, bounds=torch.tensor([[-2.] * 5, [2.] * 5]).T),
        # Ackley(dim=5, bounds=torch.tensor([[-10.] * 5, [10.] * 5]).T),
        # Ackley(dim=5, bounds=torch.tensor([[-30.] * 5, [30.] * 5]).T),
        # Ackley(dim=10, bounds=torch.tensor([[-2.] * 10, [2.] * 10]).T),
        # Ackley(dim=10, bounds=torch.tensor([[-10.] * 10, [10.] * 10]).T),
        # Ackley(dim=10, bounds=torch.tensor([[-30.] * 10, [30.] * 10]).T),
        # Ackley(dim=25, bounds=torch.tensor([[-2.] * 25, [2.] * 25]).T),
        # Ackley(dim=25, bounds=torch.tensor([[-10.] * 25, [10.] * 25]).T),
        # Ackley(dim=25, bounds=torch.tensor([[-30.] * 25, [30.] * 25]).T),
        Rosenbrock(dim=5, bounds=torch.tensor([[-2.] * 5, [2.] * 5]).T),
        Rosenbrock(dim=5, bounds=torch.tensor([[-5.] * 5, [10.] * 5]).T),
        # Rosenbrock(dim=10, bounds=torch.tensor([[-2.] * 10, [2.] * 10]).T),
        # Rosenbrock(dim=10, bounds=torch.tensor([[-5.] * 10, [10.] * 10]).T),
        # Rosenbrock(dim=25, bounds=torch.tensor([[-2.] * 25, [2.] * 25]).T),
        # Rosenbrock(dim=25, bounds=torch.tensor([[-5.] * 25, [10.] * 25]).T),
    ]

    Path(
        "output/bo_warmup_sdr_pipeline/result_analysis").mkdir(parents=True, exist_ok=True)

    test_funcs_to_plots(
        test_funcs,
        Path("output/bo_warmup_sdr_pipeline")
    )


if __name__ == '__main__':
    bo_sdr_post_processing()
