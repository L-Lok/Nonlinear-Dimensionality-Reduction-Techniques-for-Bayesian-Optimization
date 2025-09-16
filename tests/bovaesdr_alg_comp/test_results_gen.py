

import logging
from pathlib import Path

import torch

from src.bo_vae_sdr.pipeline.bo_sdr import bo_warmup_sdr_pipeline
from src.bo_vae_sdr.pipeline.vae_bo_sdr import vae_bo_sdr_pipeline
from src.bo_vae_sdr.pipeline.retrain_vae_bo_sdr import retrain_vae_bo_sdr_pipeline
from src.bo_vae_sdr.pipeline.retrain_vae_dml_bo import retrain_vae_dml_bo_pipeline
from src.bo_vae_sdr.test_funcs import Ackley, Rosenbrock, Levy, StyblinskiTang, Rastrigin


logging.basicConfig(level=logging.INFO)



def test_func_iter(test_funcs, results_pth: Path, pipeline_func):
    
    results_pth.mkdir(parents=True, exist_ok=True)

    for i, test_func in enumerate(test_funcs, start=1):
            test_func_name = test_func.name

            data_file_path = results_pth/ "data"/ f"id_{i}_{test_func_name}_results.pth"
            data_file_path.parent.mkdir(parents=True, exist_ok=True)
            if data_file_path.exists():
                result = torch.load(
                    data_file_path
                )
            else:
                result = pipeline_func(
                    test_func
                )
                torch.save(
                    result,
                    data_file_path
                )




def test_plot_part_1():

    test_funcs = [
        Ackley(dim=100, bounds=torch.tensor([[-30.] * 100, [30.] * 100]).T),
        Ackley(dim=100, bounds=torch.tensor([[-30.] * 100, [30.] * 100]).T),
        Levy(dim=100, bounds=torch.tensor([[-10.] * 100, [10.] * 100]).T),
        Levy(dim=100, bounds=torch.tensor([[-10.] * 100, [10.] * 100]).T),
        Rosenbrock(dim=100, bounds=torch.tensor([[-5.] * 100, [10.] * 100]).T),
        Rosenbrock(dim=100, bounds=torch.tensor([[-5.] * 100, [10.] * 100]).T),
        StyblinskiTang(dim=100, bounds=torch.tensor(
            [[-5.] * 100, [5.] * 100]).T),
        StyblinskiTang(dim=100, bounds=torch.tensor(
            [[-5.] * 100, [5.] * 100]).T),
        Rastrigin(dim=100, bounds=torch.tensor(
            [[-5.12] * 100, [5.12] * 100]).T),
        Rastrigin(dim=100, bounds=torch.tensor(
            [[-5.12] * 100, [5.12] * 100]).T),

    ]

    test_func_iter(
        test_funcs,
        Path("output/bovaesdr_algs_compare/vae_bo_sdr_pipeline/encoders_100_30_2"),
        lambda _test_func: vae_bo_sdr_pipeline(
            _test_func,
            encoder_layers=[100, 30, 2],
            n_repeats=1,
            n_initial_points=2*_test_func.dim,
            n_trails=350,
        )
    )
    test_func_iter(
        test_funcs,
        Path("output/bovaesdr_algs_compare/vae_bo_sdr_pipeline/encoders_100_32_10"),
        lambda _test_func: vae_bo_sdr_pipeline(
            _test_func,
            encoder_layers=[100, 32, 10],
            n_repeats=1,
            n_initial_points=2*_test_func.dim,
            n_trails=500,
        )
    )

    test_func_iter(
        test_funcs,
        Path("output/bovaesdr_algs_compare/retrain_vae_bo_sdr_pipeline/encoders_100_30_2"),
        lambda _test_func: retrain_vae_bo_sdr_pipeline(
            _test_func,
            encoder_layers=[100, 30, 2],
            n_repeats=1,
            n_initial_points=2*_test_func.dim,
            n_trails=350,
        )
    )
    test_func_iter(
        test_funcs,
        Path("output/bovaesdr_algs_compare/retrain_vae_bo_sdr_pipeline/encoders_100_32_10"),
        lambda _test_func: retrain_vae_bo_sdr_pipeline(
            _test_func,
            encoder_layers=[100, 32, 10],
            n_repeats=1,
            n_initial_points=2*_test_func.dim,
            n_trails=500,
        )
    )

    test_func_iter(
        test_funcs,
        Path("output/bovaesdr_algs_compare/retrain_vae_dml_bo_pipeline/encoders_100_30_2"),
        lambda _test_func: retrain_vae_dml_bo_pipeline(
            _test_func,
            encoder_layers=[100, 30, 2],
            n_repeats=1,
            n_initial_points=2*_test_func.dim,
            n_trails=350,
        )
    )
    test_func_iter(
        test_funcs,
        Path("output/bovaesdr_algs_compare/retrain_vae_dml_bo_pipeline/encoders_100_32_10"),
        lambda _test_func: retrain_vae_dml_bo_pipeline(
            _test_func,
            encoder_layers=[100, 32, 10],
            n_repeats=1,
            n_initial_points=2*_test_func.dim,
            n_trails=500,
        )
    )

