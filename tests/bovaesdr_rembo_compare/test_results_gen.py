import logging
from pathlib import Path

import torch

from src.bo_vae_sdr.pipeline.retrain_vae_bo_sdr import \
    retrain_vae_bo_sdr_pipeline
from src.bo_vae_sdr.pipeline.retrain_vae_dml_bo import \
    retrain_vae_dml_bo_pipeline
from src.bo_vae_sdr.pipeline.vae_bo_sdr import vae_bo_sdr_pipeline

from src.bo_vae_sdr.test_funcs.REMBOLowrank import (AckleyRank, RosenbrockRank,
                                                    Shekel5Rank, Shekel7Rank,
                                                    StyTangRank)

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


def test_plot_part_2():

    test_funcs = [
        AckleyRank(100),
        AckleyRank(100),
        RosenbrockRank(100),
        RosenbrockRank(100),
        Shekel5Rank(100),
        Shekel5Rank(100),
        Shekel7Rank(100),
        Shekel7Rank(100),
        StyTangRank(100),
        StyTangRank(100),
    ]

    test_func_iter(
        test_funcs,
        Path("output/bovaesdr_rembo_compare/vae_bo_sdr_pipeline/encoders_100_25_5"),
        lambda _test_func: vae_bo_sdr_pipeline(
            _test_func,
            encoder_layers=[100, 25, 5],
            n_repeats=1,
            n_initial_points=2*_test_func.dim,
            n_trails=350,
        )
    )
    test_func_iter(
        test_funcs,
        Path("output/bovaesdr_rembo_compare/retrain_vae_bo_sdr_pipeline/encoders_100_25_5"),
        lambda _test_func: retrain_vae_bo_sdr_pipeline(
            _test_func,
            encoder_layers=[100, 25, 5],
            n_repeats=1,
            n_initial_points=500,
            n_trails=350,
        )
    )
    test_func_iter(
        test_funcs,
        Path("output/bovaesdr_rembo_compare/retrain_vae_dml_bo_pipeline/encoders_100_25_5"),
        lambda _test_func: retrain_vae_dml_bo_pipeline(
            _test_func,
            encoder_layers=[100, 25, 5],
            n_repeats=1,
            n_initial_points=500,
            n_trails=350,
        )
    )
