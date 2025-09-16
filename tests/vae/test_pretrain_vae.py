import json
from pathlib import Path

import torch
from accelerate import Accelerator

from src.bo_vae_sdr.vae.data_generator import DatasetPrepare
from src.bo_vae_sdr.vae.model import (LSBOVAE, save_model_and_results,
                                      training_vae)
from src.bo_vae_sdr.vae.plot import VAEModel_Plot


def test_pretrain_vae():

    all_encoder_layers = [
        [5, 3, 2],
        [10, 5, 2],
        [10, 5],
        [20, 6, 2],
        [50, 20, 2],
        [50, 20, 6, 2],
        [50, 32, 8, 4],
        [100, 30, 2],
        [100, 32, 8, 2],
        [100, 32, 8, 4],
        [100, 32, 10],
        [100, 25, 5],
        [100, 50, 25],
        [100, 50],
    ]
    for encoder_layers in all_encoder_layers:
        current_output_path = Path("output/pretrained_vae") / \
            f"pretrained_layers_{'_'.join([str(x) for x in encoder_layers])}"
        current_output_path.mkdir(parents=True, exist_ok=True)

        if (current_output_path/"model.pth").exists():
            continue
        hparams = {
            "beta_start": None,
            "beta_final": 1.0,
            "beta_step": 1.1,
            "beta_step_freq": 5,
            "beta_warmup": 10,
            "beta_metric_loss": 0.,
            "metric_loss": None,
            # data generator params
            "num_sample": 20000 if encoder_layers[0] < 50 else int(1e6),
            "feature_dim": encoder_layers[0],
            "target_dim": 1,
            "data_split": 0.9,
            "sampling_method": "trunc_normal",
            "batch_size": 256 if encoder_layers[0] < 50 else 512,
            "obj_func_kw": "Rosenbrock_nd", # it does not matter here
            # weight
            "weight_type": "uniform",
            # vae encoder
            "latent_dim": encoder_layers[-1],
            "encoder_layer_dims": encoder_layers,
            "decoder_layer_dims": encoder_layers.copy()[::-1],
        }
        opt_params = {"lr": 1e-3}
        model = LSBOVAE(hparams=hparams)

        with open(current_output_path / "hparams.json", "w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=4)
        with open(current_output_path / "opt_params.json", "w", encoding="utf-8") as f:
            json.dump(opt_params, f, indent=4)

        if (current_output_path / "train_set.pth").exists() and (current_output_path / "test_set.pth").exists():
            print("Data already exists")

            train_loader, test_loader = DatasetPrepare(hparams=hparams).load(
                current_output_path / "train_set.pth",  current_output_path / "test_set.pth"
            )
        else:
            train_loader, test_loader = DatasetPrepare(
                hparams=hparams).setup()

            torch.save(train_loader.dataset.tensors,
                       current_output_path / "train_set.pth")
            torch.save(test_loader.dataset.tensors,
                       current_output_path / "test_set.pth")

        accelerator = Accelerator()

        model = accelerator.prepare(model)
        train_loader, test_loader = accelerator.prepare(
            train_loader, test_loader)
        trained_model, results = training_vae(
            epochs=300 + 1,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            opt_params=opt_params,
            loss_type="mse",
            mean=False,
            do_test=True,
            pretraining=True,
        )

        save_model_and_results(
            current_output_path, trained_model, results
        )

        VAEModel_Plot(
            hparams=hparams,
            model_path=current_output_path/"model.pth",
            result_path=current_output_path/"results.json",
            train_path=current_output_path/"train_set.pth",
            test_path=current_output_path/"test_set.pth",
        ).vae_plot_2d(output_path=current_output_path)
