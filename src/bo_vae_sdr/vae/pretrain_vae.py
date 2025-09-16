import json
from pathlib import Path

import torch

from .data_generator import DatasetPrepare
from .model import LSBOVAE, save_model_and_results, training_vae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain_vae(

    encoder_layers=[5, 3, 2],

) -> LSBOVAE:
    current_output_path = Path("output/pretrained_vae") / \
        f"pretrained_layers_{'_'.join([str(x) for x in encoder_layers])}"
    current_output_path.mkdir(parents=True, exist_ok=True)

    pre_train_hparams = {
        "beta_start": None,
        "beta_final": 1.0,
        "beta_step": 1.1,
        "beta_step_freq": 5,
        "beta_warmup": 10,
        "beta_metric_loss": 0.,
        "metric_loss": None,
        # data generator params
        "num_sample": 20000,
        "feature_dim": encoder_layers[0],
        "target_dim": 1,
        "data_split": 0.9,
        "sampling_method": "trunc_normal",
        "batch_size": 256,
        "obj_func_kw": "Rosenbrock_nd", # it does not matter
        # weight
        "weight_type": "uniform",
        # vae encoder
        "latent_dim": encoder_layers[-1],
        "encoder_layer_dims": encoder_layers,
        "decoder_layer_dims": encoder_layers.copy()[::-1],
    }
    opt_params = {"lr": 1e-3}
    model = LSBOVAE(hparams=pre_train_hparams)

    if (current_output_path / "train_set.pth").exists() and Path(current_output_path/"test_set.pth").exists():
        train_loader, test_loader = DatasetPrepare(hparams=pre_train_hparams).load(
            current_output_path / "train_set.pth",  current_output_path / "test_set.pth"
        )
    else:
        with open(current_output_path / "pre_train_hparams.json", "w", encoding="utf-8") as f:
            json.dump(pre_train_hparams, f, indent=4)
        with open(current_output_path / "opt_params.json", "w", encoding="utf-8") as f:
            json.dump(opt_params, f, indent=4)
        train_loader, test_loader = DatasetPrepare(
            hparams=pre_train_hparams).setup()

        torch.save(train_loader.dataset.tensors,
                   current_output_path / "train_set.pth")
        torch.save(test_loader.dataset.tensors,
                   current_output_path / "test_set.pth")

    if (current_output_path/"model.pth").exists():
        model.load_state_dict(
            torch.load(current_output_path/"model.pth"))
        trained_model = model.to(device)
        trained_model.eval()
    else:
        trained_model, results = training_vae(
            epochs=30 + 1,
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

    return trained_model, train_loader, test_loader
