import logging
import time
from typing import List

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from ..gp.fit_gp_model import train_gp_plain
from ..gp.gp_model import NOISE_FREE_CONST, gp_model_unit_cube
from ..gp.optimize_acqf import optimize_acqf_with_warmup
from ..test_funcs.base import BaseTestFunction
from ..vae.data_generator import DatasetPrepare
from ..vae.pretrain_vae import pretrain_vae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_x(x, current_bounds, target_bounds):  # -> Any
    """
    Transform the input tensor from the function space to the VAE space.

    Parameters:
        x (torch.Tensor): Input tensor.
        current_bounds (torch.Tensor): Bounds of the current x space. shape dim x 2
        target_bounds (torch.Tensor): Bounds of the target space. shape dim x 2

    Returns:
        torch.Tensor: The transformed tensor.
    """
    current_bounds = current_bounds.to(x)
    target_bounds = target_bounds.to(x)
    return (x - current_bounds[:, 0]) / (current_bounds[:, 1] - current_bounds[:, 0]) * (
        target_bounds[:, 1] - target_bounds[:, 0]) + target_bounds[:, 0]


def bayesian_optimization_new_points_loop(
    latent_space_dataset_z,
    latent_space_dataset_y,
    frequency,
    vae_model,
    # parent_folder,
    SeqDomainRed,
    obj_func,
    current_bounds=None,
    train_y_var=NOISE_FREE_CONST
):
    """
    Perform Bayesian optimization to acquire new data points.

    Args:
        latent_space_dataset_z (torch.Tensor): Latent space representations.
        latent_space_dataset_y (torch.Tensor): Corresponding targets.
        frequency (int): Number of iterations to run the loop.
        vae_model (torch.nn.Module): The VAE model used for decoding.
        parent_folder (str or Path): Directory to save the model and results.

    Returns:
        tuple: New sampled points (dataset_x), latent space representations (new_points_z), and targets (new_points_fx).
    """
    gp_model = None
    dataset_x = None
    best_f = torch.max(latent_space_dataset_y)

    best_value_f_trace_current_loop = torch.empty(0).to(latent_space_dataset_y)
    for k in range(frequency):
        # fit a gp model to the latent space datset
        logging.info(f"latent_space_dataset_y  {latent_space_dataset_y.shape}")

        print("latent_space_dataset_z", latent_space_dataset_z.shape)

        start_time = time.time()
        gp_model = gp_model_unit_cube(
            latent_space_dataset_z.detach(),
            latent_space_dataset_y.detach(),
            #   we guide the gp model with previous trained covariance
            # covar_module=gp_model.covar_module if gp_model is not None else None
            train_y_var=train_y_var
        )
        train_gp_plain(gp_model)
        print("train gp takes", time.time() - start_time)

        start_time = time.time()

        new_point_z, _ = optimize_acqf_with_warmup(
            gp_model=gp_model,
            best_f=best_f,
            current_bounds=current_bounds,
        )
        print("optimize acqf takes", time.time()-start_time)
        zhat = new_point_z.detach()
        print("zhat for gp:", zhat, zhat.shape)
        start_time = time.time()

        xhat = vae_model.decoder(zhat)
        print("vae decoder takes", time.time()-start_time)

        print("new_xhat", xhat)

        f_x_hat = obj_func(xhat)
        print("f_x_hat", f_x_hat)
        # print(torch.max(latent_space_dataset_y))
        # exit()
        if dataset_x is None:
            dataset_x = torch.empty(0, xhat.shape[1], device=device)
        dataset_x = torch.cat((dataset_x, xhat), dim=0)

        latent_space_dataset_z = torch.cat(
            (latent_space_dataset_z, zhat), dim=0)

        latent_space_dataset_y = torch.cat(
            (latent_space_dataset_y, f_x_hat),
            dim=0,
        )

        # shrink bounds
        if SeqDomainRed is not None:
            current_bounds = SeqDomainRed.transform(
                latent_space_dataset_z.clone(), latent_space_dataset_y.clone()
            )

        # print("Shrinking bounds:", current_bounds)
        print(
            "latent_space_dataset_y after cat",
            latent_space_dataset_y.shape,
            latent_space_dataset_y[-1],
        )
        best_f = torch.max(latent_space_dataset_y)
        logging.info(best_f)
        best_value_f_trace_current_loop = torch.cat(
            [best_value_f_trace_current_loop, best_f.detach().view(1)])
    # return new sampled points
    return (
        dataset_x,
        latent_space_dataset_z[-frequency:],
        latent_space_dataset_y[-frequency:],
        best_value_f_trace_current_loop
    )


def get_latent_space_dataset(model, data_loader) -> torch.Tensor:
    """
    Generate the latent space dataset from the VAE model.

    Args:
        model (torch.nn.Module): The VAE model used to generate the latent space.
        data_loader (DataLoader): DataLoader for the dataset.

    Returns:
        torch.Tensor: Latent space representations (z) and corresponding targets (y).
    """
    # initialize empty tensor
    for data in data_loader:
        x, y = data
        mu, logvar = model.encoder(x)
        break

    latent_space_dataset_z_mu = torch.empty(0, mu.shape[1], device=device)
    latent_space_dataset_z_logvar = torch.empty(
        0, logvar.shape[1], device=device)
    latent_space_dataset_y = torch.empty(0, y.shape[1], device=device)

    for data in data_loader:
        x, y = data
        mu, logvar = model.encoder(x)
        latent_space_dataset_z_mu = torch.cat(
            (latent_space_dataset_z_mu, mu), dim=0)
        latent_space_dataset_z_logvar = torch.cat(
            (latent_space_dataset_z_logvar, logvar), dim=0)
        latent_space_dataset_y = torch.cat((latent_space_dataset_y, y), dim=0)

    return latent_space_dataset_z_mu, latent_space_dataset_z_logvar, latent_space_dataset_y


def vae_bo_base_pipeline(
    test_func: BaseTestFunction,
    encoder_layers: List[int] = [5, 3, 2],
    n_initial_points: int = 50,
    n_trails: int = 350,
    train_y_var=NOISE_FREE_CONST
):
    assert test_func.dim == encoder_layers[0], "test function dim must be equal to the dim of the first encoder layer"
    feature_dim = test_func.dim
    vae_base_model, pretrain_vae_train_loader, pretrain_vae_test_loader = pretrain_vae(
        encoder_layers
    )

    vae_input_bounds = torch.tensor([
        [-3, 3]] * feature_dim)

    print("vae_input_bounds", vae_input_bounds.shape)
    print("test_func_bounds", test_func.bounds.shape)
    x_train_in_vae_space = pretrain_vae_train_loader.dataset.tensors[0].to(
        device)
    x_test = pretrain_vae_test_loader.dataset.tensors[0].to(device)

    # we now transform the x_train in the vae space to the function space
    x_train = transform_x(
        x_train_in_vae_space, vae_input_bounds, test_func.bounds
    )

    # obtain the function value we want to optimize
    y_train = test_func.func(x_train)
    y_test = test_func.func(x_test)
    indices = torch.randperm(x_train.size(0))[:n_initial_points]
    x_train = x_train[indices]
    y_train = y_train[indices]

    train_loader, test_loader = DatasetPrepare(
        hparams={
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
        }).load_tensor(x_train, y_train, x_test, y_test, drop_last=False)
    # general number of vae reruns.
    new_xhat_points = None
    new_z_points = None
    new_f_points = None
    # vae_model = copy.deepcopy(vae_base_model)
    current_trained_vae_model = vae_base_model
    best_value_f_trace = torch.empty(0)

    latent_bounds = torch.tensor([[-5., 5.]] * encoder_layers[-1]).to(device)

    for vae_loop_i in range(1):

        latent_space_dataset_z_mu, latent_space_dataset_z_logvar, latent_space_dataset_y = get_latent_space_dataset(
            current_trained_vae_model, train_loader)


        new_points_x, new_points_z, new_points_fx, best_value_f_trace_current_loop = (
            bayesian_optimization_new_points_loop(
                latent_space_dataset_z=latent_space_dataset_z_mu,
                latent_space_dataset_y=latent_space_dataset_y,
                frequency=n_trails,
                vae_model=current_trained_vae_model,
                SeqDomainRed=None,
                current_bounds=latent_bounds,
                # use lambda function for bounds transformation
                obj_func=lambda x: test_func.func(transform_x(
                    x, vae_input_bounds, test_func.bounds
                )
                ),
                train_y_var=train_y_var
            )
        )
        print("new_points_x", transform_x(
            new_points_x, vae_input_bounds, test_func.bounds
        ))
        print("new_points_z", new_points_z)

        merged_train_dataset_tensor_x = torch.cat(
            (
                train_loader.dataset.tensors[0].to(device),
                new_points_x.detach().to(device),
            ),
            dim=0,
        )
        merged_train_dataset_tensor_y = torch.cat(
            (
                train_loader.dataset.tensors[1].to(device),
                new_points_fx.detach().to(device),
            ),
            dim=0,
        )

        # Possible Treatments to Datasets

        # sort by the lowest function value
        # idx = torch.argsort(merged_train_dataset_tensor_y, dim=0).flatten()
        # merged_train_dataset_tensor_x = merged_train_dataset_tensor_x[idx]
        # merged_train_dataset_tensor_y = merged_train_dataset_tensor_y[idx]

        print("merged_train_dataset_tensor_y max",
              merged_train_dataset_tensor_y.max())

        # delete last `frequency`` of the dataset
        # b/c argsort is in ascending order
        # and our values are -f
        # we crop the first `frequency`` of the dataset
        # merged_train_dataset_tensor_x = merged_train_dataset_tensor_x[frequency:]
        # merged_train_dataset_tensor_y = merged_train_dataset_tensor_y[frequency:]

        # Update train_dataset
        merged_train_dataset = TensorDataset(
            merged_train_dataset_tensor_x, merged_train_dataset_tensor_y
        )
        print(merged_train_dataset.tensors[0].shape)
        print(merged_train_dataset.tensors[1].shape)

        # Update train_loader
        train_loader = DataLoader(
            merged_train_dataset,
            batch_size=256
        )

        # Saving Results
        if new_xhat_points is None:
            new_xhat_points = torch.empty(0, new_points_x.shape[1])
        new_xhat_points = torch.cat(
            (new_xhat_points, transform_x(
                new_points_x, vae_input_bounds, test_func.bounds
            ).detach().cpu()), dim=0
        )


        if new_z_points is None:
            new_z_points = torch.empty(0, new_points_z.shape[1])
        new_z_points = torch.cat(
            (new_z_points, new_points_z.detach().cpu()), dim=0)

        if new_f_points is None:
            new_f_points = torch.empty(0, new_points_fx.shape[1])
        new_f_points = torch.cat(
            (new_f_points, new_points_fx.detach().cpu()), dim=0)

        best_value_f_trace = torch.cat(
            [best_value_f_trace, best_value_f_trace_current_loop.detach().cpu()])

    return {
        "init_train_x": x_train.detach().cpu(),
        "init_train_y": y_train.detach().cpu(),
        "new_xhat_points": new_xhat_points.detach().cpu().numpy(),
        "new_z_points": new_z_points.detach().cpu().numpy(),
        "new_f_points": new_f_points.detach().cpu().numpy(),
        "best_value_f_trace": best_value_f_trace.detach().cpu().numpy()
    }


def vae_bo_pipeline(
    test_func: BaseTestFunction,
    encoder_layers: List[int] = [5, 3, 2],
    n_initial_points: int = 50,
    n_trails: int = 350,
    n_repeats: int = 5,
    train_y_var=NOISE_FREE_CONST
):
    results = []
    for i in range(n_repeats):
        results.append(
            vae_bo_base_pipeline(
                test_func=test_func,
                encoder_layers=encoder_layers,
                n_initial_points=n_initial_points,
                n_trails=n_trails,
                train_y_var=train_y_var
            )
        )
    return results
