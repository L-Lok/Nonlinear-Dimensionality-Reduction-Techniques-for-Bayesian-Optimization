import json
import os
import torch
from torch import Tensor, nn

from .metrics import TripletLossTorch

torch.set_default_dtype(torch.float64)

class Encoder(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            layers.append(nn.Softplus())  # Softplus
        self.network = nn.Sequential(*layers)

        # latent mean and variance
        self.mean_layer = nn.Linear(layer_dims[-1], layer_dims[-1])
        self.logvar_layer = nn.Linear(layer_dims[-1], layer_dims[-1])

    def forward(self, x):
        z_minus_1 = self.network(x)
        mu, logvar = self.mean_layer(z_minus_1), self.logvar_layer(z_minus_1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

            if i != len(layer_dims) - 2:
                layers.append(nn.Softplus())  # Softplus

        self.network = nn.Sequential(*layers)

    def forward(self, z):
        x = self.network(z)
        return x


class LSBOVAE(torch.nn.Module):
    """
    Latent Space Bayesian Optimization Variational Auto Encoders
    Non-linear VAE for global optimising HBC functions"""

    def __init__(self, hparams: dict):
        super().__init__()
        # set up encoder and decoder
        self.encoder = Encoder(hparams["encoder_layer_dims"])
        self.decoder = Decoder(hparams["decoder_layer_dims"])

        # Basic parameters
        self.latent_dim: int = hparams["latent_dim"]

        # Create beta
        self.beta = hparams["beta_final"]
        self.beta_final = hparams["beta_final"]
        self.beta_start = hparams["beta_start"]
        self.beta_step = hparams["beta_step"]
        self.beta_step_freq = hparams["beta_step_freq"]
        self.beta_warmup = hparams["beta_warmup"]
        self.beta_annealing = False
        if self.beta_start is not None:
            self.beta_annealing = True
            self.beta = self.beta_start
            assert (
                self.beta_step is not None
                and self.beta_step_freq is not None
                and self.beta_warmup is not None
            )

        self.beta_metric_loss = 0
        if hparams["beta_metric_loss"] is not None:
            self.beta_metric_loss = hparams["beta_metric_loss"]

        # Metric Losses
        self.metric_loss = hparams["metric_loss"]

    def sample_latent(self, mu, logvar):
        # logstd -> logvar
        # it is (likely to be) logvar
        scale_safe = torch.exp(0.5 * logvar) + 1e-10  # numerically stable
        encoder_distribution = torch.distributions.Normal(
            loc=mu, scale=scale_safe)
        z_sample = encoder_distribution.rsample()

        # or ref:
        # https://github.com/bvezilic/Variational-autoencoder/blob/master/vae/model.py
        # eps = torch.randn_like(scale_safe)
        # z_sample = scale_safe*eps + mu

        return z_sample

    def kl_loss(self, mu, logvar, z_sample, mean=False):
        # Manual formula for kl divergence (more numerically stable!)
        # ref:
        # https://github.com/bvezilic/Variational-autoencoder/blob/master/vae/loss.py
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Whether to take the average:
        # according to the github link above, it seems that
        # averaging in mse might reconstruct the same image
        if mean:
            loss = kl_div / z_sample.shape[0]
        else:
            loss = kl_div
        return loss

    def encode_to_params(self, x: Tensor):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def _decoder_loss_mse(self, z, x_orig, mean=False):
        x_recon = self.decoder(z)
        assert (
            x_orig.shape == x_recon.shape
        ), f"orig shape {x_orig.shape} and recon shape {x_recon.shape} should be the same"
        if mean:
            mse = nn.MSELoss(reduction="mean")
        else:
            mse = nn.MSELoss(reduction="sum")
        losses = mse(x_orig, x_recon)

        return losses

    def _decoder_loss_huber(self, z, x_orig, mean=False):
        x_recon = self.decoder(z)
        assert (
            x_orig.shape == x_recon.shape
        ), f"orig shape {x_orig.shape} and recon shape {x_recon.shape} should be the same"
        if mean:
            loss_fn = nn.HuberLoss(reduction="mean")
        else:
            loss_fn = nn.HuberLoss(reduction="sum")
        losses = loss_fn(x_orig, x_recon)
        return losses

    def decoder_loss(self, z, x_orig, loss_type="mse", mean=False):
        if loss_type == "mse":
            mse_loss = self._decoder_loss_mse(z=z, x_orig=x_orig, mean=mean)
            return mse_loss
        elif loss_type == "huber":
            huber_loss = self._decoder_loss_huber(
                z=z, x_orig=x_orig, mean=mean)
            return huber_loss

    def forward(self, x, y, loss_type="mean", mean=False, validation=False):

        # reparametrisation trick
        mu, logvar = self.encoder(x)
        z_sample = self.sample_latent(mu, logvar)

        # KL divergence and reconstruction error
        kl_loss = self.kl_loss(mu=mu, logvar=logvar,
                               z_sample=z_sample, mean=mean)
        reconstruction_loss = self.decoder_loss(
            z=z_sample, x_orig=x, loss_type=loss_type, mean=mean
        )

        # Final Loss
        if validation:
            beta = self.beta_final
        else:
            beta = self.beta

        metric_loss = 0

        if self.metric_loss is not None:
            if self.metric_loss["type"] == "triplet":
                triplet_loss = TripletLossTorch(
                    eta=self.metric_loss["eta"],
                    margin=self.metric_loss["margin"],
                    soft=self.metric_loss["soft"],
                    nu=self.metric_loss["nu"],
                )
                metric_loss = triplet_loss(z_sample, y)
            else:
                raise ValueError(
                    f"{self.metric_loss['type']} is not supported!")

        # total loss
        loss = (
            reconstruction_loss + beta * kl_loss + self.beta_metric_loss * metric_loss
        )

        return loss

    def increment_beta(self, global_step):
        if not self.beta_annealing:
            return

        if global_step > self.beta_warmup and global_step % self.beta_step_freq == 0:
            self.beta = min(self.beta_final, self.beta * self.beta_step)


def save_model_and_results(path, model, results):
    """
    Save the model state and results to the specified path.

    Args:
        path (str or Path): Directory to save the model and results.
        model (torch.nn.Module): The model to save.
        results (dict): The results to save.
    """
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pth"))
    with open(os.path.join(path, "results.json"), "w") as f:
        json.dump(results, f)


def training_vae(
    epochs,
    model: nn.Module,
    train_loader,
    test_loader,
    opt_params,
    loss_type: str = "mse",
    mean: bool = True,
    do_test: bool = True,
    pretraining: bool = False,  # pretraining stage
    pretrained: bool = False,  # whether pre-trained
):
    if pretraining:
        model.beta_metric_loss = 0  # Weight for metric loss component

    if pretrained:
        model.beta_start = None  # beta-annealing

    optimiser = torch.optim.Adam(model.parameters(), lr=opt_params["lr"])

    results = {"train": [], "test": []}
    for epoch in range(epochs):
        print("model.beta_metric_loss", model.beta_metric_loss)
        model.train()

        model.increment_beta(epoch)
        print("model_beta", model.beta)
        epoch_train_loss = 0
        for _batch_idx, (batch_data_x, batch_data_y) in enumerate(train_loader):
            # The gradients are set to zero
            optimiser.zero_grad()
            x = batch_data_x
            y = batch_data_y
            print(x.dtype, y.dtype)
            loss = model(x, y, loss_type=loss_type, mean=mean)

            epoch_train_loss += loss.item()

            # the gradient is computed and stored.
            # .step() performs parameter update
            loss.backward()
            optimiser.step()

        # normalise the loss
        normalizer_train = len(train_loader.dataset.tensors[0])
        total_epoch_loss_train = epoch_train_loss / normalizer_train
        results["train"].append(total_epoch_loss_train)
        print(
            "[vae epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )

        # model.eval() <- put in the if-statement
        # evaluate to test
        if do_test and epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                epoch_test_loss = 0
                for _batch_idx, (batch_data_x, batch_data_y) in enumerate(test_loader):
                    x_test = batch_data_x
                    y_test = batch_data_y
                    loss = model(x_test, y_test,
                                 loss_type=loss_type, mean=mean)

                    epoch_test_loss += loss.item()

                normalizer_test = len(test_loader.dataset.tensors[0])
                total_epoch_loss_test = epoch_test_loss / normalizer_test
                results["test"].append(total_epoch_loss_test)
                print(
                    "[vae epoch %03d]  average test loss: %.4f"
                    % (epoch, total_epoch_loss_test)
                )
    model.eval()
    return model, results


def get_pretrained_vae(
    encoder_layers=[],

):
    pass
