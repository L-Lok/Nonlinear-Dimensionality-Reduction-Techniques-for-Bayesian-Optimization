"""Neural-network definition for the publication BO-VAE models."""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn


def _activation_module(name: str) -> nn.Module:
    if str(name).lower() == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported hidden activation {name!r}; use 'SiLU'")


class Encoder(nn.Module):
    """Feed-forward encoder returning the latent mean and log variance."""

    def __init__(self, layer_dims: Sequence[int], activation: str = "SiLU") -> None:
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("encoder_layer_dims must contain input and latent dimensions")
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.extend(
                [nn.Linear(int(input_dim), int(output_dim)), _activation_module(activation)]
            )
        self.network = nn.Sequential(*layers)
        latent_dim = int(layer_dims[-1])
        self.mean_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.network(x)
        return self.mean_layer(hidden), self.logvar_layer(hidden)


class Decoder(nn.Module):
    """Feed-forward decoder with the manuscript output activations."""

    def __init__(
        self,
        layer_dims: Sequence[int],
        activation: str = "SiLU",
        output_activation: str = "linear",
        output_lower: float | list[float] | None = None,
        output_upper: float | list[float] | None = None,
    ) -> None:
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("decoder_layer_dims must contain latent and output dimensions")
        layers: list[nn.Module] = []
        last_layer_index = len(layer_dims) - 2
        for index, (input_dim, output_dim) in enumerate(
            zip(layer_dims[:-1], layer_dims[1:])
        ):
            layers.append(nn.Linear(int(input_dim), int(output_dim)))
            if index != last_layer_index:
                layers.append(_activation_module(activation))
        self.network = nn.Sequential(*layers)
        self.output_activation = str(output_activation).lower()
        if self.output_activation == "scaled_tanh":
            if output_lower is None or output_upper is None:
                raise ValueError(
                    "scaled_tanh decoder output requires output_lower and output_upper"
                )
            output_dim = int(layer_dims[-1])
            lower = torch.as_tensor(output_lower, dtype=torch.get_default_dtype())
            upper = torch.as_tensor(output_upper, dtype=torch.get_default_dtype())
            if lower.ndim == 0:
                lower = lower.repeat(output_dim)
            if upper.ndim == 0:
                upper = upper.repeat(output_dim)
            if lower.shape != (output_dim,) or upper.shape != (output_dim,):
                raise ValueError(
                    "decoder output bounds must be scalar or output-dimensional"
                )
            if torch.any(upper <= lower):
                raise ValueError("decoder output upper bounds must exceed lower bounds")
            self.register_buffer("output_lower", lower)
            self.register_buffer("output_upper", upper)
        elif self.output_activation != "linear":
            raise ValueError(
                f"Unsupported decoder output activation {output_activation!r}; "
                "use 'linear' or 'scaled_tanh'"
            )

    def forward(self, z: Tensor) -> Tensor:
        decoded = self.network(z)
        if self.output_activation == "linear":
            return decoded
        unit = (torch.tanh(decoded) + 1.0) / 2.0
        return self.output_lower.to(decoded) + unit * (
            self.output_upper.to(decoded) - self.output_lower.to(decoded)
        )


class LSBOVAE(nn.Module):
    """Variational autoencoder used by the BO-VAE publication pipelines.

    Its native objective is mean-squared reconstruction plus beta-weighted KL
    divergence. The manuscript DML pipeline adds its triplet term explicitly
    in :mod:`bo_vae_sdr.vae.artifacts`; the model itself has no hidden or
    alternative loss branches.
    """

    def __init__(self, hparams: dict[str, Any]) -> None:
        super().__init__()
        activation = str(hparams.get("activation", "SiLU"))
        self.encoder = Encoder(hparams["encoder_layer_dims"], activation=activation)
        self.decoder = Decoder(
            hparams["decoder_layer_dims"],
            activation=activation,
            output_activation=hparams.get("decoder_output_activation", "linear"),
            output_lower=hparams.get("decoder_output_lower"),
            output_upper=hparams.get("decoder_output_upper"),
        )
        self.latent_dim = int(hparams["latent_dim"])
        self.beta_final = float(hparams["beta_final"])
        self.beta_start = hparams.get("beta_start")
        self.beta_step = hparams.get("beta_step")
        self.beta_step_freq = hparams.get("beta_step_freq")
        self.beta_warmup = hparams.get("beta_warmup")
        self.beta_annealing = self.beta_start is not None
        self.beta = float(self.beta_start) if self.beta_annealing else self.beta_final
        if self.beta_annealing and any(
            value is None
            for value in (self.beta_step, self.beta_step_freq, self.beta_warmup)
        ):
            raise ValueError("beta annealing requires step, frequency, and warmup")

    @staticmethod
    def sample_latent(mu: Tensor, logvar: Tensor) -> Tensor:
        scale = torch.exp(0.5 * logvar) + 1e-10
        return torch.distributions.Normal(loc=mu, scale=scale).rsample()

    @staticmethod
    def kl_loss(mu: Tensor, logvar: Tensor, *, mean: bool = True) -> Tensor:
        per_point = -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1
        )
        return per_point.mean() if mean else per_point.sum()

    def reconstruction_loss(self, z: Tensor, x: Tensor, *, mean: bool = True) -> Tensor:
        reconstructed = self.decoder(z)
        if x.shape != reconstructed.shape:
            raise ValueError(
                f"input shape {tuple(x.shape)} and reconstruction shape "
                f"{tuple(reconstructed.shape)} must match"
            )
        per_point = torch.sum((reconstructed - x).pow(2), dim=1)
        return per_point.mean() if mean else per_point.sum()

    def forward(
        self,
        x: Tensor,
        *,
        beta: float | None = None,
        mean: bool = True,
        validation: bool = False,
    ) -> Tensor:
        mu, logvar = self.encoder(x)
        z = self.sample_latent(mu, logvar)
        effective_beta = self.beta_final if validation else self.beta
        if beta is not None:
            effective_beta = float(beta)
        return self.reconstruction_loss(z, x, mean=mean) + effective_beta * self.kl_loss(
            mu, logvar, mean=mean
        )

    def increment_beta(self, global_step: int) -> None:
        if not self.beta_annealing:
            return
        if (
            global_step > int(self.beta_warmup)
            and global_step % int(self.beta_step_freq) == 0
        ):
            self.beta = min(self.beta_final, self.beta * float(self.beta_step))


__all__ = ["Decoder", "Encoder", "LSBOVAE"]
