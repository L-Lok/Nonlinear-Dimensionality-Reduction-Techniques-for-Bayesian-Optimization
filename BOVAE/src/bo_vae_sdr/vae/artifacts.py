"""VAE pretraining, loading, and manuscript retraining utilities.

The public retraining objective contains reconstruction loss, KL
regularization, and (for the DML method) the manuscript triplet loss.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from botorch.utils.probability import TruncatedMultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

from ..pipeline.transforms import BoxTransform, tensor_to_list
from .dml import (
    TargetNormalizer,
    dml_triplet_stats,
    manuscript_triplet_config,
    normalize_dml_targets,
    resolve_dml_threshold,
)
from .metrics import TripletLossTorch
from .model import LSBOVAE


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default))
        handle.write("\n")


@dataclass
class VAETrainingConfig:
    vae_id: str
    ambient_dim: int
    latent_dim: int
    encoder_layer_dims: list[int]
    decoder_layer_dims: list[int]
    activation: str = "SiLU"
    decoder_output_activation: str = "linear"
    num_samples: int = 10000
    validation_fraction: float = 0.10
    epochs: int = 150
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    beta_start: float = 0.0
    beta_final: float = 1.0
    beta_step_freq: int = 10
    beta_step: float = 0.1
    beta_warmup: int = 0
    checkpoint_selection: str = "best_validation_after_min_beta"
    checkpoint_min_beta: float | None = None
    data_distribution: str = "truncated_high_correlation_normal"
    correlation_value: float = 0.9
    train_bounds_lower: float = -5.0
    train_bounds_upper: float = 5.0
    seed: int = 0
    dtype: str = "float64"
    source: str = "NDRT revised manuscript Table 1 and Table C5"

    @classmethod
    def manuscript_d10(cls, latent_dim: int, *, seed: int = 0) -> "VAETrainingConfig":
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if latent_dim == 2:
            return cls(
                vae_id="VAE-4.2_D10_d2",
                ambient_dim=10,
                latent_dim=2,
                encoder_layer_dims=[10, 5, 2],
                decoder_layer_dims=[2, 5, 10],
                seed=seed,
            )
        if latent_dim == 5:
            return cls(
                vae_id="VAE-4.1_D10_d5",
                ambient_dim=10,
                latent_dim=5,
                encoder_layer_dims=[10, 5],
                decoder_layer_dims=[5, 10],
                seed=seed,
            )
        return cls(
            vae_id=f"VAE-D10-d{int(latent_dim)}",
            ambient_dim=10,
            latent_dim=int(latent_dim),
            encoder_layer_dims=[10, int(latent_dim)],
            decoder_layer_dims=[int(latent_dim), 10],
            seed=seed,
            source="D10 latent-dimension sweep generic baseline",
        )

    @classmethod
    def from_json(cls, path: Path) -> "VAETrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        public_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in public_fields})

    def dtype_obj(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype {self.dtype}")

    def train_box(self) -> BoxTransform:
        bounds = torch.tensor(
            [[self.train_bounds_lower, self.train_bounds_upper]] * self.ambient_dim,
            dtype=self.dtype_obj(),
        )
        return BoxTransform(bounds=bounds, name="bovae_vae_input")

    def model_hparams(self) -> dict[str, Any]:
        return {
            "beta_start": None,
            "beta_final": self.beta_final,
            "beta_step": self.beta_step,
            "beta_step_freq": self.beta_step_freq,
            "beta_warmup": self.beta_warmup,
            "activation": self.activation,
            "decoder_output_activation": self.decoder_output_activation,
            "decoder_output_lower": [self.train_bounds_lower] * self.ambient_dim,
            "decoder_output_upper": [self.train_bounds_upper] * self.ambient_dim,
            "latent_dim": self.latent_dim,
            "encoder_layer_dims": list(self.encoder_layer_dims),
            "decoder_layer_dims": list(self.decoder_layer_dims),
        }


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def _select_device(device: torch.device | str | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def beta_for_epoch(config: VAETrainingConfig, epoch: int, *, anneal: bool = True) -> float:
    if not anneal:
        return float(config.beta_final)
    increments = max(epoch // max(config.beta_step_freq, 1), 0)
    return float(min(float(config.beta_start) + increments * float(config.beta_step), config.beta_final))


def checkpoint_min_beta(config: VAETrainingConfig) -> float:
    return float(config.beta_final if config.checkpoint_min_beta is None else config.checkpoint_min_beta)


def checkpoint_is_selectable(config: VAETrainingConfig, beta: float) -> bool:
    if config.checkpoint_selection == "best_validation":
        return True
    if config.checkpoint_selection == "best_validation_after_min_beta":
        return float(beta) >= checkpoint_min_beta(config)
    if config.checkpoint_selection == "final":
        return False
    raise ValueError(
        "checkpoint_selection must be one of "
        "{'best_validation', 'best_validation_after_min_beta', 'final'}"
    )


def high_correlation_cov(dim: int, correlation_value: float, dtype: torch.dtype) -> torch.Tensor:
    if not -1.0 < correlation_value < 1.0:
        raise ValueError("correlation_value must be in (-1, 1)")
    cov = torch.full((dim, dim), float(correlation_value), dtype=dtype)
    cov.fill_diagonal_(1.0)
    return cov + 1e-4 * torch.eye(dim, dtype=dtype)


def generate_pretraining_data(config: VAETrainingConfig) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    torch.manual_seed(int(config.seed))
    dtype = config.dtype_obj()
    dim = int(config.ambient_dim)
    bounds = config.train_box().bounds.to(dtype=dtype)
    distribution = str(config.data_distribution)
    if distribution == "truncated_high_correlation_normal":
        mvn = TruncatedMultivariateNormal(
            loc=torch.zeros(dim, dtype=dtype),
            covariance_matrix=high_correlation_cov(dim, config.correlation_value, dtype),
            bounds=bounds,
        )
        samples = mvn.sample(torch.Size([int(config.num_samples)])).to(dtype=dtype)
    elif distribution == "uniform_box":
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        unit = torch.rand((int(config.num_samples), dim), dtype=dtype)
        samples = lower + unit * (upper - lower)
    else:
        raise ValueError(
            "data_distribution must be one of "
            "{'truncated_high_correlation_normal', 'uniform_box'}"
        )
    split = int(round((1.0 - float(config.validation_fraction)) * config.num_samples))
    split = min(max(split, 1), config.num_samples - 1)
    train_x = samples[:split].contiguous()
    val_x = samples[split:].contiguous()
    metadata = {
        "distribution": distribution,
        "correlation_value": config.correlation_value,
        "bounds": tensor_to_list(bounds),
        "num_samples": config.num_samples,
        "train_size": int(train_x.shape[0]),
        "validation_size": int(val_x.shape[0]),
        "seed": config.seed,
    }
    return train_x, val_x, metadata


def make_model(config: VAETrainingConfig) -> LSBOVAE:
    torch.set_default_dtype(config.dtype_obj())
    return LSBOVAE(config.model_hparams())


def _vae_loss_terms(
    model: LSBOVAE,
    x: torch.Tensor,
    *,
    beta: float,
    y_metric: torch.Tensor | None = None,
    dml_triplet_config: dict[str, Any] | None = None,
    dml_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mu, logvar = model.encoder(x)
    z = model.sample_latent(mu, logvar)
    reconstruction = torch.mean(torch.sum((model.decoder(z) - x).pow(2), dim=1))
    kl = torch.mean(-0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    triplet = torch.zeros((), dtype=x.dtype, device=x.device)
    if dml_triplet_config is not None and y_metric is not None and dml_loss_weight > 0.0:
        stats = dml_triplet_stats(y_metric.detach(), float(dml_triplet_config["threshold"]))
        if stats["valid_triplets"] > 0:
            triplet = TripletLossTorch(
                threshold=float(dml_triplet_config["threshold"]),
                eta=float(dml_triplet_config.get("eta", 0.2)),
            )(z, y_metric)
    loss = reconstruction + float(beta) * kl + float(dml_loss_weight) * triplet
    terms = {
        "loss": float(loss.detach().item()),
        "reconstruction_loss": float(reconstruction.detach().item()),
        "kl_loss": float(kl.detach().item()),
    }
    if dml_triplet_config is not None:
        terms["dml_triplet_loss"] = float(triplet.detach().item())
    return loss, terms


def _evaluate_vae(
    model: LSBOVAE,
    loader: DataLoader,
    *,
    beta: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("loss", "reconstruction_loss", "kl_loss")}
    count = 0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            _, terms = _vae_loss_terms(model, x, beta=beta)
            count += x.shape[0]
            for key in totals:
                totals[key] += terms[key] * x.shape[0]
    if count == 0:
        raise ValueError("validation loader is empty")
    return {key: value / count for key, value in totals.items()}


def train_vae(
    config: VAETrainingConfig,
    output_dir: Path,
    *,
    device: torch.device | str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass overwrite=True")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "training_trace.jsonl"
    validation_path = output_dir / "validation_trace.jsonl"
    if overwrite:
        trace_path.unlink(missing_ok=True)
        validation_path.unlink(missing_ok=True)
    selected_device = _select_device(device)
    dtype = config.dtype_obj()
    torch.manual_seed(config.seed)
    if selected_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_x, val_x, data_metadata = generate_pretraining_data(config)
    torch.save((train_x, torch.ones(train_x.shape[0], 1, dtype=dtype)), output_dir / "train_data.pt")
    torch.save((val_x, torch.ones(val_x.shape[0], 1, dtype=dtype)), output_dir / "validation_data.pt")
    write_json(output_dir / "vae_config.json", asdict(config))
    write_json(output_dir / "run_metadata.json", {
        "artifact_type": "bovae_vae_pretraining",
        "config": asdict(config),
        "environment": environment_metadata(selected_device),
        "data": data_metadata,
    })
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    train_loader = DataLoader(
        TensorDataset(train_x),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(TensorDataset(val_x), batch_size=config.batch_size)
    model = make_model(config).to(device=selected_device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_any = (float("inf"), -1)
    selected: tuple[float, int, float | None, dict[str, torch.Tensor] | None] = (
        float("inf"), -1, None, None
    )
    final_state: dict[str, torch.Tensor] | None = None
    final_validation = float("inf")
    final_beta = config.beta_final
    for epoch in range(config.epochs):
        model.train()
        beta = beta_for_epoch(config, epoch)
        totals = {key: 0.0 for key in ("loss", "reconstruction_loss", "kl_loss")}
        seen = 0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device=selected_device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = _vae_loss_terms(model, x_batch, beta=beta)
            loss.backward()
            optimizer.step()
            seen += x_batch.shape[0]
            for key in totals:
                totals[key] += terms[key] * x_batch.shape[0]
        append_jsonl(trace_path, {
            "epoch": epoch,
            "beta": beta,
            **{key: value / seen for key, value in totals.items()},
        })
        validation = _evaluate_vae(model, val_loader, beta=beta, device=selected_device)
        append_jsonl(validation_path, {"epoch": epoch, "beta": beta, **validation})
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if validation["loss"] < best_any[0]:
            best_any = (validation["loss"], epoch)
            torch.save(state, output_dir / "best_any_model_state_dict.pt")
        if checkpoint_is_selectable(config, beta) and validation["loss"] < selected[0]:
            selected = (validation["loss"], epoch, beta, state)
            torch.save(state, output_dir / "best_model_state_dict.pt")
        final_state, final_validation, final_beta = state, validation["loss"], beta
    if final_state is None:
        raise RuntimeError("VAE training did not produce a checkpoint")
    if config.checkpoint_selection == "final":
        selected = (final_validation, config.epochs - 1, final_beta, final_state)
    if selected[3] is None:
        raise RuntimeError("VAE training did not reach the configured checkpoint-selection beta")
    torch.save(selected[3], output_dir / "model_state_dict.pt")
    torch.save(final_state, output_dir / "final_model_state_dict.pt")
    validation = validate_vae_checkpoint(output_dir, device=selected_device)
    summary = {
        "artifact_type": "bovae_vae_pretraining_summary",
        "output_dir": str(output_dir),
        "selected_epoch": selected[1],
        "selected_beta": selected[2],
        "selected_validation_loss": selected[0],
        "unconstrained_best_epoch": best_any[1],
        "unconstrained_best_validation_loss": best_any[0],
        "validation": validation,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "reconstruction_summary.json", validation)
    return summary


def resolve_vae_state_path(checkpoint_dir: Path, state_file: str = "model_state_dict.pt") -> Path:
    state_name = Path(str(state_file))
    if state_name.is_absolute() or state_name.name != str(state_file) or str(state_file) in {"", ".", ".."}:
        raise ValueError("vae_state_file must be a file name inside the checkpoint directory")
    return Path(checkpoint_dir) / state_name


def load_pretrained_vae(
    checkpoint_dir: Path,
    *,
    device: torch.device | str | None = None,
    state_file: str = "model_state_dict.pt",
) -> LSBOVAE:
    checkpoint_dir = Path(checkpoint_dir)
    config = VAETrainingConfig.from_json(checkpoint_dir / "vae_config.json")
    selected_device = _select_device(device)
    model = make_model(config).to(device=selected_device, dtype=config.dtype_obj())
    state_path = resolve_vae_state_path(checkpoint_dir, state_file)
    if not state_path.exists():
        raise FileNotFoundError(f"Missing BO-VAE VAE state dict: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location=selected_device))
    model.eval()
    return model


def validate_vae_checkpoint(
    checkpoint_dir: Path,
    *,
    device: torch.device | str | None = None,
    state_file: str = "model_state_dict.pt",
) -> dict[str, Any]:
    config = VAETrainingConfig.from_json(Path(checkpoint_dir) / "vae_config.json")
    model = load_pretrained_vae(checkpoint_dir, device=device, state_file=state_file)
    selected_device = next(model.parameters()).device
    x = torch.zeros(4, config.ambient_dim, dtype=config.dtype_obj(), device=selected_device)
    with torch.no_grad():
        mu, logvar = model.encoder(x)
        decoded = model.decoder(mu)
    return {
        "ambient_dim": config.ambient_dim,
        "latent_dim": config.latent_dim,
        "state_file": state_file,
        "state_path": str(resolve_vae_state_path(checkpoint_dir, state_file)),
        "encoder_mu_shape": list(mu.shape),
        "encoder_logvar_shape": list(logvar.shape),
        "decoder_output_shape": list(decoded.shape),
        "reconstruction_mse_at_zero": float(torch.mean((decoded - x).pow(2)).item()),
        "device": str(selected_device),
        "dtype": str(config.dtype_obj()),
    }


def retrain_vae_on_labeled_data(
    model: LSBOVAE,
    x_vae: torch.Tensor,
    y_original: torch.Tensor,
    *,
    config: VAETrainingConfig,
    output_dir: Path,
    stage_index: int,
    epochs: int = 2,
    batch_size: int = 128,
    use_dml: bool = False,
    dml_threshold: float = 0.01,
    dml_eta: float = 0.2,
    dml_loss_weight: float = 1.0,
    learning_rate: float | None = None,
    weight_decay: float = 0.0,
) -> dict[str, Any]:
    """Retrain with the standard VAE loss and optional normalized DML term."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x_vae = x_vae.detach().to(device=device, dtype=dtype)
    y_original = y_original.detach().reshape(-1, 1).to(device=device, dtype=dtype)
    target_normalizer: TargetNormalizer | None = None
    y_metric: torch.Tensor | None = None
    dml_triplet_config: dict[str, Any] | None = None
    threshold_metadata: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None
    if use_dml:
        y_metric, target_normalizer = normalize_dml_targets(y_original)
        effective_threshold, threshold_metadata = resolve_dml_threshold(
            y_metric,
            dml_threshold,
        )
        dml_triplet_config = manuscript_triplet_config(effective_threshold, dml_eta)
        stats = dml_triplet_stats(y_metric, effective_threshold)
    dataset = TensorDataset(x_vae, y_metric) if y_metric is not None else TensorDataset(x_vae)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + stage_index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    effective_learning_rate = config.learning_rate if learning_rate is None else learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=effective_learning_rate,
        weight_decay=weight_decay,
    )
    trace_path = output_dir / "vae_retraining_trace.jsonl"
    beta = float(config.beta_final)
    for epoch in range(epochs):
        model.train()
        term_names = ["loss", "reconstruction_loss", "kl_loss"]
        if use_dml:
            term_names.append("dml_triplet_loss")
        totals = {key: 0.0 for key in term_names}
        seen = 0
        for batch_values in loader:
            x_batch = batch_values[0].to(device=device, dtype=dtype)
            y_batch = batch_values[1].to(device=device, dtype=dtype) if len(batch_values) == 2 else None
            optimizer.zero_grad(set_to_none=True)
            loss, terms = _vae_loss_terms(
                model,
                x_batch,
                beta=beta,
                y_metric=y_batch,
                dml_triplet_config=dml_triplet_config,
                dml_loss_weight=dml_loss_weight if use_dml else 0.0,
            )
            loss.backward()
            optimizer.step()
            seen += x_batch.shape[0]
            for key in totals:
                totals[key] += terms[key] * x_batch.shape[0]
        append_jsonl(trace_path, {
            "stage": stage_index,
            "epoch": epoch,
            "use_dml": use_dml,
            "beta": beta,
            **{key: value / max(seen, 1) for key, value in totals.items()},
        })
    checkpoint_path = output_dir / f"vae_retrain_stage_{stage_index:03d}.pt"
    torch.save({
        "stage": stage_index,
        "epochs": epochs,
        "use_dml": use_dml,
        "learning_rate": effective_learning_rate,
        "weight_decay": weight_decay,
        "model_state_dict": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    model.eval()
    return {
        "stage": stage_index,
        "epochs": epochs,
        "use_dml": use_dml,
        "triplet_stats": stats,
        "dml_threshold": threshold_metadata,
        "target_normalizer": target_normalizer.metadata() if target_normalizer else None,
        "learning_rate": effective_learning_rate,
        "weight_decay": weight_decay,
        "checkpoint": str(checkpoint_path),
    }
