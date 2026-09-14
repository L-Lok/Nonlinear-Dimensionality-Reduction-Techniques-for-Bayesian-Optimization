"""Configuration objects for the public BO-VAE pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from benchmarks.curved_preimage import is_curved_preimage_key
from ..vae.artifacts import resolve_vae_state_path
from .core import PIPELINE_MODES, RETRAIN_ACCEPTANCE_POLICIES


_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class PipelineConfig:
    mode: str
    vae_checkpoint: str
    output_dir: str
    vae_state_file: str = "model_state_dict.pt"
    problem: str = "ackley"
    dim: int = 10
    latent_dim: int = 2
    seed: int = 0
    budget: int = 350
    initial_points: int = 100

    retrain_start_iteration: int = 0
    retrain_min_points: int = 0
    retrain_period: int = 50
    retrain_schedule_kind: str = "fixed"
    retrain_adaptive_min_gap: int = 10
    retrain_adaptive_max_gap: int = 50
    retrain_adaptive_feedback_incumbent_threshold: float = 0.05
    retrain_epochs: int = 2
    retrain_batch_size: int = 128
    retrain_learning_rate: float | None = None
    retrain_weight_decay: float = 0.0
    retrain_acceptance_policy: str = "always_accept"
    retrain_acceptance_loss_rel_tol: float = 0.05
    retrain_acceptance_loss_abs_tol: float = 1e-8
    retrain_acceptance_reconstruction_rel_tol: float = 0.25
    retrain_acceptance_reconstruction_abs_tol: float = 1e-8
    retrain_acceptance_kl_rel_tol: float = 1.0
    retrain_acceptance_kl_abs_tol: float = 1e-8
    retrain_acceptance_check_anchor: bool = True

    sdr_method: str = "original"
    sdr_period: int = 10
    original_sdr_minimum_window_unit: float = 0.05

    acquisition: str = "logei"
    gp_target_transform: str = "adaptive_rank_gaussian"
    gp_kernel: str = "matern52"
    gp_fit_strategy: str = "botorch_default"
    gp_duplicate_handling: str = "none"
    gp_duplicate_tol: float = 1e-10
    gp_matern_use_scale_kernel: bool = True
    gp_lengthscale_lower_bound: float = 1e-3
    gp_lengthscale_upper_bound: float = 2.0
    gp_outputscale_lower_bound: float = 1e-4
    gp_outputscale_upper_bound: float = 1e4
    gp_train_y_var_floor: float = 1e-6
    gp_torch_fallback_steps: int = 75
    gp_torch_fallback_lr: float = 0.05
    gp_maxiter: int = 50
    train_y_var: float = 1e-6

    acq_warmup: int = 1024
    acq_raw_samples: int = 256
    acq_num_restarts: int = 16
    acq_maxiter: int = 60
    acq_timeout_sec: float = 2.0

    vae_input_lower: float = -5.0
    vae_input_upper: float = 5.0
    problem_bound_lower: float | None = None
    problem_bound_upper: float | None = None
    latent_transform_kind: str = "identity"
    latent_transform_anchor_path: str | None = None
    latent_transform_coverage: float = 0.995
    latent_transform_epsilon: float = 1e-6
    latent_transform_estimator: str = "mean_std"

    dml_threshold: float = 0.01
    dml_eta: float = 0.2
    beta_metric_loss: float = 1.0

    device: str = "auto"
    dtype: str = "float64"
    resume: bool = False
    overwrite: bool = False
    initial_design_path: str | None = None
    initial_design_trust_verified_objectives: bool = False
    curved_preimage_artifact_path: str | None = None
    curved_preimage_metadata_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in PIPELINE_MODES:
            raise ValueError(f"mode must be one of {sorted(PIPELINE_MODES)}")
        if self.sdr_method not in {"none", "original"}:
            raise ValueError("sdr_method must be 'none' or 'original'")
        no_sdr_modes = {"ambient_bo", "fixed_no_sdr", "retrain_dml"}
        if self.mode in no_sdr_modes and self.sdr_method != "none":
            raise ValueError(f"{self.mode} requires sdr_method='none'")
        if self.mode not in no_sdr_modes and self.sdr_method != "original":
            raise ValueError(f"{self.mode} requires sdr_method='original'")
        if self.budget <= 0 or self.initial_points <= 1:
            raise ValueError("budget must be positive and initial_points must exceed one")
        if self.retrain_period <= 0 or self.retrain_epochs <= 0 or self.retrain_batch_size <= 0:
            raise ValueError("retraining period, epochs, and batch size must be positive")
        if self.retrain_schedule_kind not in {"fixed", "acceptance_feedback"}:
            raise ValueError(
                "retrain_schedule_kind must be 'fixed' or 'acceptance_feedback'"
            )
        if self.retrain_adaptive_min_gap <= 0:
            raise ValueError("retrain_adaptive_min_gap must be positive")
        if self.retrain_adaptive_max_gap < self.retrain_adaptive_min_gap:
            raise ValueError("retrain_adaptive_max_gap must not be smaller than the minimum")
        if self.retrain_acceptance_policy not in RETRAIN_ACCEPTANCE_POLICIES:
            raise ValueError("unsupported retrain_acceptance_policy")
        for name in (
            "retrain_acceptance_loss_rel_tol",
            "retrain_acceptance_loss_abs_tol",
            "retrain_acceptance_reconstruction_rel_tol",
            "retrain_acceptance_reconstruction_abs_tol",
            "retrain_acceptance_kl_rel_tol",
            "retrain_acceptance_kl_abs_tol",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.dim <= 0 or self.latent_dim <= 0:
            raise ValueError("dim and latent_dim must be positive")
        if self.latent_transform_kind not in {"identity", "diagonal", "whitened"}:
            raise ValueError(
                "latent_transform_kind must be 'identity', 'diagonal', or 'whitened'"
            )
        if self.gp_kernel != "matern52":
            raise ValueError("the publication BO-VAE pipeline uses gp_kernel='matern52'")
        if self.acquisition not in {"ei", "logei"}:
            raise ValueError("acquisition must be 'ei' or 'logei'")
        if not 0.0 < float(self.dml_threshold) < 1.0:
            raise ValueError("dml_threshold must be in (0, 1)")
        if float(self.dml_eta) <= 0.0 or float(self.beta_metric_loss) < 0.0:
            raise ValueError("invalid DML parameters")
        self.dtype_obj()
        self.selected_device()
        if not self.mode.startswith("ambient_"):
            resolve_vae_state_path(Path(self.vae_checkpoint), self.vae_state_file)
            if is_curved_preimage_key(self.problem):
                if self.curved_preimage_artifact_path is None:
                    raise ValueError("curved_preimage_artifact_path is required")
                if self.curved_preimage_metadata_path is None:
                    raise ValueError("curved_preimage_metadata_path is required")

    @classmethod
    def from_json(cls, path: Path | str) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "mode" not in payload:
            payload["mode"] = (
                "ambient_bo"
                if payload.get("sdr_method", "none") == "none"
                else "ambient_bo_sdr"
            )
            payload["dim"] = int(payload.pop("D"))
            payload["vae_checkpoint"] = ""
            payload["latent_dim"] = int(payload.get("latent_dim", payload["dim"]))
            if "lower" in payload:
                payload["problem_bound_lower"] = float(payload["lower"])
            if "upper" in payload:
                payload["problem_bound_upper"] = float(payload["upper"])
        if payload["mode"] in {"ambient_bo", "fixed_no_sdr", "retrain_dml"}:
            payload["sdr_method"] = "none"
        for key, value in list(payload.items()):
            if value and isinstance(value, str) and (
                key in {"output_dir", "vae_checkpoint"} or key.endswith("_path")
            ):
                candidate = Path(value)
                payload[key] = str(
                    candidate if candidate.is_absolute() else _WORKSPACE_ROOT / candidate
                )
        public_fields = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in public_fields})

    @property
    def settings(self) -> "PipelineSettings":
        return nested_settings(self)

    def dtype_obj(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"unsupported dtype {self.dtype!r}")

    def selected_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass(frozen=True)
class GPSettings:
    kernel: str
    target_transform: str
    fit_strategy: str
    duplicate_handling: str
    max_iterations: int
    observation_variance: float
    matern_use_scale_kernel: bool


@dataclass(frozen=True)
class AcquisitionSettings:
    kind: str
    warmup: int
    raw_samples: int
    restarts: int
    max_iterations: int
    timeout_seconds: float


@dataclass(frozen=True)
class VAESettings:
    checkpoint: str
    state_file: str
    latent_dimension: int
    input_lower: float
    input_upper: float


@dataclass(frozen=True)
class RetrainingSettings:
    start_iteration: int
    minimum_points: int
    period: int
    schedule: str
    epochs: int
    batch_size: int
    learning_rate: float | None
    acceptance_policy: str


@dataclass(frozen=True)
class DMLSettings:
    threshold: float
    eta: float
    triplet_loss_weight: float
    normalize_targets: bool = True


@dataclass(frozen=True)
class SDRSettings:
    method: str
    period: int
    original_minimum_window_unit: float


@dataclass(frozen=True)
class PipelineSettings:
    gp: GPSettings
    acquisition: AcquisitionSettings
    vae: VAESettings
    retraining: RetrainingSettings
    dml: DMLSettings
    sdr: SDRSettings


def nested_settings(config: PipelineConfig) -> PipelineSettings:
    return PipelineSettings(
        gp=GPSettings(
            kernel=config.gp_kernel,
            target_transform=config.gp_target_transform,
            fit_strategy=config.gp_fit_strategy,
            duplicate_handling=config.gp_duplicate_handling,
            max_iterations=config.gp_maxiter,
            observation_variance=config.train_y_var,
            matern_use_scale_kernel=config.gp_matern_use_scale_kernel,
        ),
        acquisition=AcquisitionSettings(
            kind=config.acquisition,
            warmup=config.acq_warmup,
            raw_samples=config.acq_raw_samples,
            restarts=config.acq_num_restarts,
            max_iterations=config.acq_maxiter,
            timeout_seconds=config.acq_timeout_sec,
        ),
        vae=VAESettings(
            checkpoint=config.vae_checkpoint,
            state_file=config.vae_state_file,
            latent_dimension=config.latent_dim,
            input_lower=config.vae_input_lower,
            input_upper=config.vae_input_upper,
        ),
        retraining=RetrainingSettings(
            start_iteration=config.retrain_start_iteration,
            minimum_points=config.retrain_min_points,
            period=config.retrain_period,
            schedule=config.retrain_schedule_kind,
            epochs=config.retrain_epochs,
            batch_size=config.retrain_batch_size,
            learning_rate=config.retrain_learning_rate,
            acceptance_policy=config.retrain_acceptance_policy,
        ),
        dml=DMLSettings(
            threshold=config.dml_threshold,
            eta=config.dml_eta,
            triplet_loss_weight=config.beta_metric_loss,
        ),
        sdr=SDRSettings(
            method=config.sdr_method,
            period=config.sdr_period,
            original_minimum_window_unit=config.original_sdr_minimum_window_unit,
        ),
    )


__all__ = [
    "AcquisitionSettings",
    "DMLSettings",
    "GPSettings",
    "PipelineConfig",
    "PipelineSettings",
    "RetrainingSettings",
    "SDRSettings",
    "VAESettings",
    "nested_settings",
]
