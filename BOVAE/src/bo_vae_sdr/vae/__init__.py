"""VAE construction, pretraining, retraining, and metric learning."""

from .artifacts import (
    VAETrainingConfig,
    generate_pretraining_data,
    load_pretrained_vae,
    train_vae,
)
from .dml import TargetNormalizer, dml_triplet_stats, normalize_dml_targets

__all__ = [
    "TargetNormalizer",
    "VAETrainingConfig",
    "dml_triplet_stats",
    "generate_pretraining_data",
    "load_pretrained_vae",
    "normalize_dml_targets",
    "train_vae",
]
