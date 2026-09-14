"""Maintained BO and BO-VAE optimization pipelines.

The public facade is loaded lazily so the independent SDR packages can use
the coordinate contracts without creating a package import cycle.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "PipelineConfig": (".runner", "PipelineConfig"),
    "PipelineRunner": (".runner", "PipelineRunner"),
    "run_pipeline": (".runner", "run_pipeline"),
    "AcquisitionSettings": (".config", "AcquisitionSettings"),
    "DMLSettings": (".config", "DMLSettings"),
    "GPSettings": (".config", "GPSettings"),
    "PipelineSettings": (".config", "PipelineSettings"),
    "RetrainingSettings": (".config", "RetrainingSettings"),
    "SDRSettings": (".config", "SDRSettings"),
    "VAESettings": (".config", "VAESettings"),
    "run_ambient_bo": (".algorithms", "run_ambient_bo"),
    "run_ambient_bo_sdr": (".algorithms", "run_ambient_bo_sdr"),
    "run_dml_retrained_bovae_sdr": (".algorithms", "run_dml_retrained_bovae_sdr"),
    "run_fixed_bovae": (".algorithms", "run_fixed_bovae"),
    "run_fixed_bovae_sdr": (".algorithms", "run_fixed_bovae_sdr"),
    "run_retrained_bovae_sdr": (".algorithms", "run_retrained_bovae_sdr"),
    "acquisition_values": (".acquisition", "acquisition_values"),
    "optimize_acquisition": (".acquisition", "optimize_acquisition"),
    "fit_gp": (".gaussian_process", "fit_gp"),
    "rank_gaussian_transform": (".gaussian_process", "rank_gaussian_transform"),
    "transform_gp_targets": (".gaussian_process", "transform_gp_targets"),
    "VAETrainingConfig": ("..vae.artifacts", "VAETrainingConfig"),
    "load_pretrained_vae": ("..vae.artifacts", "load_pretrained_vae"),
    "train_vae": ("..vae.artifacts", "train_vae"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
