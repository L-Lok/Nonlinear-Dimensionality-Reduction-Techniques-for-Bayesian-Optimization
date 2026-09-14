"""Thin public entry points for the maintained manuscript algorithms."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .runner import PipelineConfig


def _run(config: PipelineConfig, **changes: Any) -> dict[str, Any]:
    from .runner import run_pipeline

    return run_pipeline(replace(config, **changes))


def run_ambient_bo(config: PipelineConfig) -> dict[str, Any]:
    return _run(config, mode="ambient_bo", sdr_method="none")


def run_ambient_bo_sdr(config: PipelineConfig) -> dict[str, Any]:
    return _run(config, mode="ambient_bo_sdr", sdr_method="original")


def run_fixed_bovae(config: PipelineConfig) -> dict[str, Any]:
    return _run(config, mode="fixed_no_sdr", sdr_method="none")


def run_fixed_bovae_sdr(config: PipelineConfig) -> dict[str, Any]:
    return _run(config, mode="fixed_sdr", sdr_method="original")


def run_retrained_bovae_sdr(config: PipelineConfig) -> dict[str, Any]:
    return _run(config, mode="retrain_sdr", sdr_method="original")


def run_dml_retrained_bovae_sdr(config: PipelineConfig) -> dict[str, Any]:
    return _run(config, mode="retrain_dml_sdr", sdr_method="original")


__all__ = [
    "run_ambient_bo",
    "run_ambient_bo_sdr",
    "run_dml_retrained_bovae_sdr",
    "run_fixed_bovae",
    "run_fixed_bovae_sdr",
    "run_retrained_bovae_sdr",
]
