"""Public composition point for the maintained optimization engine."""

from __future__ import annotations

from typing import Any

from .config import PipelineConfig
from .core import synchronized_perf_counter
from .engine import PipelineRunner
from .surrogate import (
    acquisition_values,
    fit_gp,
    load_configured_vae,
    make_problem,
    optimize_acquisition,
    problem_bounds,
    rank_gaussian_transform,
    transform_gp_targets,
)


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    if config.mode in {"ambient_bo", "ambient_bo_sdr"}:
        from .ambient import AmbientPipelineRunner

        return AmbientPipelineRunner(config).run()
    return PipelineRunner(config).run()


__all__ = [
    "PipelineConfig",
    "PipelineRunner",
    "acquisition_values",
    "fit_gp",
    "load_configured_vae",
    "make_problem",
    "optimize_acquisition",
    "problem_bounds",
    "rank_gaussian_transform",
    "run_pipeline",
    "synchronized_perf_counter",
    "transform_gp_targets",
]
