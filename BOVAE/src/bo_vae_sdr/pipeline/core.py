"""Small shared utilities for the public BO-VAE pipeline."""

from __future__ import annotations

import time
from typing import Any

import torch


PIPELINE_MODES = {
    "ambient_bo",
    "ambient_bo_sdr",
    "fixed_no_sdr",
    "fixed_sdr",
    "retrain_sdr",
    "retrain_dml",
    "retrain_dml_sdr",
}

RETRAIN_ACCEPTANCE_METRICS = {
    "always_accept": (),
    "reconstruction_not_worse": ("reconstruction_loss",),
    "kl_not_worse": ("kl_loss",),
    "vae_loss_not_worse": ("vae_loss",),
}
RETRAIN_ACCEPTANCE_POLICIES = set(RETRAIN_ACCEPTANCE_METRICS)


def synchronized_perf_counter(device: torch.device) -> float:
    """Return a timestamp after queued device work has completed."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def normalized_problem_key(name: str) -> str:
    return str(name).lower().replace("-", "_")


def canonical_problem_key(name: str, dim: int) -> str:
    del dim
    return normalized_problem_key(name)


def problem_keys_match(candidate: str, configured: str, dim: int) -> bool:
    return canonical_problem_key(candidate, dim) == canonical_problem_key(configured, dim)


def retrain_schedule_decision(
    *,
    kind: str,
    iteration: int,
    fixed_period: int,
    adaptive_min_gap: int,
    adaptive_max_gap: int,
    adaptive_feedback_incumbent_threshold: float,
    current_best_original: float,
    observed_min_original: float,
    observed_max_original: float,
    last_proposal_iteration: int | None,
    last_proposal_best_original: float | None,
) -> dict[str, Any]:
    """Return a deterministic retraining decision from observed history only."""

    iteration = int(iteration)
    common = {
        "kind": kind,
        "iteration": iteration,
        "current_best_original": float(current_best_original),
        "observed_min_original": float(observed_min_original),
        "observed_max_original": float(observed_max_original),
        "uses_observed_data_only": True,
    }
    if kind == "fixed":
        trigger = iteration % int(fixed_period) == 0
        return {
            **common,
            "trigger": trigger,
            "reason": "fixed_period_boundary" if trigger else "not_fixed_period_boundary",
            "fixed_period": int(fixed_period),
        }
    if kind != "acceptance_feedback":
        raise ValueError(f"unsupported retrain schedule {kind!r}")
    if last_proposal_iteration is None or last_proposal_best_original is None:
        return {**common, "trigger": True, "reason": "initial_retrain_proposal"}

    gap = iteration - int(last_proposal_iteration)
    observed_span = max(
        0.0,
        float(observed_max_original) - float(observed_min_original),
    )
    gain = max(0.0, float(last_proposal_best_original) - float(current_best_original))
    normalized_gain = gain / observed_span if observed_span > 0.0 else 0.0
    threshold = float(adaptive_feedback_incumbent_threshold)
    if gap < int(adaptive_min_gap):
        trigger, reason = False, "minimum_gap_not_reached"
    elif normalized_gain >= threshold:
        trigger, reason = True, "observed_incumbent_progress"
    elif gap >= int(adaptive_max_gap):
        trigger, reason = True, "maximum_gap_reached"
    else:
        trigger, reason = False, "insufficient_observed_incumbent_progress"
    return {
        **common,
        "trigger": trigger,
        "reason": reason,
        "gap": gap,
        "normalized_incumbent_gain": normalized_gain,
        "adaptive_min_gap": int(adaptive_min_gap),
        "adaptive_max_gap": int(adaptive_max_gap),
        "adaptive_feedback_incumbent_threshold": threshold,
    }


__all__ = [
    "PIPELINE_MODES",
    "RETRAIN_ACCEPTANCE_METRICS",
    "RETRAIN_ACCEPTANCE_POLICIES",
    "canonical_problem_key",
    "normalized_problem_key",
    "problem_keys_match",
    "retrain_schedule_decision",
    "synchronized_perf_counter",
]
