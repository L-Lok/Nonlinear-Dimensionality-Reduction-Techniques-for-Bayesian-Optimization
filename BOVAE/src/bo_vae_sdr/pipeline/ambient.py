"""Ambient-space BO loop used by the manuscript BO/SDR studies."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..sdr import OriginalSDR
from ..vae.artifacts import append_jsonl, environment_metadata, write_json
from .checkpointing import load_checkpoint, save_checkpoint
from .runner import (
    PipelineConfig,
    fit_gp,
    make_problem,
    optimize_acquisition,
    problem_bounds,
    synchronized_perf_counter,
    transform_gp_targets,
)
from .transforms import ObjectiveTransform, tensor_to_list, unit_bounds


def _trim_jsonl(path: Path, next_iteration: int) -> None:
    if not path.exists():
        return
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        import json

        payload = json.loads(line)
        if int(payload.get("iteration", -1)) < int(next_iteration):
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _initial_design(
    config: PipelineConfig,
    objective: Any,
    box: ObjectiveTransform,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.initial_design_path:
        payload = torch.load(
            Path(config.initial_design_path), map_location=device, weights_only=False
        )
        candidate = next(
            (
                payload[key]
                for key in ("x_obj", "x_problem", "x", "train_x", "x_vae")
                if key in payload
            ),
            None,
        )
        if candidate is None:
            raise ValueError("initial design does not contain recognized coordinates")
        candidate = candidate[: config.initial_points].to(device=device, dtype=dtype)
        if torch.all(candidate >= 0.0) and torch.all(candidate <= 1.0):
            x_unit = candidate
            x_objective = box.from_unit(candidate)
        else:
            x_objective = candidate
            x_unit = box.to_unit(candidate)
        if "y_original" in payload:
            y_original = payload["y_original"][: config.initial_points].to(
                device=device, dtype=dtype
            ).reshape(-1, 1)
            y_internal = -y_original
        elif "y_internal" in payload:
            y_internal = payload["y_internal"][: config.initial_points].to(
                device=device, dtype=dtype
            ).reshape(-1, 1)
            y_original = -y_internal
        else:
            y_internal = objective.func(x_objective).to(device=device, dtype=dtype)
            y_original = -y_internal
    else:
        x_unit = torch.rand(
            config.initial_points, config.dim, device=device, dtype=dtype
        )
        x_objective = box.from_unit(x_unit)
        y_internal = objective.func(x_objective).to(device=device, dtype=dtype)
        y_original = -y_internal
    if x_unit.shape != (config.initial_points, config.dim):
        raise ValueError("initial design has an incompatible shape")
    if not torch.allclose(y_internal, -y_original, rtol=1e-9, atol=1e-9):
        raise ValueError("initial design violates internal_target = -original_objective")
    return x_unit.detach(), y_internal.reshape(-1, 1), y_original.reshape(-1, 1)


class AmbientPipelineRunner:
    """Shared implementation for plain ambient BO and ambient BO-SDR."""

    def __init__(self, config: PipelineConfig) -> None:
        if config.mode not in {"ambient_bo", "ambient_bo_sdr"}:
            raise ValueError("AmbientPipelineRunner requires an ambient mode")
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = config.selected_device()
        self.dtype = config.dtype_obj()
        torch.manual_seed(int(config.seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(config.seed))
        declared = problem_bounds(config, self.dtype)
        self.objective = make_problem(
            config.problem,
            config.dim,
            bounds=declared,
            curved_preimage_artifact_path=config.curved_preimage_artifact_path,
            curved_preimage_metadata_path=config.curved_preimage_metadata_path,
        )
        self.box = ObjectiveTransform(
            self.objective.bounds.to(device=self.device, dtype=self.dtype),
            name="objective_domain",
        )
        self.sdr = None
        if config.mode == "ambient_bo_sdr":
            self.sdr = OriginalSDR(
                config.dim,
                config.budget,
                period=config.sdr_period,
                minimum_window_unit=config.original_sdr_minimum_window_unit,
                device=self.device,
                dtype=self.dtype,
            )
        self.iteration_trace = self.output_dir / "iteration_trace.jsonl"
        self.gp_trace = self.output_dir / "gp_trace.jsonl"
        self.acq_trace = self.output_dir / "acq_trace.jsonl"
        self.sdr_trace = self.output_dir / "sdr_trace.jsonl"
        self.checkpoint_path = self.output_dir / "checkpoint.pt"
        self.elapsed_offset = 0.0
        if config.resume and self.checkpoint_path.exists():
            state = load_checkpoint(self.checkpoint_path, map_location=self.device)
            self.start_iteration = int(state["next_iteration"])
            self.x_unit = state["x_unit"].to(device=self.device, dtype=self.dtype)
            self.y_internal = state["y_internal"].to(device=self.device, dtype=self.dtype)
            self.y_original = state["y_original"].to(device=self.device, dtype=self.dtype)
            self.elapsed_offset = float(state.get("elapsed_seconds", 0.0))
            torch.set_rng_state(state["torch_cpu_rng_state"])
            if self.device.type == "cuda" and state.get("torch_cuda_rng_states"):
                torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])
            if self.sdr is not None and state.get("sdr_state") is not None:
                self.sdr.load_state_dict(state["sdr_state"])
            for path in (
                self.iteration_trace,
                self.gp_trace,
                self.acq_trace,
                self.sdr_trace,
            ):
                _trim_jsonl(path, self.start_iteration)
        else:
            self.start_iteration = 0
            self.x_unit, self.y_internal, self.y_original = _initial_design(
                config,
                self.objective,
                self.box,
                device=self.device,
                dtype=self.dtype,
            )
            write_json(
                self.output_dir / "run_metadata.json",
                {
                    "artifact_type": "ambient_pipeline_run",
                    "objective_convention": "internal_target = -original_objective",
                    "config": asdict(config),
                    "environment": environment_metadata(self.device),
                    "objective_bounds": self.box.bounds,
                },
            )

    def _save(self, next_iteration: int, elapsed_seconds: float) -> None:
        save_checkpoint(
            self.checkpoint_path,
            {
                "next_iteration": int(next_iteration),
                "x_unit": self.x_unit.detach().cpu(),
                "y_internal": self.y_internal.detach().cpu(),
                "y_original": self.y_original.detach().cpu(),
                "sdr_state": self.sdr.state_dict() if self.sdr is not None else None,
                "elapsed_seconds": float(elapsed_seconds),
                "torch_cpu_rng_state": torch.get_rng_state().cpu(),
                "torch_cuda_rng_states": (
                    [value.cpu() for value in torch.cuda.get_rng_state_all()]
                    if self.device.type == "cuda"
                    else []
                ),
            },
        )

    def run(self) -> dict[str, Any]:
        segment_started = synchronized_perf_counter(self.device)

        def elapsed() -> float:
            return self.elapsed_offset + synchronized_perf_counter(self.device) - segment_started

        for iteration in range(self.start_iteration, self.config.budget):
            y_gp, target_info = transform_gp_targets(
                self.y_internal, self.config.gp_target_transform
            )
            model, gp_info = fit_gp(
                self.x_unit,
                y_gp,
                self.config.train_y_var,
                self.config.gp_maxiter,
                kernel=self.config.gp_kernel,
                fit_strategy=self.config.gp_fit_strategy,
                duplicate_handling=self.config.gp_duplicate_handling,
                duplicate_tol=self.config.gp_duplicate_tol,
                matern_use_scale_kernel=self.config.gp_matern_use_scale_kernel,
                lengthscale_lower_bound=self.config.gp_lengthscale_lower_bound,
                lengthscale_upper_bound=self.config.gp_lengthscale_upper_bound,
                outputscale_lower_bound=self.config.gp_outputscale_lower_bound,
                outputscale_upper_bound=self.config.gp_outputscale_upper_bound,
                train_y_var_floor=self.config.gp_train_y_var_floor,
                torch_fallback_steps=self.config.gp_torch_fallback_steps,
                torch_fallback_lr=self.config.gp_torch_fallback_lr,
            )
            bounds = (
                self.sdr.current_bounds
                if self.sdr is not None
                else unit_bounds(self.config.dim, device=self.device, dtype=self.dtype)
            )
            candidate, acquisition_value, acquisition_info = optimize_acquisition(
                model,
                bounds,
                acquisition=self.config.acquisition,
                best_f=y_gp.max(),
                warmup=self.config.acq_warmup,
                raw_samples=self.config.acq_raw_samples,
                num_restarts=self.config.acq_num_restarts,
                maxiter=self.config.acq_maxiter,
                timeout_sec=self.config.acq_timeout_sec,
            )
            x_objective = self.box.from_unit(candidate)
            internal_new = self.objective.func(x_objective).to(
                device=self.device, dtype=self.dtype
            ).reshape(-1, 1)
            original_new = -internal_new
            previous_best = float(self.y_original.min().item())
            self.x_unit = torch.cat([self.x_unit, candidate.to(self.x_unit)])
            self.y_internal = torch.cat([self.y_internal, internal_new.to(self.y_internal)])
            self.y_original = torch.cat([self.y_original, original_new.to(self.y_original)])
            best_original = float(self.y_original.min().item())
            sdr_info = None
            if self.sdr is not None:
                near_boundary = self.sdr.record_candidate_boundary(candidate)
                _, sdr_info = self.sdr.update(
                    train_unit_x=self.x_unit,
                    train_internal_y=self.y_internal,
                    best_original=best_original,
                    iteration=iteration,
                )
                append_jsonl(
                    self.sdr_trace,
                    {"iteration": iteration, "candidate_near_boundary": near_boundary, **sdr_info},
                )
            append_jsonl(
                self.gp_trace,
                {"iteration": iteration, "target_transform": target_info, **gp_info},
            )
            append_jsonl(self.acq_trace, {"iteration": iteration, **acquisition_info})
            append_jsonl(
                self.iteration_trace,
                {
                    "iteration": iteration,
                    "enrichment_evaluation": iteration + 1,
                    "evaluation_count": self.config.initial_points + iteration + 1,
                    "candidate_unit": tensor_to_list(candidate),
                    "candidate_original_coordinates": tensor_to_list(x_objective),
                    "internal_target": float(internal_new.item()),
                    "original_value": float(original_new.item()),
                    "best_internal": float(self.y_internal.max().item()),
                    "best_original": best_original,
                    "improved_original": best_original < previous_best,
                    "acquisition_value": float(acquisition_value.item()),
                    "sdr_method": self.config.sdr_method,
                    "active_bounds": tensor_to_list(bounds),
                    "sdr_updated": bool(sdr_info and sdr_info.get("updated")),
                },
            )
            self._save(iteration + 1, elapsed())
        summary = {
            "mode": self.config.mode,
            "problem": self.config.problem,
            "dim": self.config.dim,
            "seed": self.config.seed,
            "gp_kernel": self.config.gp_kernel,
            "acquisition": self.config.acquisition,
            "sdr_method": self.config.sdr_method,
            "sdr_update_count": self.sdr.update_count if self.sdr is not None else 0,
            "n_observations": int(self.y_original.shape[0]),
            "final_best_internal": float(self.y_internal.max().item()),
            "final_best_original": float(self.y_original.min().item()),
            "elapsed_seconds": elapsed(),
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary


__all__ = ["AmbientPipelineRunner"]
