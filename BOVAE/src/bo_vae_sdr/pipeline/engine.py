"""Stateful iteration engine for the public BO-VAE algorithms."""

from __future__ import annotations

import os
import platform
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..sdr import OriginalSDR, original_sdr_provenance
from ..vae.artifacts import (
    VAETrainingConfig,
    append_jsonl,
    environment_metadata,
    retrain_vae_on_labeled_data,
    validate_vae_checkpoint,
    write_json,
)
from .config import PipelineConfig
from .core import (
    RETRAIN_ACCEPTANCE_METRICS,
    canonical_problem_key,
    problem_keys_match,
    retrain_schedule_decision,
    synchronized_perf_counter,
)
from .surrogate import (
    configured_vae_dimensions_match,
    fit_gp,
    load_configured_vae,
    make_problem,
    optimize_acquisition,
    problem_bounds,
    transform_gp_targets,
)
from .transforms import (
    DiagonalLatentTransform,
    IdentityLatentBoxTransform,
    ObjectiveTransform,
    VAEInputTransform,
    WhitenedLatentTransform,
    load_latent_transform_state,
    map_between_boxes,
    tensor_to_list,
    unit_bounds,
)


class PipelineRunner:
    """Run fixed or retrained BO-VAE in the full learned latent space."""

    def __init__(self, config: PipelineConfig) -> None:
        self._run_segment_started = time.perf_counter()
        initialization_started = self._run_segment_started
        self._elapsed_offset_seconds = 0.0
        self.config = config
        self.output_dir = Path(config.output_dir)
        if self.output_dir.exists() and config.overwrite and not config.resume:
            shutil.rmtree(self.output_dir)
        if (
            self.output_dir.exists()
            and any(self.output_dir.iterdir())
            and not (config.overwrite or config.resume)
        ):
            raise FileExistsError(f"{self.output_dir} already exists; set overwrite or resume")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = config.selected_device()
        self.dtype = config.dtype_obj()
        torch.set_default_dtype(self.dtype)
        torch.manual_seed(int(config.seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(config.seed))

        bounds = problem_bounds(config, self.dtype)
        self.problem = make_problem(
            config.problem,
            config.dim,
            bounds=bounds,
            curved_preimage_artifact_path=config.curved_preimage_artifact_path,
            curved_preimage_metadata_path=config.curved_preimage_metadata_path,
        )
        self.problem.bounds = self.problem.bounds.to(dtype=self.dtype)
        self.problem_box = ObjectiveTransform(
            self.problem.bounds,
            name="objective_domain",
        ).to(device=self.device, dtype=self.dtype)
        self.vae_box = VAEInputTransform(
            torch.tensor(
                [[config.vae_input_lower, config.vae_input_upper]] * config.dim,
                dtype=self.dtype,
            ),
            name="vae_input_domain",
        ).to(device=self.device, dtype=self.dtype)
        if (
            getattr(self.problem, "is_curved_preimage_benchmark", False)
            and not torch.allclose(
                self.problem_box.bounds,
                self.vae_box.bounds,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "curved-preimage BO-VAE requires identical objective and VAE domains"
            )

        self.model = load_configured_vae(config, device=self.device).to(dtype=self.dtype)
        self.vae_config = VAETrainingConfig.from_json(
            Path(config.vae_checkpoint) / "vae_config.json"
        )
        if not configured_vae_dimensions_match(config, self.vae_config):
            raise ValueError("VAE checkpoint dimensions do not match PipelineConfig")
        self.latent_anchor_x = self._load_latent_anchor()
        self.latent_transform_version = 0
        self.latent_transform = IdentityLatentBoxTransform(
            torch.tensor(
                [[-5.0, 5.0]] * config.latent_dim,
                dtype=self.dtype,
                device=self.device,
            ),
            name="latent_manuscript_box",
        )
        if config.mode in {"fixed_sdr", "retrain_sdr", "retrain_dml_sdr"}:
            self.sdr: OriginalSDR | None = OriginalSDR(
                dim=config.latent_dim,
                budget=config.budget,
                period=config.sdr_period,
                minimum_window_unit=config.original_sdr_minimum_window_unit,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.sdr = None

        self.iteration_trace = self.output_dir / "iteration_trace.jsonl"
        self.gp_trace = self.output_dir / "gp_trace.jsonl"
        self.acq_trace = self.output_dir / "acq_trace.jsonl"
        self.sdr_trace = self.output_dir / "sdr_trace.jsonl"
        self.retrain_schedule_trace = self.output_dir / "retrain_schedule_trace.jsonl"
        self.timing_trace = self.output_dir / "timing_trace.jsonl"
        self.start_iteration = 0
        self.start_stage_index = 0
        self.last_retrain_proposal_iteration: int | None = None
        self.last_retrain_best_original: float | None = None
        if config.resume and (self.output_dir / "checkpoint.pt").exists():
            self._load_checkpoint()
        else:
            self.fit_latent_transform(stage_index=0)
            (
                self.x_vae_raw_history,
                self.x_vae,
                self.y_internal,
                self.y_original,
            ) = self._initial_design()
            self._write_initial_design()

        initialization_finished = synchronized_perf_counter(self.device)
        append_jsonl(
            self.timing_trace,
            {
                "phase": "pipeline_initialization",
                "resumed": bool(config.resume and self.start_iteration > 0),
                "segment_duration_seconds": initialization_finished - initialization_started,
                "cumulative_elapsed_seconds": self.elapsed_seconds(),
                "device": str(self.device),
                "dtype": str(self.dtype),
            },
        )

    def _load_latent_anchor(self) -> torch.Tensor:
        anchor_path = (
            Path(self.config.latent_transform_anchor_path)
            if self.config.latent_transform_anchor_path
            else Path(self.config.vae_checkpoint) / "train_data.pt"
        )
        if not anchor_path.exists():
            raise FileNotFoundError(f"Missing latent-transform anchor data: {anchor_path}")
        payload = torch.load(anchor_path, map_location="cpu", weights_only=False)
        anchor_x = payload[0] if isinstance(payload, (tuple, list)) else payload
        anchor_x = anchor_x.to(device=self.device, dtype=self.dtype)
        if anchor_x.ndim != 2 or anchor_x.shape[1] != self.config.dim:
            raise ValueError("latent-transform anchor data has incompatible shape")
        return anchor_x

    def fit_latent_transform(self, stage_index: int) -> dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            anchor_mu, _ = self.model.encoder(self.latent_anchor_x)
        kind = self.config.latent_transform_kind
        if kind == "identity":
            transform = IdentityLatentBoxTransform(
                torch.tensor(
                    [[-5.0, 5.0]] * self.config.latent_dim,
                    dtype=self.dtype,
                    device=self.device,
                ),
                name="latent_manuscript_box",
            )
        elif kind == "diagonal":
            transform = DiagonalLatentTransform.fit_anchor(
                anchor_mu,
                estimator=self.config.latent_transform_estimator,
                coverage_quantile=self.config.latent_transform_coverage,
            )
        elif kind == "whitened":
            transform = WhitenedLatentTransform.fit_anchor(
                anchor_mu,
                coverage_quantile=self.config.latent_transform_coverage,
                epsilon=self.config.latent_transform_epsilon,
            )
        else:
            raise ValueError(f"unsupported latent transform {kind!r}")
        self.latent_transform = transform.to(device=self.device, dtype=self.dtype)
        self.latent_transform_version = int(stage_index)
        metadata = {
            "stage_index": int(stage_index),
            "anchor_shape": list(self.latent_anchor_x.shape),
            "anchor_source": self.config.latent_transform_anchor_path
            or str(Path(self.config.vae_checkpoint) / "train_data.pt"),
            "diagnostics": transform.diagnostics(anchor_mu),
            "metadata": transform.metadata(),
        }
        write_json(
            self.output_dir / f"latent_transform_state_stage_{stage_index:03d}.json",
            metadata,
        )
        return metadata

    def _initial_design(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.initial_design_path:
            return self._initial_design_from_objective(Path(self.config.initial_design_path))
        train_path = Path(self.config.vae_checkpoint) / "train_data.pt"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing VAE training data: {train_path}")
        payload = torch.load(train_path, map_location="cpu", weights_only=False)
        x_train = payload[0] if isinstance(payload, (tuple, list)) else payload
        x_train = x_train.to(device=self.device, dtype=self.dtype)
        if x_train.shape[1] != self.config.dim:
            raise ValueError("VAE training-data dimension does not match PipelineConfig")
        generator = torch.Generator(device=self.device).manual_seed(int(self.config.seed))
        order = torch.randperm(x_train.shape[0], generator=generator, device=self.device)
        raw = x_train[order[: self.config.initial_points]].detach().clone()
        evaluated, _ = self.project_x_vae(raw)
        internal, original, _ = self.evaluate_x_vae(raw)
        return raw, evaluated, internal, original

    def _initial_design_from_objective(
        self,
        path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not path.exists():
            raise FileNotFoundError(f"Missing initial design: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        x_obj = payload["x_obj"].to(device=self.device, dtype=self.dtype)
        if (
            x_obj.ndim != 2
            or x_obj.shape[1] != self.config.dim
            or x_obj.shape[0] < self.config.initial_points
        ):
            raise ValueError("initial-design coordinates have incompatible shape")
        x_obj = x_obj[: self.config.initial_points].detach().clone()
        if "problem_key" in payload and not problem_keys_match(
            str(payload["problem_key"]),
            self.config.problem,
            self.config.dim,
        ):
            raise ValueError("initial-design problem does not match PipelineConfig")
        x_vae = map_between_boxes(
            x_obj,
            self.problem_box,
            self.vae_box,
            clip_source_unit=True,
        )
        if self.config.initial_design_trust_verified_objectives:
            if not bool(payload.get("objective_values_verified", False)):
                raise ValueError("trusted initial design must declare verified objectives")
            if "y_original" not in payload:
                raise ValueError("trusted initial design is missing y_original")
            original = payload["y_original"][: self.config.initial_points].to(
                device=self.device,
                dtype=self.dtype,
            ).reshape(-1, 1)
            internal = -original
        else:
            internal = self.problem.func(x_obj).to(device=self.device, dtype=self.dtype)
            original = -internal
        if "y_original" in payload:
            expected = payload["y_original"][: self.config.initial_points].to(
                device=self.device,
                dtype=self.dtype,
            ).reshape_as(original)
            if not torch.allclose(expected, original, rtol=1e-9, atol=1e-9):
                raise ValueError("initial-design objectives do not match the benchmark")
            original = expected.detach().clone()
            internal = -original
        return x_vae.detach().clone(), x_vae.detach().clone(), internal, original

    def _write_initial_design(self) -> None:
        if self.config.initial_design_path:
            payload = torch.load(
                Path(self.config.initial_design_path),
                map_location="cpu",
                weights_only=False,
            )
            x_obj = payload["x_obj"][: self.config.initial_points].detach().cpu()
        else:
            x_obj = map_between_boxes(
                self.x_vae,
                self.vae_box,
                self.problem_box,
                clip_source_unit=True,
            ).detach().cpu()
        torch.save(
            {
                "seed": self.config.seed,
                "initial_points": self.config.initial_points,
                "initial_design_path": self.config.initial_design_path,
                "x_vae_raw": self.x_vae_raw_history.detach().cpu(),
                "x_vae_eval": self.x_vae.detach().cpu(),
                "x_obj": x_obj,
                "y_internal": self.y_internal.detach().cpu(),
                "y_original": self.y_original.detach().cpu(),
                "problem": self.problem.name,
                "problem_key": canonical_problem_key(self.config.problem, self.config.dim),
                "problem_metadata": self.problem.metadata()
                if hasattr(self.problem, "metadata")
                else None,
                "vae_checkpoint": self.config.vae_checkpoint,
            },
            self.output_dir / "initial_design.pt",
        )

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(
            self.output_dir / "checkpoint.pt",
            map_location=self.device,
            weights_only=False,
        )
        self._elapsed_offset_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
        self._run_segment_started = time.perf_counter()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.x_vae = checkpoint["x_vae_eval"].to(self.device, self.dtype)
        self.x_vae_raw_history = checkpoint["x_vae_raw"].to(self.device, self.dtype)
        self.y_internal = checkpoint["y_internal"].to(self.device, self.dtype)
        self.y_original = checkpoint["y_original"].to(self.device, self.dtype)
        self.start_iteration = int(checkpoint["iteration"]) + 1
        self.start_stage_index = int(checkpoint.get("stage_index", 0))
        self.last_retrain_proposal_iteration = checkpoint.get(
            "last_retrain_proposal_iteration"
        )
        self.last_retrain_best_original = checkpoint.get("last_retrain_best_original")
        self.latent_transform = load_latent_transform_state(
            checkpoint["latent_transform_state"]
        ).to(device=self.device, dtype=self.dtype)
        self.latent_transform_version = self.start_stage_index
        if self.sdr is not None and checkpoint.get("sdr_state") is not None:
            self.sdr.load_state_dict(checkpoint["sdr_state"])
        if checkpoint.get("torch_cpu_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_cpu_rng_state"].cpu())
        if checkpoint.get("torch_cuda_rng_states") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["torch_cuda_rng_states"]]
            )

    def project_x_vae(self, raw_x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        raw_x = raw_x.detach().to(device=self.device, dtype=self.dtype)
        violation = self.vae_box.violation(raw_x)
        evaluated = self.vae_box.clip(raw_x)
        delta = evaluated - raw_x
        point_projected = violation.max(dim=1).values > 0
        return evaluated, {
            "max_vae_bound_violation": float(violation.max().item()),
            "clipped_fraction": float(point_projected.to(self.dtype).mean().item()),
            "projection_l2": tensor_to_list(torch.linalg.vector_norm(delta, ord=2, dim=1)),
            "projection_linf": tensor_to_list(
                torch.linalg.vector_norm(delta, ord=float("inf"), dim=1)
            ),
            "x_vae_eval": tensor_to_list(evaluated),
        }

    def evaluate_x_vae(
        self,
        x_vae: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        evaluated, trace = self.project_x_vae(x_vae)
        objective_x = map_between_boxes(
            evaluated,
            self.vae_box,
            self.problem_box,
            clip_source_unit=True,
        )
        internal = self.problem.func(objective_x).to(device=self.device, dtype=self.dtype)
        trace["objective_x"] = tensor_to_list(objective_x)
        return internal.reshape(-1, 1), (-internal).reshape(-1, 1), trace

    def encode_dataset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encoder(self.x_vae)
        unit_raw = self.latent_transform.z_to_u(mu)
        unit = unit_raw.clamp(0.0, 1.0)
        outside = ((unit_raw < 0.0) | (unit_raw > 1.0)).any(dim=1)
        return unit, {
            "latent_mu_min": tensor_to_list(mu.min(dim=0).values),
            "latent_mu_max": tensor_to_list(mu.max(dim=0).values),
            "latent_logvar_mean": tensor_to_list(logvar.mean(dim=0)),
            "encoded_unit_clip_fraction": float(outside.to(self.dtype).mean().item()),
            "unit_min": tensor_to_list(unit.min(dim=0).values),
            "unit_max": tensor_to_list(unit.max(dim=0).values),
        }

    def model_state_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def restore_model_state(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(
            {key: value.to(self.device, self.dtype) for key, value in state.items()}
        )
        self.model.eval()

    def vae_dataset_diagnostics(self, x_vae: torch.Tensor) -> dict[str, float]:
        self.model.eval()
        x = x_vae.detach().to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            mu, logvar = self.model.encoder(x)
            reconstruction = self.model.decoder(mu)
        reconstruction_loss = float(
            torch.sum((reconstruction - x).pow(2), dim=1).mean().item()
        )
        kl_loss = float(
            torch.sum(-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()), dim=1)
            .mean()
            .item()
        )
        beta = float(self.vae_config.beta_final)
        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "vae_loss": reconstruction_loss + beta * kl_loss,
        }

    def retrain_acceptance_diagnostics(self) -> dict[str, dict[str, float]]:
        diagnostics = {"observed": self.vae_dataset_diagnostics(self.x_vae)}
        if self.config.retrain_acceptance_check_anchor:
            diagnostics["anchor"] = self.vae_dataset_diagnostics(self.latent_anchor_x)
        return diagnostics

    @staticmethod
    def _acceptance_limit(before: float, relative: float, absolute: float) -> float:
        return abs(before) * (1.0 + relative) + absolute

    def assess_retrain_acceptance(
        self,
        before: dict[str, dict[str, float]] | None,
        after: dict[str, dict[str, float]] | None,
    ) -> tuple[bool, list[dict[str, Any]]]:
        if self.config.retrain_acceptance_policy == "always_accept":
            return True, []
        if before is None or after is None:
            return False, [{"reason": "missing_retraining_diagnostics"}]
        tolerances = {
            "vae_loss": (
                self.config.retrain_acceptance_loss_rel_tol,
                self.config.retrain_acceptance_loss_abs_tol,
            ),
            "reconstruction_loss": (
                self.config.retrain_acceptance_reconstruction_rel_tol,
                self.config.retrain_acceptance_reconstruction_abs_tol,
            ),
            "kl_loss": (
                self.config.retrain_acceptance_kl_rel_tol,
                self.config.retrain_acceptance_kl_abs_tol,
            ),
        }
        selected_metrics = RETRAIN_ACCEPTANCE_METRICS[
            self.config.retrain_acceptance_policy
        ]
        failures: list[dict[str, Any]] = []
        for dataset, before_values in before.items():
            after_values = after.get(dataset)
            if after_values is None:
                failures.append({"dataset": dataset, "reason": "missing_after_values"})
                continue
            for metric in selected_metrics:
                relative, absolute = tolerances[metric]
                limit = self._acceptance_limit(before_values[metric], relative, absolute)
                if after_values[metric] > limit:
                    failures.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "before": before_values[metric],
                            "after": after_values[metric],
                            "limit": limit,
                        }
                    )
        return not failures, failures

    def write_metadata(self) -> None:
        metadata = {
            "artifact_type": "bovae_pipeline_run",
            "config": asdict(self.config),
            "pipeline_mode": self.config.mode,
            "problem": self.problem.name,
            "problem_key": canonical_problem_key(self.config.problem, self.config.dim),
            "problem_metadata": self.problem.metadata()
            if hasattr(self.problem, "metadata")
            else None,
            "objective_convention": {
                "internal_target": "negative original objective, maximized by BO",
                "original_objective": "minimized and reported in manuscript outputs",
            },
            "latent_search": {
                "dimension": self.config.latent_dim,
                "coordinates": "full learned latent space",
                "candidate_path": "unit box -> latent transform -> VAE decoder -> bounded objective domain",
            },
            "vae_retraining": {
                "loss_terms": ["reconstruction", "kl"],
                "dml_adds": "triplet loss on normalized observed objectives",
                "acceptance_policy": self.config.retrain_acceptance_policy,
            },
            "transforms": {
                "vae_input": self.vae_box.metadata(),
                "objective": self.problem_box.metadata(),
                "latent": self.latent_transform.metadata(),
            },
            "vae_checkpoint_validation": validate_vae_checkpoint(
                Path(self.config.vae_checkpoint),
                device=self.device,
                state_file=self.config.vae_state_file,
            ),
            "sdr_method": self.config.sdr_method if self.sdr is not None else None,
            "original_sdr": original_sdr_provenance() if self.sdr is not None else None,
            "environment": {**environment_metadata(self.device), "platform": platform.platform()},
        }
        write_json(self.output_dir / "run_metadata.json", metadata)
        write_json(self.output_dir / "pipeline_config.json", asdict(self.config))
        if self.sdr is not None:
            write_json(self.output_dir / "sdr_profile.json", self.sdr.metadata())

    def maybe_retrain(self, iteration: int, stage_index: int) -> dict[str, Any] | None:
        if self.config.mode in {"fixed_no_sdr", "fixed_sdr"}:
            return None
        if iteration < self.config.retrain_start_iteration:
            return None
        if (
            self.config.retrain_min_points > 0
            and self.y_original.shape[0] < self.config.retrain_min_points
        ):
            return None
        current_best = float(self.y_original.min().item())
        schedule = retrain_schedule_decision(
            kind=self.config.retrain_schedule_kind,
            iteration=iteration,
            fixed_period=self.config.retrain_period,
            adaptive_min_gap=self.config.retrain_adaptive_min_gap,
            adaptive_max_gap=self.config.retrain_adaptive_max_gap,
            adaptive_feedback_incumbent_threshold=(
                self.config.retrain_adaptive_feedback_incumbent_threshold
            ),
            current_best_original=current_best,
            observed_min_original=current_best,
            observed_max_original=float(self.y_original.max().item()),
            last_proposal_iteration=self.last_retrain_proposal_iteration,
            last_proposal_best_original=self.last_retrain_best_original,
        )
        append_jsonl(self.retrain_schedule_trace, schedule)
        if not schedule["trigger"]:
            return None

        before_state = self.model_state_snapshot()
        before = (
            None
            if self.config.retrain_acceptance_policy == "always_accept"
            else self.retrain_acceptance_diagnostics()
        )
        use_dml = self.config.mode in {"retrain_dml", "retrain_dml_sdr"}
        result = retrain_vae_on_labeled_data(
            self.model,
            self.x_vae,
            self.y_original,
            config=self.vae_config,
            output_dir=self.output_dir,
            stage_index=stage_index,
            epochs=self.config.retrain_epochs,
            batch_size=self.config.retrain_batch_size,
            use_dml=use_dml,
            dml_threshold=self.config.dml_threshold,
            dml_eta=self.config.dml_eta,
            dml_loss_weight=self.config.beta_metric_loss,
            learning_rate=self.config.retrain_learning_rate,
            weight_decay=self.config.retrain_weight_decay,
        )
        after = (
            None
            if self.config.retrain_acceptance_policy == "always_accept"
            else self.retrain_acceptance_diagnostics()
        )
        accepted, failures = self.assess_retrain_acceptance(before, after)
        result.update(
            {
                "iteration": iteration,
                "retrain_schedule": schedule,
                "retrain_accepted": accepted,
                "retrain_acceptance_policy": self.config.retrain_acceptance_policy,
                "retrain_acceptance_before": before,
                "retrain_acceptance_after": after,
                "retrain_rejection_reasons": failures,
            }
        )
        self.last_retrain_proposal_iteration = iteration
        self.last_retrain_best_original = current_best
        if not accepted:
            self.restore_model_state(before_state)
            result["sdr_reset"] = "not_reset_retrain_rejected" if self.sdr else None
            return result
        result["latent_transform"] = self.fit_latent_transform(stage_index + 1)
        if self.sdr is not None:
            self.sdr.reset()
            result["sdr_reset"] = "reset_after_vae_retraining"
        return result

    def save_checkpoint(self, iteration: int, stage_index: int) -> None:
        path = self.output_dir / "checkpoint.pt"
        temporary = self.output_dir / "checkpoint.pt.tmp"
        torch.save(
            {
                "iteration": iteration,
                "stage_index": stage_index,
                "x_vae_eval": self.x_vae.detach().cpu(),
                "x_vae_raw": self.x_vae_raw_history.detach().cpu(),
                "y_internal": self.y_internal.detach().cpu(),
                "y_original": self.y_original.detach().cpu(),
                "model_state_dict": self.model_state_snapshot(),
                "latent_transform_state": self.latent_transform.state_dict(),
                "sdr_state": self.sdr.state_dict() if self.sdr is not None else None,
                "elapsed_seconds": self.elapsed_seconds(),
                "last_retrain_proposal_iteration": self.last_retrain_proposal_iteration,
                "last_retrain_best_original": self.last_retrain_best_original,
                "torch_cpu_rng_state": torch.get_rng_state().cpu(),
                "torch_cuda_rng_states": (
                    [state.cpu() for state in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available()
                    else None
                ),
            },
            temporary,
        )
        os.replace(temporary, path)

    def run(self) -> dict[str, Any]:
        resume_cpu_state = None
        resume_cuda_states = None
        if self.config.resume and self.start_iteration > 0:
            resume_cpu_state = torch.get_rng_state().cpu()
            if torch.cuda.is_available():
                resume_cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]
        self.write_metadata()
        if resume_cpu_state is not None:
            torch.set_rng_state(resume_cpu_state)
        if resume_cuda_states is not None:
            torch.cuda.set_rng_state_all(resume_cuda_states)

        stage_index = self.start_stage_index
        for iteration in range(self.start_iteration, self.config.budget):
            iteration_started = synchronized_perf_counter(self.device)
            retrain_started = iteration_started
            retrain_info = self.maybe_retrain(iteration, stage_index)
            retrain_finished = synchronized_perf_counter(self.device)
            if retrain_info is not None:
                append_jsonl(
                    self.output_dir / "stage_trace.jsonl",
                    {"iteration": iteration, **retrain_info},
                )
                if retrain_info["retrain_accepted"]:
                    stage_index += 1

            gp_started = synchronized_perf_counter(self.device)
            train_unit, encode_trace = self.encode_dataset()
            y_gp, target_trace = transform_gp_targets(
                self.y_internal,
                self.config.gp_target_transform,
            )
            gp_model, gp_info = fit_gp(
                train_unit,
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
            gp_finished = synchronized_perf_counter(self.device)
            append_jsonl(
                self.gp_trace,
                {
                    "iteration": iteration,
                    **gp_info,
                    "target_transform": target_trace,
                    **encode_trace,
                },
            )

            bounds = (
                self.sdr.current_bounds
                if self.sdr is not None
                else unit_bounds(self.config.latent_dim, device=self.device, dtype=self.dtype)
            )
            acquisition_started = synchronized_perf_counter(self.device)
            candidate_unit, acquisition_value, acquisition_info = optimize_acquisition(
                gp_model,
                bounds,
                acquisition=self.config.acquisition,
                best_f=torch.max(y_gp),
                warmup=self.config.acq_warmup,
                raw_samples=self.config.acq_raw_samples,
                num_restarts=self.config.acq_num_restarts,
                maxiter=self.config.acq_maxiter,
                timeout_sec=self.config.acq_timeout_sec,
            )
            acquisition_finished = synchronized_perf_counter(self.device)
            latent = self.latent_transform.u_to_z(candidate_unit)
            self.model.eval()
            with torch.no_grad():
                raw_x = self.model.decoder(latent).detach()
            evaluated_x, projection_trace = self.project_x_vae(raw_x)
            internal_new, original_new, evaluation_trace = self.evaluate_x_vae(raw_x)
            evaluation_trace.update(projection_trace)
            evaluation_finished = synchronized_perf_counter(self.device)

            previous_best = float(self.y_original.min().item())
            duplicate = bool(
                torch.any(
                    torch.linalg.vector_norm(
                        self.x_vae - evaluated_x.to(self.x_vae),
                        ord=float("inf"),
                        dim=1,
                    )
                    <= 1e-10
                ).item()
            )
            self.x_vae_raw_history = torch.cat(
                [self.x_vae_raw_history, raw_x.to(self.x_vae_raw_history)]
            )
            self.x_vae = torch.cat([self.x_vae, evaluated_x.to(self.x_vae)])
            self.y_internal = torch.cat([self.y_internal, internal_new.to(self.y_internal)])
            self.y_original = torch.cat([self.y_original, original_new.to(self.y_original)])
            best_original = float(self.y_original.min().item())

            candidate_near_boundary = None
            sdr_info = None
            sdr_started = synchronized_perf_counter(self.device)
            if self.sdr is not None:
                candidate_near_boundary = self.sdr.record_candidate_boundary(candidate_unit)
                train_unit_after, _ = self.encode_dataset()
                _, sdr_info = self.sdr.update(
                    train_unit_x=train_unit_after,
                    train_internal_y=self.y_internal,
                    best_original=best_original,
                    iteration=iteration,
                )
                append_jsonl(self.sdr_trace, sdr_info)
            sdr_finished = synchronized_perf_counter(self.device)

            phase_timing = {
                "retraining_seconds": retrain_finished - retrain_started,
                "gp_seconds": gp_finished - gp_started,
                "acquisition_seconds": acquisition_finished - acquisition_started,
                "decode_and_evaluation_seconds": evaluation_finished - acquisition_finished,
                "sdr_seconds": sdr_finished - sdr_started,
            }
            append_jsonl(self.acq_trace, {"iteration": iteration, **acquisition_info})
            append_jsonl(
                self.iteration_trace,
                {
                    "iteration": iteration,
                    "evaluation_index": self.config.initial_points + iteration,
                    "elapsed_seconds": self.elapsed_seconds(),
                    "mode": self.config.mode,
                    "candidate_unit": tensor_to_list(candidate_unit),
                    "candidate_latent": tensor_to_list(latent),
                    "candidate_x_vae_raw": tensor_to_list(raw_x),
                    "candidate_x_vae_eval": tensor_to_list(evaluated_x),
                    "candidate_duplicate_x_vae_eval": duplicate,
                    "internal_target": float(internal_new.item()),
                    "original_value": float(original_new.item()),
                    "best_internal": float(self.y_internal.max().item()),
                    "best_original": best_original,
                    "improved_original": best_original < previous_best,
                    "candidate_near_sdr_boundary": candidate_near_boundary,
                    "sdr_bounds": tensor_to_list(self.sdr.current_bounds)
                    if self.sdr is not None
                    else None,
                    "evaluation": evaluation_trace,
                    "acq_value": float(acquisition_value.reshape(-1)[0].item()),
                    "phase_timing_seconds": phase_timing,
                },
            )
            checkpoint_started = synchronized_perf_counter(self.device)
            self.save_checkpoint(iteration, stage_index)
            checkpoint_finished = synchronized_perf_counter(self.device)
            append_jsonl(
                self.timing_trace,
                {
                    "phase": "bo_iteration",
                    "iteration": iteration,
                    "duration_seconds": checkpoint_finished - iteration_started,
                    **phase_timing,
                    "checkpoint_seconds": checkpoint_finished - checkpoint_started,
                    "retraining_proposed": retrain_info is not None,
                    "retraining_accepted": (
                        None if retrain_info is None else retrain_info["retrain_accepted"]
                    ),
                },
            )

        summary = {
            "mode": self.config.mode,
            "problem": self.problem.name,
            "seed": self.config.seed,
            "budget": self.config.budget,
            "initial_points": self.config.initial_points,
            "final_best_internal": float(self.y_internal.max().item()),
            "final_best_original": float(self.y_original.min().item()),
            "best_index": int(torch.argmin(self.y_original.reshape(-1)).item()),
            "n_observations": int(self.y_original.shape[0]),
            "elapsed_seconds": self.elapsed_seconds(),
            "sdr_method": self.config.sdr_method if self.sdr is not None else None,
            "sdr_update_count": self.sdr.update_count if self.sdr is not None else None,
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary

    def elapsed_seconds(self) -> float:
        return self._elapsed_offset_seconds + (
            time.perf_counter() - self._run_segment_started
        )


__all__ = ["PipelineRunner"]
