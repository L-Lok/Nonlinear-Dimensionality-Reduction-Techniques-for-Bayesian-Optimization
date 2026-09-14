"""EGORSE optimizer implementation for the MB_10/MB_100 reproduction."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.optimizer_interface import OptimizationHistory
from egorse.acquisition import ApproximateConstrainedEIConfig, propose_constrained_ei
from egorse.embeddings import build_embedding, methods_for_variant
from egorse.experiment import BudgetSpec
from egorse.geometry import (
    embedding_box,
    evaluate_projected,
    positive_constraint_for_known_x,
)


@dataclass(frozen=True)
class EGORSEConfig:
    """Configuration for Algorithm 2 in the EGORSE paper."""

    variant: str
    effective_dim: int = 2
    max_nb_it: int = 10
    max_nb_it_sub: int = 40
    backend_mode: str = "reconstructed_cbo"
    backend_env_path: str = ""
    backend_exact: bool = False
    backend_fallback_reason: str = ""
    backend_versions: dict[str, str] = field(default_factory=dict)
    backend_commits: dict[str, str] = field(default_factory=dict)
    allow_approximate_backend: bool = False
    gamma_tol: float = 1e-7
    smt_reference_path: str | None = None
    acquisition: ApproximateConstrainedEIConfig = ApproximateConstrainedEIConfig()

    @property
    def budget_spec(self) -> BudgetSpec:
        return BudgetSpec(
            variant=self.variant,
            effective_dim=self.effective_dim,
            max_nb_it=self.max_nb_it,
            max_nb_it_sub=self.max_nb_it_sub,
        )


class EGORSEOptimizer:
    """Run EGORSE Algorithm 2 on a minimization problem."""

    def __init__(self, config: EGORSEConfig):
        self.config = config
        if config.backend_mode not in {"exact_segomoe", "reconstructed_cbo", "smoke"}:
            raise ValueError(
                "backend_mode must be one of exact_segomoe, reconstructed_cbo, or smoke"
            )
        if config.backend_mode == "exact_segomoe":
            raise ValueError(
                "exact_segomoe requires an importable SEGOMOE runtime and SNOPT "
                "verification. SBArchOpt exposes a SEGOMOE interface, but its "
                "docs state SEGOMOE is not openly available and HAS_SEGOMOE did "
                "not verify in this workspace."
            )
        if config.backend_mode == "smoke" and not config.allow_approximate_backend:
            raise ValueError("smoke backend requires allow_approximate_backend=True")
        if config.backend_mode == "reconstructed_cbo" and not config.allow_approximate_backend:
            raise ValueError(
                "SEGOMOE runtime/SNOPT is unavailable in this setup. Set "
                "allow_approximate_backend=True to use the labelled approximate "
                "reconstructed_cbo backend."
            )

    def run(
        self,
        problem: Any,
        budget: int,
        init_X: np.ndarray,
        seed: int,
        record_callback: Callable[[dict[str, object]], None] | None = None,
        state_callback: Callable[[dict[str, object]], None] | None = None,
        *,
        init_Y: np.ndarray | None = None,
        init_wall_times: np.ndarray | None = None,
        resume_state: dict[str, object] | None = None,
        resume_records: list[dict[str, Any]] | None = None,
    ) -> OptimizationHistory:
        """Run EGORSE with a fixed enrichment-evaluation budget.

        ``init_Y`` allows a precomputed, shared initial design to be reused
        without physically evaluating it again for each optimizer.  When it is
        omitted, the historical behavior of evaluating ``init_X`` is kept.
        """

        start = time.perf_counter()
        methods = methods_for_variant(self.config.variant)
        init_arr = np.asarray(init_X, dtype=float)
        if init_arr.ndim != 2:
            raise ValueError("init_X must be two-dimensional")
        if init_Y is None:
            initial = None
            initial_times = []
            initial_values_source = "evaluated_by_optimizer"
        else:
            initial = np.asarray(init_Y, dtype=float).reshape(-1)
            if initial.shape[0] != init_arr.shape[0]:
                raise ValueError("init_Y must contain one value per row of init_X")
            if init_wall_times is None:
                initial_times = [0.0] * init_arr.shape[0]
            else:
                wall_times = np.asarray(init_wall_times, dtype=float).reshape(-1)
                if wall_times.shape[0] != init_arr.shape[0]:
                    raise ValueError(
                        "init_wall_times must contain one value per row of init_X"
                    )
                if np.any(wall_times < 0.0) or np.any(np.diff(wall_times) < 0.0):
                    raise ValueError("init_wall_times must be nonnegative and monotone")
                initial_times = wall_times.tolist()
            initial_values_source = "precomputed_shared_design"
        history = OptimizationHistory(
            metadata={
                "optimizer": "EGORSE",
                "variant": self.config.variant,
                "seed": seed,
                "budget": budget,
                "backend": self.config.backend_mode,
                "backend_mode": self.config.backend_mode,
                "backend_env_path": self.config.backend_env_path,
                "backend_exact": bool(self.config.backend_exact),
                "backend_is_approximate": not bool(self.config.backend_exact),
                "backend_fallback_reason": self.config.backend_fallback_reason,
                "backend_versions": dict(self.config.backend_versions),
                "backend_commits": dict(self.config.backend_commits),
                "initial_values_source": initial_values_source,
                "physical_objective_evaluations_in_optimizer": (
                    int(budget) if init_Y is not None else int(budget) + len(init_arr)
                ),
            }
        )

        if resume_state is None:
            rng = np.random.default_rng(seed)
            x_values = [row.copy() for row in init_arr]
            if initial is None:
                y_values = []
                initial_times = []
                for row in init_arr:
                    y_values.append(float(problem.evaluate(row)))
                    initial_times.append(time.perf_counter() - start)
            else:
                y_values = initial.tolist()
            initial_time_offset = float(initial_times[-1]) if initial_times else 0.0
            best_valid = float("inf")
            best_any = float("inf")
            eval_index = 0
            for row_idx, (x, y, initial_wall_time) in enumerate(
                zip(x_values, y_values, initial_times)
            ):
                eval_index += 1
                best_valid = min(best_valid, y)
                best_any = min(best_any, y)
                self._append_record(
                    history,
                    record_callback,
                    self._record(
                        problem=problem,
                        eval_index=eval_index,
                        seed=seed,
                        x=x,
                        u=problem.project(x),
                        f=y,
                        feasibility=True,
                        best_valid=best_valid,
                        best_any=best_any,
                        wall_time=float(initial_wall_time),
                        phase="initial_doe",
                        outer_iter=0,
                        subspace_method="initial_doe",
                        subspace_eval_index=row_idx + 1,
                        backend="initial_doe",
                        backend_is_approximate=False,
                        embedding_backend="not_applicable",
                        embedding_approximate=False,
                        gamma_status="not_applicable",
                        gamma_backend="not_applicable",
                        g=positive_constraint_for_known_x(x),
                    ),
                )
            enrichment_done = 0
            outer_iter = 0
            method_index = 0
            sub_iter = 0
            active_embedding = None
            elapsed_offset = initial_time_offset
        else:
            if int(resume_state["seed"]) != int(seed) or int(resume_state["budget"]) != int(budget):
                raise ValueError("EGORSE resume state does not match seed/budget")
            rng = np.random.default_rng()
            rng.bit_generator.state = dict(resume_state["rng_state"])
            x_array = np.asarray(resume_state["x_values"], dtype=float)
            y_array = np.asarray(resume_state["y_values"], dtype=float).reshape(-1)
            x_values = [row.copy() for row in x_array]
            y_values = y_array.tolist()
            enrichment_done = int(resume_state["enrichment_done"])
            eval_index = int(resume_state["eval_index"])
            best_valid = float(resume_state["best_valid"])
            best_any = float(resume_state["best_any"])
            outer_iter = int(resume_state["next_outer_iter"])
            method_index = int(resume_state["next_method_index"])
            sub_iter = int(resume_state["next_sub_iter"])
            active_embedding = resume_state.get("active_embedding")
            elapsed_offset = float(resume_state.get("elapsed_seconds", 0.0))
            history.records = list(resume_records or [])

        def checkpoint_state() -> None:
            if state_callback is None:
                return
            state_callback(
                {
                    "artifact_type": "egorse_exact_continuation_state_v1",
                    "seed": int(seed),
                    "budget": int(budget),
                    "x_values": np.asarray(x_values, dtype=float),
                    "y_values": np.asarray(y_values, dtype=float),
                    "enrichment_done": int(enrichment_done),
                    "eval_index": int(eval_index),
                    "best_valid": float(best_valid),
                    "best_any": float(best_any),
                    "next_outer_iter": int(outer_iter),
                    "next_method_index": int(method_index),
                    "next_sub_iter": int(sub_iter),
                    "active_embedding": active_embedding,
                    "rng_state": rng.bit_generator.state,
                    "elapsed_seconds": elapsed_offset + time.perf_counter() - start,
                    "history_records": list(history.records),
                }
            )

        checkpoint_state()
        outer_iterations = self.config.budget_spec.outer_iterations
        while enrichment_done < budget and outer_iter < outer_iterations:
            method = methods[method_index]
            if active_embedding is None:
                x_train = np.asarray(x_values, dtype=float)
                y_train = np.asarray(y_values, dtype=float)
                embedding = build_embedding(
                    method=method,
                    x_train=x_train,
                    y_train=y_train,
                    effective_dim=self.config.effective_dim,
                    rng=rng,
                    smt_reference_path=self.config.smt_reference_path,
                )
                bounds = embedding_box(embedding.matrix)
                u_sub = x_train @ embedding.matrix.T
                f_sub = y_train.copy()
                g_sub = np.asarray([positive_constraint_for_known_x(x) for x in x_train])
                active_embedding = {
                    "matrix": embedding.matrix,
                    "method": embedding.method,
                    "backend": embedding.backend,
                    "approximate": embedding.approximate,
                    "metadata": embedding.metadata,
                    "bounds": bounds,
                    "u_sub": u_sub,
                    "f_sub": f_sub,
                    "g_sub": g_sub,
                }
            matrix = np.asarray(active_embedding["matrix"], dtype=float)
            bounds = np.asarray(active_embedding["bounds"], dtype=float)
            u_sub = np.asarray(active_embedding["u_sub"], dtype=float)
            f_sub = np.asarray(active_embedding["f_sub"], dtype=float)
            g_sub = np.asarray(active_embedding["g_sub"], dtype=float)
            acquisition = propose_constrained_ei(
                bounds=bounds,
                u_train=u_sub,
                f_train=f_sub,
                g_train=g_sub,
                rng=rng,
                config=self.config.acquisition,
            )
            projected = evaluate_projected(
                matrix=matrix,
                u=acquisition.u,
                objective=problem.evaluate,
                tol=self.config.gamma_tol,
            )
            x_values.append(projected.x.copy())
            y_values.append(projected.f)
            u_sub = np.vstack([u_sub, projected.u])
            f_sub = np.append(f_sub, projected.f)
            g_sub = np.append(g_sub, projected.g)
            enrichment_done += 1
            eval_index += 1
            best_any = min(best_any, projected.f)
            if projected.feasible:
                best_valid = min(best_valid, projected.f)
            self._append_record(
                history,
                record_callback,
                self._record(
                    problem=problem,
                    eval_index=eval_index,
                    seed=seed,
                    x=projected.x,
                    u=projected.u,
                    f=projected.f,
                    feasibility=projected.feasible,
                    best_valid=best_valid,
                    best_any=best_any,
                    wall_time=elapsed_offset + time.perf_counter() - start,
                    phase="egorse_enrichment",
                    outer_iter=outer_iter + 1,
                    subspace_method=method,
                    subspace_eval_index=sub_iter + 1,
                    backend=acquisition.backend,
                    backend_is_approximate=True,
                    embedding_backend=str(active_embedding["backend"]),
                    embedding_approximate=bool(active_embedding["approximate"]),
                    gamma_status=projected.gamma_status,
                    gamma_backend=projected.gamma_backend,
                    g=projected.g,
                    embedding_metadata=dict(active_embedding["metadata"]),
                    acquisition_metadata=acquisition.metadata,
                    acquisition_value=acquisition.acquisition_value,
                ),
            )
            sub_iter += 1
            if sub_iter >= self.config.max_nb_it_sub:
                sub_iter = 0
                method_index += 1
                active_embedding = None
                if method_index >= len(methods):
                    method_index = 0
                    outer_iter += 1
            else:
                active_embedding = {
                    **active_embedding,
                    "u_sub": u_sub,
                    "f_sub": f_sub,
                    "g_sub": g_sub,
                }
            checkpoint_state()
        history.metadata["enrichment_evaluations_completed"] = enrichment_done
        history.metadata["total_evaluations_completed"] = eval_index
        return history

    def _append_record(
        self,
        history: OptimizationHistory,
        record_callback: Callable[[dict[str, object]], None] | None,
        record: dict[str, object],
    ) -> None:
        history.append(record)
        if record_callback is not None:
            record_callback(dict(record))

    def _record(
        self,
        problem: Any,
        eval_index: int,
        seed: int,
        x: np.ndarray,
        u: np.ndarray,
        f: float,
        feasibility: bool,
        best_valid: float,
        best_any: float,
        wall_time: float,
        phase: str,
        outer_iter: int,
        subspace_method: str,
        subspace_eval_index: int,
        backend: str,
        backend_is_approximate: bool,
        embedding_backend: str,
        embedding_approximate: bool,
        gamma_status: str,
        gamma_backend: str,
        g: float,
        embedding_metadata: dict[str, object] | None = None,
        acquisition_metadata: dict[str, object] | None = None,
        acquisition_value: float | None = None,
    ) -> dict[str, object]:
        return {
            "run_id": "",
            "seed": seed,
            "problem": problem.name,
            "d": problem.dim,
            "variant": self.config.variant,
            "doe_size": 0,
            "eval_index": eval_index,
            "phase": phase,
            "outer_iter": outer_iter,
            "subspace_method": subspace_method,
            "subspace_eval_index": subspace_eval_index,
            "x": np.asarray(x, dtype=float).tolist(),
            "u": np.asarray(problem.project(x), dtype=float).tolist(),
            "subspace_u": np.asarray(u, dtype=float).tolist(),
            "f": float(f),
            "g": float(g),
            "feasibility": bool(feasibility),
            "incumbent_best_valid_f": float(best_valid),
            "incumbent_best_f": float(best_any),
            "wall_time": float(wall_time),
            "backend": backend,
            "backend_is_approximate": bool(backend_is_approximate),
            "backend_mode": self.config.backend_mode,
            "backend_env_path": self.config.backend_env_path,
            "backend_exact": bool(self.config.backend_exact),
            "backend_fallback_reason": self.config.backend_fallback_reason,
            "backend_versions": dict(self.config.backend_versions),
            "backend_commits": dict(self.config.backend_commits),
            "embedding_backend": embedding_backend,
            "embedding_is_approximate": bool(embedding_approximate),
            "gamma_status": gamma_status,
            "gamma_backend": gamma_backend,
            "acquisition_value": acquisition_value,
            "embedding_metadata": embedding_metadata or {},
            "acquisition_metadata": acquisition_metadata or {},
        }
