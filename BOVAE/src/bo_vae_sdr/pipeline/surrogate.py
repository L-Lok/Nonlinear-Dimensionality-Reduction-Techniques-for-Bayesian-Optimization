"""VAE loading, benchmark construction, GP fitting, and acquisition optimization."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import torch
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_scipy, fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.core import OptimizationStatus
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from benchmarks import (
    Ackley,
    CanonicalLevy,
    CanonicalStyblinskiTang,
    CurvedPreimage,
    Rastrigin,
    Rosenbrock,
)
from benchmarks.curved_preimage import is_curved_preimage_key
from ..vae.artifacts import VAETrainingConfig, load_pretrained_vae
from .core import normalized_problem_key
from .config import PipelineConfig
from .transforms import assert_unit_box, assert_unit_tensor, tensor_to_list, unit_bounds

def load_configured_vae(
    config: PipelineConfig,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """Load the configured manuscript checkpoint."""

    return load_pretrained_vae(
        Path(config.vae_checkpoint),
        device=device,
        state_file=config.vae_state_file,
    )


def configured_vae_dimensions_match(
    config: PipelineConfig,
    vae_config: VAETrainingConfig,
) -> bool:
    """Validate checkpoint dimensions against the pipeline configuration."""

    if int(vae_config.ambient_dim) != int(config.dim):
        return False
    return int(vae_config.latent_dim) == int(config.latent_dim)


def make_problem(
    name: str,
    dim: int,
    bounds: torch.Tensor | None = None,
    *,
    curved_preimage_artifact_path: str | None = None,
    curved_preimage_metadata_path: str | None = None,
):
    key = normalized_problem_key(name)
    if key == "ackley":
        return Ackley(dim=dim, bounds=bounds)
    if key == "rosenbrock":
        return Rosenbrock(dim=dim, bounds=bounds)
    if key == "rastrigin":
        return Rastrigin(dim=dim, bounds=bounds)
    if key in {"canonical_levy", "levy_sfu"}:
        return CanonicalLevy(dim=dim, bounds=bounds)
    if key in {"canonical_styblinski_tang", "styblinski_tang_canonical"}:
        return CanonicalStyblinskiTang(dim=dim, bounds=bounds)
    if is_curved_preimage_key(key):
        if curved_preimage_artifact_path is None or curved_preimage_metadata_path is None:
            raise ValueError("curved-preimage benchmarks require artifact and metadata paths")
        return CurvedPreimage(
            dim=dim,
            bounds=bounds,
            artifact_path=curved_preimage_artifact_path,
            metadata_path=curved_preimage_metadata_path,
            expected_problem=key,
        )
    raise ValueError(f"Unsupported BO-VAE problem {name}")


def problem_bounds(config: PipelineConfig, dtype: torch.dtype) -> torch.Tensor | None:
    if config.problem_bound_lower is None and config.problem_bound_upper is None:
        return None
    if config.problem_bound_lower is None or config.problem_bound_upper is None:
        raise ValueError("problem_bound_lower and problem_bound_upper must be provided together")
    return torch.tensor(
        [[float(config.problem_bound_lower), float(config.problem_bound_upper)]] * config.dim,
        dtype=dtype,
    )


def rank_gaussian_transform(y: torch.Tensor) -> torch.Tensor:
    flat = y.reshape(-1)
    n = int(flat.numel())
    if n <= 1:
        return torch.zeros_like(y)
    order = torch.argsort(flat)
    ranks = torch.empty_like(order, dtype=y.dtype)
    ranks[order] = torch.arange(n, dtype=y.dtype, device=y.device)
    probs = (ranks + 0.5) / n
    normal = torch.distributions.Normal(
        loc=torch.tensor(0.0, dtype=y.dtype, device=y.device),
        scale=torch.tensor(1.0, dtype=y.dtype, device=y.device),
    )
    return normal.icdf(probs).reshape_as(y)


def transform_gp_targets(y_internal: torch.Tensor, mode: str) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode == "none":
        return y_internal, {"mode": "none"}
    if mode in {"rank_gaussian", "adaptive_rank_gaussian"}:
        transformed = rank_gaussian_transform(y_internal)
        return transformed, {
            "mode": mode,
            "effective_mode": "rank_gaussian",
            "raw_min": float(y_internal.min().item()),
            "raw_max": float(y_internal.max().item()),
            "transformed_min": float(transformed.min().item()),
            "transformed_max": float(transformed.max().item()),
        }
    raise ValueError(f"Unsupported GP target transform {mode}")



def _set_model_outputscale(model: SingleTaskGP, outputscale: float = 1.0) -> bool:
    covar = model.covar_module
    if not hasattr(covar, "outputscale"):
        return False
    covar.outputscale = torch.as_tensor(float(outputscale), device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    return True


def gp_training_diagnostics(train_x: torch.Tensor, train_y: torch.Tensor, *, duplicate_tol: float) -> dict[str, Any]:
    n_train, dim = int(train_x.shape[0]), int(train_x.shape[1])
    finite_x = bool(torch.isfinite(train_x).all().item())
    finite_y = bool(torch.isfinite(train_y).all().item())
    diag: dict[str, Any] = {
        "n_train": n_train,
        "dim": dim,
        "finite_x": finite_x,
        "finite_y": finite_y,
        "x_min": tensor_to_list(train_x.min(dim=0).values.reshape(-1)),
        "x_max": tensor_to_list(train_x.max(dim=0).values.reshape(-1)),
        "x_std": tensor_to_list(train_x.std(dim=0, unbiased=False).reshape(-1)),
        "x_at_lower_fraction": float((train_x <= duplicate_tol).to(dtype=torch.float64).mean().item()),
        "x_at_upper_fraction": float((train_x >= 1.0 - duplicate_tol).to(dtype=torch.float64).mean().item()),
        "target_min": float(train_y.min().item()),
        "target_max": float(train_y.max().item()),
        "target_std": float(train_y.std(unbiased=False).item()) if train_y.numel() > 1 else 0.0,
    }
    centered = train_x - train_x.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, n_train)
    eigvals = torch.linalg.eigvalsh(cov).detach().cpu()
    positive = eigvals[eigvals > 1e-14]
    diag.update(
        {
            "cov_eig_min": float(eigvals.min().item()) if eigvals.numel() else None,
            "cov_eig_max": float(eigvals.max().item()) if eigvals.numel() else None,
            "cov_effective_rank": int(positive.numel()),
            "cov_condition": float((positive.max() / positive.min()).item()) if positive.numel() else None,
        }
    )
    if n_train > 1:
        diff = train_x.unsqueeze(0) - train_x.unsqueeze(1)
        linf = torch.linalg.vector_norm(diff, ord=float("inf"), dim=-1)
        l2 = torch.linalg.vector_norm(diff, ord=2, dim=-1)
        upper = torch.triu(torch.ones(n_train, n_train, dtype=torch.bool, device=train_x.device), diagonal=1)
        linf_upper = linf[upper]
        l2_upper = l2[upper]
        duplicate_mask = linf_upper <= duplicate_tol
        near_mask = linf_upper <= max(duplicate_tol, 1e-6)
        positive_l2 = l2_upper[l2_upper > duplicate_tol]
        diag.update(
            {
                "duplicate_pairs_linf": int(duplicate_mask.sum().item()),
                "near_duplicate_pairs_linf_1e_minus_6": int(near_mask.sum().item()),
                "min_positive_l2": float(positive_l2.min().item()) if positive_l2.numel() else None,
                "median_l2": float(l2_upper.median().item()) if l2_upper.numel() else None,
            }
        )
    else:
        diag.update(
            {
                "duplicate_pairs_linf": 0,
                "near_duplicate_pairs_linf_1e_minus_6": 0,
                "min_positive_l2": None,
                "median_l2": None,
            }
        )
    return diag


def prepare_gp_training_data(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    train_y_var: float,
    duplicate_handling: str,
    duplicate_tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    raw_diag = gp_training_diagnostics(train_x, train_y, duplicate_tol=duplicate_tol)
    if duplicate_handling == "none":
        yvar = torch.full_like(train_y, float(train_y_var))
        return train_x, train_y, yvar, {
            "strategy": "none",
            "raw": raw_diag,
            "effective": raw_diag,
            "merged_groups": 0,
            "max_group_size": 1,
        }
    if duplicate_handling != "average_targets":
        raise ValueError(f"Unsupported duplicate handling strategy {duplicate_handling}")
    groups: dict[tuple[int, ...], list[int]] = {}
    scale = float(duplicate_tol)
    rounded = torch.round(train_x.detach().cpu().to(dtype=torch.float64) / scale).to(dtype=torch.int64)
    for index, row in enumerate(rounded.tolist()):
        groups.setdefault(tuple(int(item) for item in row), []).append(index)
    x_rows = []
    y_rows = []
    yvar_rows = []
    group_sizes = []
    for indices in groups.values():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=train_x.device)
        x_group = train_x.index_select(0, index_tensor)
        y_group = train_y.index_select(0, index_tensor)
        group_sizes.append(len(indices))
        x_rows.append(x_group.mean(dim=0, keepdim=True).clamp(0.0, 1.0))
        y_rows.append(y_group.mean(dim=0, keepdim=True))
        group_var = y_group.var(dim=0, unbiased=False, keepdim=True) if len(indices) > 1 else torch.zeros_like(y_rows[-1])
        yvar_rows.append(torch.clamp(group_var + float(train_y_var), min=float(train_y_var)))
    effective_x = torch.cat(x_rows, dim=0)
    effective_y = torch.cat(y_rows, dim=0)
    effective_yvar = torch.cat(yvar_rows, dim=0)
    effective_diag = gp_training_diagnostics(effective_x, effective_y, duplicate_tol=duplicate_tol)
    merged = sum(1 for size in group_sizes if size > 1)
    return effective_x, effective_y, effective_yvar, {
        "strategy": "average_targets",
        "raw": raw_diag,
        "effective": effective_diag,
        "merged_groups": int(merged),
        "max_group_size": int(max(group_sizes) if group_sizes else 0),
        "n_raw": int(train_x.shape[0]),
        "n_effective": int(effective_x.shape[0]),
        "yvar_min": float(effective_yvar.min().item()),
        "yvar_max": float(effective_yvar.max().item()),
    }


def warning_summary(warnings_list: list[str]) -> dict[str, Any]:
    counts = {
        "total": len(warnings_list),
        "deprecation": 0,
        "numerical_jitter": 0,
        "optimization_failure": 0,
        "model_fitting_error": 0,
        "input_data": 0,
        "other": 0,
    }
    for warning in warnings_list:
        lower = str(warning).lower()
        matched = False
        if "deprecated" in lower or "`disp`" in lower or "disp and iprint" in lower or "iprint" in lower:
            counts["deprecation"] += 1
            matched = True
        if "jitter" in lower or "not p.d." in lower:
            counts["numerical_jitter"] += 1
            matched = True
        if "optimizationstatus.failure" in lower or "abnormal" in lower or "line search" in lower:
            counts["optimization_failure"] += 1
            matched = True
        if "modelfittingerror" in lower:
            counts["model_fitting_error"] += 1
            matched = True
        if "input data" in lower:
            counts["input_data"] += 1
            matched = True
        if not matched:
            counts["other"] += 1
    return counts


def optimization_result_metadata(result: Any) -> dict[str, Any]:
    status = getattr(result, "status", None)
    runtime = getattr(result, "runtime", None)
    return {
        "status": getattr(status, "name", str(status)),
        "message": str(getattr(result, "message", "")),
        "fval": float(getattr(result, "fval")) if getattr(result, "fval", None) is not None else None,
        "runtime": float(runtime) if runtime is not None else None,
        "step": int(getattr(result, "step")) if getattr(result, "step", None) is not None else None,
    }


def fit_gp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_y_var: float,
    maxiter: int,
    *,
    kernel: str = "matern52",
    fit_strategy: str = "botorch_default",
    duplicate_handling: str = "none",
    duplicate_tol: float = 1e-10,
    matern_use_scale_kernel: bool = True,
    lengthscale_lower_bound: float = 1e-3,
    lengthscale_upper_bound: float = 2.0,
    outputscale_lower_bound: float = 1e-4,
    outputscale_upper_bound: float = 1e4,
    train_y_var_floor: float = 1e-6,
    torch_fallback_steps: int = 75,
    torch_fallback_lr: float = 0.05,
) -> tuple[SingleTaskGP, dict[str, Any]]:
    if train_x.ndim != 2 or train_y.ndim != 2:
        raise ValueError("GP train_x and train_y must be rank-2 tensors")
    assert_unit_box(unit_bounds(train_x.shape[1], device=train_x.device, dtype=train_x.dtype), name="unit reference")
    assert_unit_tensor(train_x, name="BO-VAE GP train_x")
    if not torch.isfinite(train_x).all() or not torch.isfinite(train_y).all():
        raise ValueError("GP train_x and train_y must be finite")
    effective_train_y_var = max(float(train_y_var), float(train_y_var_floor))
    train_x_fit, train_y_fit, train_yvar_fit, duplicate_info = prepare_gp_training_data(
        train_x,
        train_y,
        train_y_var=effective_train_y_var,
        duplicate_handling=duplicate_handling,
        duplicate_tol=duplicate_tol,
    )
    if kernel != "matern52":
        raise ValueError("the publication BO-VAE surrogate uses Matérn-5/2")
    base_kernel = MaternKernel(
        nu=2.5,
        ard_num_dims=train_x_fit.shape[1],
        lengthscale_constraint=Interval(
            float(lengthscale_lower_bound), float(lengthscale_upper_bound)
        ),
    )
    covar_module = (
        ScaleKernel(
            base_kernel,
            outputscale_constraint=Interval(
                float(outputscale_lower_bound), float(outputscale_upper_bound)
            ),
        )
        if matern_use_scale_kernel
        else base_kernel
    )
    model = SingleTaskGP(
        train_X=train_x_fit,
        train_Y=train_y_fit,
        train_Yvar=train_yvar_fit,
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
    ).to(train_x_fit)
    outputscale_initialized = _set_model_outputscale(model, 1.0)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_x_fit)
    start = time.time()
    status = "ok"
    caught_warnings: list[str] = []
    fit_attempts: list[dict[str, Any]] = []

    def record_caught(caught: list[warnings.WarningMessage]) -> None:
        caught_warnings.extend(f"{item.category.__name__}: {item.message}" for item in caught)

    def fit_scipy(attempt_name: str) -> bool:
        model.train()
        mll.train()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = fit_gpytorch_mll_scipy(
                    mll,
                    options={"maxiter": int(maxiter)},
                )
                metadata = optimization_result_metadata(result)
                fit_attempts.append({"name": attempt_name, "optimizer": "scipy_l_bfgs_b", **metadata})
                record_caught(caught)
                return result.status in {OptimizationStatus.SUCCESS, OptimizationStatus.STOPPED}
            except Exception as exc:
                caught_warnings.append(repr(exc))
                fit_attempts.append({"name": attempt_name, "optimizer": "scipy_l_bfgs_b", "status": "EXCEPTION", "message": repr(exc)})
                record_caught(caught)
                return False

    def fit_torch(attempt_name: str) -> bool:
        model.train()
        mll.train()

        def optimizer_factory(params: Any) -> torch.optim.Optimizer:
            return torch.optim.Adam(params, lr=float(torch_fallback_lr))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = fit_gpytorch_mll_torch(
                    mll,
                    step_limit=int(torch_fallback_steps),
                    optimizer=optimizer_factory,
                )
                metadata = optimization_result_metadata(result)
                fit_attempts.append({"name": attempt_name, "optimizer": "torch_adam", **metadata})
                record_caught(caught)
                return result.status in {OptimizationStatus.SUCCESS, OptimizationStatus.STOPPED}
            except Exception as exc:
                caught_warnings.append(repr(exc))
                fit_attempts.append({"name": attempt_name, "optimizer": "torch_adam", "status": "EXCEPTION", "message": repr(exc)})
                record_caught(caught)
                return False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if fit_strategy == "botorch_default":
            try:
                fit_gpytorch_mll(
                    mll,
                    optimizer_kwargs={"options": {"maxiter": int(maxiter)}},
                )
                fit_attempts.append({"name": "botorch_default", "optimizer": "fit_gpytorch_mll", "status": "SUCCESS"})
            except TypeError:
                fit_gpytorch_mll(mll)
                fit_attempts.append({"name": "botorch_default_typeerror_fallback", "optimizer": "fit_gpytorch_mll", "status": "SUCCESS"})
            except Exception as exc:
                status = "exception"
                caught_warnings.append(repr(exc))
                fit_attempts.append({"name": "botorch_default", "optimizer": "fit_gpytorch_mll", "status": "EXCEPTION", "message": repr(exc)})
            record_caught(caught)
        elif fit_strategy == "scipy":
            if not fit_scipy("scipy"):
                status = "optimizer_failure"
            record_caught(caught)
        elif fit_strategy == "scipy_then_torch":
            scipy_ok = fit_scipy("scipy")
            if scipy_ok:
                status = "ok"
            else:
                torch_ok = fit_torch("torch_fallback")
                status = "ok_after_torch_fallback" if torch_ok else "optimizer_failure"
            record_caught(caught)
        else:
            raise ValueError(f"Unsupported GP fit strategy {fit_strategy}")
    model.eval()
    warning_counts = warning_summary(caught_warnings)
    return model, {
        "status": status,
        "elapsed_sec": time.time() - start,
        "warnings": caught_warnings,
        "warning_summary": warning_counts,
        "fit_strategy": fit_strategy,
        "fit_attempts": fit_attempts,
        "n_train": int(train_x.shape[0]),
        "n_train_effective": int(train_x_fit.shape[0]),
        "dim": int(train_x.shape[1]),
        "kernel": kernel,
        "matern_use_scale_kernel": bool(matern_use_scale_kernel) if kernel == "matern52" else None,
        "outputscale_initialized": bool(outputscale_initialized),
        "effective_train_y_var": float(effective_train_y_var),
        "train_yvar_min": float(train_yvar_fit.min().item()),
        "train_yvar_max": float(train_yvar_fit.max().item()),
        "duplicate_handling": duplicate_info,
    }


def acquisition_values(
    model: SingleTaskGP,
    candidates: torch.Tensor,
    *,
    acquisition: str,
    best_f: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the configured BO-VAE acquisition on unit-space candidates."""

    acq_cls = LogExpectedImprovement if acquisition == "logei" else ExpectedImprovement
    acq = acq_cls(model, best_f=best_f).to(candidates)
    return acq(candidates.unsqueeze(1)).reshape(-1)


def optimize_acquisition(
    model: SingleTaskGP,
    bounds: torch.Tensor,
    *,
    acquisition: str,
    best_f: torch.Tensor,
    warmup: int,
    raw_samples: int,
    num_restarts: int,
    maxiter: int,
    timeout_sec: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    assert_unit_box(bounds, name="BO-VAE acquisition bounds")
    bounds = bounds.to(next(model.parameters()))
    acq_cls = LogExpectedImprovement if acquisition == "logei" else ExpectedImprovement
    acq = acq_cls(model, best_f=best_f).to(bounds)
    x_tries = torch.rand(int(warmup), bounds.shape[0], dtype=bounds.dtype, device=bounds.device)
    x_tries = x_tries * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    with torch.no_grad():
        warmup_values = acq(x_tries.unsqueeze(1)).reshape(-1)
    warmup_index = int(torch.argmax(warmup_values).item())
    warmup_x = x_tries[warmup_index : warmup_index + 1]
    warmup_acq = warmup_values[warmup_index].reshape(1)
    start = time.time()
    status = "ok"
    warning_text: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            opt_x, opt_acq = optimize_acqf(
                acq_function=acq,
                bounds=bounds.T.contiguous(),
                q=1,
                num_restarts=int(num_restarts),
                raw_samples=int(raw_samples),
                options={"maxiter": int(maxiter), "sample_around_best": True},
                timeout_sec=float(timeout_sec),
                return_best_only=True,
            )
        except Exception as exc:
            opt_x = warmup_x
            opt_acq = warmup_acq
            status = "exception_using_warmup"
            warning_text.append(repr(exc))
        warning_text.extend(str(item.message) for item in caught)
    if warmup_acq.reshape(-1)[0] > opt_acq.reshape(-1)[0]:
        candidate = warmup_x
        value = warmup_acq
        selected = "warmup"
    else:
        candidate = opt_x
        value = opt_acq.reshape(1)
        selected = "optimize_acqf"
    return candidate.detach(), value.detach(), {
        "acquisition": acquisition,
        "acquisition_class": acq_cls.__name__,
        "bounds": tensor_to_list(bounds),
        "best_f": float(best_f.detach().reshape(-1)[0].item()),
        "warmup_best_candidate": tensor_to_list(warmup_x),
        "warmup_best_acq": float(warmup_acq.reshape(-1)[0].item()),
        "optimized_candidate": tensor_to_list(opt_x),
        "optimized_acq": float(opt_acq.reshape(-1)[0].item()),
        "selected_source": selected,
        "selected_acq": float(value.reshape(-1)[0].item()),
        "status": status,
        "warnings": warning_text,
        "elapsed_sec": time.time() - start,
    }


__all__ = [name for name in globals() if not name.startswith("__")]
