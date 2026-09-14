"""Approximate constrained-EI backend for EGORSE.

The paper used SEGO/SEGOMOE with NLOPT ISRES and PyOptSparse/SNOPT
(Section V.A). SBArchOpt provides a public SEGOMOE interface, but not the
SEGOMOE runtime itself; SNOPT is also unavailable here. This module is
therefore deliberately labelled approximate and only used when the config flag
allows it. The faithful reconstruction path fits SMT KRG surrogates for
f^(t) and g^(t), then optimizes constrained EI over B with the verified NLOPT
ISRES global start and an available local refiner.
"""

from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


SMT_DEFAULT_NUGGET = 2.220446049250313e-14
DEFAULT_SMT_NUGGET_SEQUENCE = (
    SMT_DEFAULT_NUGGET,
    1e-13,
    1e-12,
    1e-11,
    1e-10,
    1e-8,
)


@dataclass(frozen=True)
class AcquisitionResult:
    """Candidate returned by the approximate constrained-EI backend."""

    u: np.ndarray
    acquisition_value: float
    backend: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class ApproximateConstrainedEIConfig:
    """Settings for the labelled approximate backend."""

    n_candidates: int = 1024
    n_local_starts: int = 4
    gp_restarts: int = 1
    constraint_strategy: str = "mean_constraint"
    global_optimizer: str = "nlopt_isres"
    isres_maxeval: int = 256
    local_refiner: str = "scipy_slsqp"
    surrogate_backend: str = "smt_krg"
    smt_theta0: float = 1e-3
    smt_n_start: int = 1
    smt_nugget_sequence: tuple[float, ...] = DEFAULT_SMT_NUGGET_SEQUENCE
    duplicate_tolerance: float = 1e-8


@dataclass(frozen=True)
class _FittedSurrogate:
    """Tiny adapter over SMT or sklearn GP models."""

    model: object
    backend: str
    metadata: dict[str, object]

    def predict(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points_arr = np.asarray(points, dtype=float)
        if self.backend.startswith("smt."):
            mu = np.asarray(self.model.predict_values(points_arr), dtype=float).reshape(-1)
            variance = np.asarray(self.model.predict_variances(points_arr), dtype=float).reshape(-1)
            return mu, np.sqrt(np.maximum(variance, 1e-18))
        mu, std = self.model.predict(points_arr, return_std=True)
        return np.asarray(mu, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1)


def propose_constrained_ei(
    bounds: np.ndarray,
    u_train: np.ndarray,
    f_train: np.ndarray,
    g_train: np.ndarray,
    rng: np.random.Generator,
    config: ApproximateConstrainedEIConfig,
) -> AcquisitionResult:
    """Propose the next u in B using approximate constrained EI."""

    bounds_arr = np.asarray(bounds, dtype=float)
    dim = bounds_arr.shape[0]
    u_arr = np.asarray(u_train, dtype=float)
    f_arr = np.asarray(f_train, dtype=float).reshape(-1)
    g_arr = np.asarray(g_train, dtype=float).reshape(-1)
    feasible = g_arr >= 0.0
    best_f = float(np.min(f_arr[feasible])) if np.any(feasible) else float(np.min(f_arr))

    f_gp = _fit_surrogate(
        u_arr,
        f_arr,
        dim,
        int(rng.integers(0, 2**31 - 1)),
        config,
        role="objective",
    )
    g_gp = _fit_surrogate(
        u_arr,
        g_arr,
        dim,
        int(rng.integers(0, 2**31 - 1)),
        config,
        role="constraint",
    )

    def predictions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        points_arr = np.asarray(points, dtype=float)
        mu_f, std_f = f_gp.predict(points_arr)
        mu_g, std_g = g_gp.predict(points_arr)
        ei = expected_improvement_min(mu_f, std_f, best_f)
        return mu_f, std_f, mu_g, std_g, ei

    def acquisition(points: np.ndarray) -> np.ndarray:
        points_arr = np.asarray(points, dtype=float)
        _, _, mu_g, std_g, ei = predictions(points_arr)
        if config.constraint_strategy == "mean":
            feasibility_weight = (mu_g >= 0.0).astype(float)
        elif config.constraint_strategy == "mean_constraint":
            feasibility_weight = (mu_g >= 0.0).astype(float)
        else:
            feasibility_weight = norm.cdf(mu_g / np.maximum(std_g, 1e-12))
        return ei * feasibility_weight * novelty_weight(points_arr)

    def novelty_weight(points: np.ndarray) -> np.ndarray:
        points_arr = np.asarray(points, dtype=float)
        if points_arr.size == 0:
            return np.asarray([], dtype=float)
        distances = np.linalg.norm(points_arr[:, None, :] - u_arr[None, :, :], axis=2)
        nearest = np.min(distances, axis=1)
        return (nearest > float(config.duplicate_tolerance)).astype(float)

    def constraint_mean(u: np.ndarray) -> float:
        _, _, mu_g, _, _ = predictions(np.asarray(u, dtype=float).reshape(1, -1))
        return float(mu_g[0])

    random_points = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1], size=(config.n_candidates, dim))
    center = np.mean(bounds_arr, axis=1, keepdims=True).T
    candidates = np.vstack([random_points, center])
    values = acquisition(candidates)
    order = np.argsort(values)[::-1]
    best_u = candidates[order[0]].copy()
    best_value = float(values[order[0]])
    metadata: dict[str, object] = {
        "best_f_for_ei": best_f,
        "random_candidates": int(config.n_candidates),
        "constraint_strategy": config.constraint_strategy,
        "surrogate_backend_requested": config.surrogate_backend,
        "duplicate_tolerance": float(config.duplicate_tolerance),
        "f_surrogate_backend": f_gp.backend,
        "g_surrogate_backend": g_gp.backend,
        "f_surrogate_metadata": f_gp.metadata,
        "g_surrogate_metadata": g_gp.metadata,
        "global_optimizer": config.global_optimizer,
        "isres_requested": config.global_optimizer == "nlopt_isres",
        "isres_maxeval": int(config.isres_maxeval),
        "isres_available": False,
        "isres_used": False,
        "isres_result_code": None,
        "isres_error": None,
        "local_refiner": config.local_refiner,
    }

    if config.global_optimizer == "nlopt_isres":
        isres_result = _nlopt_isres_start(
            bounds_arr=bounds_arr,
            acquisition=acquisition,
            constraint=constraint_mean if config.constraint_strategy == "mean_constraint" else None,
            x0=best_u,
            seed=int(rng.integers(0, 2**31 - 1)),
            maxeval=int(config.isres_maxeval),
        )
        metadata.update(isres_result["metadata"])
        if isres_result["u"] is not None and isres_result["value"] > best_value:
            best_u = np.asarray(isres_result["u"], dtype=float)
            best_value = float(isres_result["value"])
    elif config.global_optimizer != "random":
        metadata["global_optimizer_warning"] = (
            f"unknown global optimizer {config.global_optimizer!r}; used random screen"
        )

    local_attempts = 0
    if config.local_refiner == "none":
        local_order = []
    else:
        local_order = order[: max(0, config.n_local_starts)]
    for idx in local_order:
        local_attempts += 1

        def objective(u: np.ndarray) -> float:
            return -float(acquisition(np.asarray(u, dtype=float).reshape(1, -1))[0])

        method = _scipy_method(config.local_refiner)
        minimize_kwargs: dict[str, object] = {
            "fun": objective,
            "x0": candidates[idx],
            "method": method,
            "bounds": [tuple(row) for row in bounds_arr],
            "options": {"maxiter": 80, "ftol": 1e-9},
        }
        if method == "SLSQP" and config.constraint_strategy == "mean_constraint":
            minimize_kwargs["constraints"] = {
                "type": "ineq",
                "fun": lambda u: constraint_mean(np.asarray(u, dtype=float)),
            }
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step.*",
                category=RuntimeWarning,
            )
            result = minimize(**minimize_kwargs)
        if result.success:
            value = -float(result.fun)
            if value > best_value:
                best_value = value
                best_u = np.asarray(result.x, dtype=float)
    metadata["local_attempts"] = local_attempts
    backend = f"approximate_{_backend_slug(f_gp.backend)}_constrained_ei"
    if metadata["isres_used"]:
        backend = f"approximate_{_backend_slug(f_gp.backend)}_nlopt_isres_constrained_ei"

    return AcquisitionResult(
        u=np.clip(best_u, bounds_arr[:, 0], bounds_arr[:, 1]),
        acquisition_value=best_value,
        backend=backend,
        metadata=metadata,
    )


def expected_improvement_min(mu: np.ndarray, sigma: np.ndarray, best_f: float) -> np.ndarray:
    """Expected improvement for minimization."""

    sigma_safe = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    improvement = best_f - np.asarray(mu, dtype=float)
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _nlopt_isres_start(
    bounds_arr: np.ndarray,
    acquisition,
    constraint,
    x0: np.ndarray,
    seed: int,
    maxeval: int,
) -> dict[str, object]:
    """Run the paper-style NLOPT ISRES global start on the approximate EI."""

    metadata = {
        "isres_available": False,
        "isres_used": False,
        "isres_result_code": None,
        "isres_error": None,
    }
    try:
        import nlopt
    except Exception as exc:
        metadata["isres_error"] = f"{type(exc).__name__}: {exc}"
        return {"u": None, "value": float("-inf"), "metadata": metadata}

    metadata["isres_available"] = True
    try:
        nlopt.srand(int(seed))
        optimizer = nlopt.opt(nlopt.GN_ISRES, int(bounds_arr.shape[0]))
        optimizer.set_lower_bounds(bounds_arr[:, 0].tolist())
        optimizer.set_upper_bounds(bounds_arr[:, 1].tolist())
        optimizer.set_maxeval(max(1, int(maxeval)))

        def objective(u: np.ndarray, grad: np.ndarray) -> float:
            return -float(acquisition(np.asarray(u, dtype=float).reshape(1, -1))[0])

        optimizer.set_min_objective(objective)
        if constraint is not None:
            optimizer.add_inequality_constraint(
                lambda u, grad: -float(constraint(np.asarray(u, dtype=float))),
                1e-8,
            )
        start = np.clip(np.asarray(x0, dtype=float), bounds_arr[:, 0], bounds_arr[:, 1])
        candidate = optimizer.optimize(start)
        value = -float(optimizer.last_optimum_value())
        metadata["isres_result_code"] = int(optimizer.last_optimize_result())
        metadata["isres_used"] = bool(np.all(np.isfinite(candidate)) and np.isfinite(value))
        if not metadata["isres_used"]:
            return {"u": None, "value": float("-inf"), "metadata": metadata}
        return {"u": candidate, "value": value, "metadata": metadata}
    except Exception as exc:
        metadata["isres_error"] = f"{type(exc).__name__}: {exc}"
        return {"u": None, "value": float("-inf"), "metadata": metadata}


def _fit_surrogate(
    x: np.ndarray,
    y: np.ndarray,
    dim: int,
    seed: int,
    config: ApproximateConstrainedEIConfig,
    role: str,
) -> _FittedSurrogate:
    requested = config.surrogate_backend
    x_fit, y_fit, duplicates_removed = _deduplicate_training_rows(x, y)
    if requested == "smt_krg":
        smt_result = _fit_smt_krg_with_nugget_retries(x_fit, y_fit, dim, seed, config)
        if smt_result is not None:
            surrogate, attempts = smt_result
            metadata = dict(surrogate.metadata)
            metadata.update(
                {
                    "role": role,
                    "theta0": float(config.smt_theta0),
                    "n_start": max(1, int(config.smt_n_start)),
                    "training_rows": int(np.asarray(x, dtype=float).shape[0]),
                    "fit_rows": int(x_fit.shape[0]),
                    "duplicates_removed": int(duplicates_removed),
                    "smt_nugget_attempts": attempts,
                }
            )
            return _FittedSurrogate(
                model=surrogate.model,
                backend=surrogate.backend,
                metadata=metadata,
            )
        sklearn = _fit_sklearn_gp(x_fit, y_fit, dim, seed, config.gp_restarts)
        metadata = dict(sklearn.metadata)
        attempts = _fit_smt_krg_with_nugget_retries.last_attempts
        fallback_reason = attempts[-1]["error"] if attempts else "unknown SMT KRG failure"
        metadata.update(
            {
                "role": role,
                "requested_backend": requested,
                "fallback_reason": fallback_reason,
                "smt_nugget_attempts": attempts,
                "training_rows": int(np.asarray(x, dtype=float).shape[0]),
                "fit_rows": int(x_fit.shape[0]),
                "duplicates_removed": int(duplicates_removed),
            }
        )
        return _FittedSurrogate(
            model=sklearn.model,
            backend="sklearn.GaussianProcessRegressor_after_smt_failure",
            metadata=metadata,
        )
    if requested != "sklearn_gp":
        sklearn = _fit_sklearn_gp(x_fit, y_fit, dim, seed, config.gp_restarts)
        metadata = dict(sklearn.metadata)
        metadata.update(
            {
                "role": role,
                "unknown_requested_backend": requested,
                "training_rows": int(np.asarray(x, dtype=float).shape[0]),
                "fit_rows": int(x_fit.shape[0]),
                "duplicates_removed": int(duplicates_removed),
            }
        )
        return _FittedSurrogate(
            model=sklearn.model,
            backend="sklearn.GaussianProcessRegressor_after_unknown_backend",
            metadata=metadata,
        )
    sklearn = _fit_sklearn_gp(x_fit, y_fit, dim, seed, config.gp_restarts)
    metadata = dict(sklearn.metadata)
    metadata["role"] = role
    metadata["training_rows"] = int(np.asarray(x, dtype=float).shape[0])
    metadata["fit_rows"] = int(x_fit.shape[0])
    metadata["duplicates_removed"] = int(duplicates_removed)
    return _FittedSurrogate(model=sklearn.model, backend=sklearn.backend, metadata=metadata)


def _fit_smt_krg_with_nugget_retries(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    dim: int,
    seed: int,
    config: ApproximateConstrainedEIConfig,
) -> tuple[_FittedSurrogate, list[dict[str, object]]] | None:
    """Fit SMT KRG, increasing only the nugget when SMT hits ill-conditioning.

    SMT KRG sometimes raises ``KeyError: 'C'`` after a failed Cholesky path
    leaves ``optimal_par`` without the Cholesky factor. This retry ladder keeps
    the intended SMT KRG backend and uses the smallest configured jitter that
    yields a usable model.
    """

    try:
        from smt.surrogate_models import KRG
    except Exception as exc:
        attempts = [
            {
                "nugget": None,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]
        _fit_smt_krg_with_nugget_retries.last_attempts = attempts
        return None

    attempts: list[dict[str, object]] = []
    for nugget in _validated_smt_nugget_sequence(config.smt_nugget_sequence):
        try:
            model = KRG(
                print_global=False,
                theta0=[float(config.smt_theta0)] * dim,
                n_start=max(1, int(config.smt_n_start)),
                corr="squar_exp",
                poly="constant",
                random_state=seed,
                nugget=float(nugget),
            )
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                warnings.filterwarnings(
                    "ignore",
                    message="Warning: multiple x input features have the same value.*",
                    category=UserWarning,
                )
                model.set_training_values(x_fit, y_fit)
                model.train()
                _validate_smt_krg_model(model, x_fit)
            attempts.append(
                {
                    "nugget": float(nugget),
                    "status": "success",
                }
            )
            return (
                _FittedSurrogate(
                    model=model,
                    backend="smt.KRG",
                    metadata={
                        "smt_nugget": float(nugget),
                        "smt_nugget_retries": len(attempts) - 1,
                    },
                ),
                attempts,
            )
        except Exception as exc:
            attempts.append(
                {
                    "nugget": float(nugget),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    _fit_smt_krg_with_nugget_retries.last_attempts = attempts
    return None


_fit_smt_krg_with_nugget_retries.last_attempts = []


def _validated_smt_nugget_sequence(values: tuple[float, ...]) -> tuple[float, ...]:
    nuggets: list[float] = []
    for value in values:
        nugget = float(value)
        if not np.isfinite(nugget) or nugget < 0.0:
            continue
        if nugget not in nuggets:
            nuggets.append(nugget)
    return tuple(nuggets) or (SMT_DEFAULT_NUGGET,)


def _validate_smt_krg_model(model: object, x_fit: np.ndarray) -> None:
    optimal_par = getattr(model, "optimal_par", {})
    if "C" not in optimal_par:
        raise KeyError("C")
    probe = np.asarray(x_fit, dtype=float)[:1]
    model.predict_values(probe)
    model.predict_variances(probe)


def _fit_sklearn_gp(
    x: np.ndarray,
    y: np.ndarray,
    dim: int,
    seed: int,
    restarts: int,
) -> _FittedSurrogate:
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-3, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-3))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        random_state=seed,
        n_restarts_optimizer=max(0, restarts),
    )
    model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float).reshape(-1))
    return _FittedSurrogate(
        model=model,
        backend="sklearn.GaussianProcessRegressor",
        metadata={"gp_restarts": max(0, restarts)},
    )


def _scipy_method(local_refiner: str) -> str:
    if local_refiner.lower() in {"slsqp", "scipy_slsqp"}:
        return "SLSQP"
    return "L-BFGS-B"


def _backend_slug(backend: str) -> str:
    if backend == "smt.KRG":
        return "smt_krg"
    if backend.startswith("sklearn."):
        return "sklearn_gp"
    return backend.lower().replace(".", "_")


def _deduplicate_training_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    unique_x, inverse = np.unique(x_arr, axis=0, return_inverse=True)
    if unique_x.shape[0] == x_arr.shape[0]:
        return x_arr, y_arr, 0
    summed = np.zeros(unique_x.shape[0], dtype=float)
    counts = np.zeros(unique_x.shape[0], dtype=float)
    for idx, value in zip(inverse, y_arr):
        summed[idx] += float(value)
        counts[idx] += 1.0
    return unique_x, summed / np.maximum(counts, 1.0), int(x_arr.shape[0] - unique_x.shape[0])
