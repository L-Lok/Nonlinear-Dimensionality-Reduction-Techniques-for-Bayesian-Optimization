"""Transfer matrix builders for the six EGORSE variants in Section V.B."""

from __future__ import annotations

import contextlib
import io
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


EmbeddingMethod = Literal["gaussian", "hash", "pls", "mgp"]


VARIANT_METHODS: dict[str, tuple[EmbeddingMethod, ...]] = {
    "gaussian": ("gaussian",),
    "hash": ("hash",),
    "pls": ("pls",),
    "pls_gaussian": ("pls", "gaussian"),
    "mgp": ("mgp",),
    "mgp_gaussian": ("mgp", "gaussian"),
}

VARIANT_DISPLAY_NAMES: dict[str, str] = {
    "gaussian": "EGORSE Gaussian",
    "hash": "EGORSE Hash",
    "pls": "EGORSE PLS",
    "pls_gaussian": "EGORSE PLS + Gaussian",
    "mgp": "EGORSE MGP",
    "mgp_gaussian": "EGORSE MGP + Gaussian",
}


@dataclass(frozen=True)
class EmbeddingBuildResult:
    """Transfer matrix and provenance for one Algorithm 2 subspace."""

    matrix: np.ndarray
    method: str
    backend: str
    approximate: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


def methods_for_variant(variant: str) -> tuple[EmbeddingMethod, ...]:
    """Return the dimension-reduction methods used by a Section V.B variant."""

    try:
        return VARIANT_METHODS[variant]
    except KeyError as exc:
        valid = ", ".join(sorted(VARIANT_METHODS))
        raise ValueError(f"unknown EGORSE variant {variant!r}; valid variants: {valid}") from exc


def build_embedding(
    method: EmbeddingMethod,
    x_train: np.ndarray,
    y_train: np.ndarray,
    effective_dim: int,
    rng: np.random.Generator,
    smt_reference_path: str | None = None,
) -> EmbeddingBuildResult:
    """Build A^(t) in Algorithm 2 line 3 for one reduction method."""

    x_arr = np.asarray(x_train, dtype=float)
    y_arr = np.asarray(y_train, dtype=float).reshape(-1)
    if x_arr.ndim != 2:
        raise ValueError("x_train must be two-dimensional")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x_train and y_train length mismatch")
    if method == "gaussian":
        return _gaussian_embedding(x_arr.shape[1], effective_dim, rng)
    if method == "hash":
        return _hash_embedding(x_arr.shape[1], effective_dim, rng)
    if method == "pls":
        return _pls_embedding(x_arr, y_arr, effective_dim, rng)
    if method == "mgp":
        return _mgp_embedding(x_arr, y_arr, effective_dim, rng, smt_reference_path)
    raise ValueError(f"unsupported embedding method {method!r}")


def _gaussian_embedding(
    dim: int,
    effective_dim: int,
    rng: np.random.Generator,
) -> EmbeddingBuildResult:
    matrix = rng.normal(size=(effective_dim, dim))
    matrix = _repair_degenerate_rows(matrix, rng)
    return EmbeddingBuildResult(
        matrix=matrix,
        method="gaussian",
        backend="numpy_normal",
        metadata={"distribution": "standard_normal"},
    )


def _hash_embedding(
    dim: int,
    effective_dim: int,
    rng: np.random.Generator,
) -> EmbeddingBuildResult:
    matrix = np.zeros((effective_dim, dim), dtype=float)
    rows = rng.integers(0, effective_dim, size=dim)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=dim)
    matrix[rows, np.arange(dim)] = signs
    matrix = _repair_degenerate_rows(matrix, rng)
    return EmbeddingBuildResult(
        matrix=matrix,
        method="hash",
        backend="hesbo_style_hash",
        metadata={"nonzeros_per_column": 1},
    )


def _pls_embedding(
    x_train: np.ndarray,
    y_train: np.ndarray,
    effective_dim: int,
    rng: np.random.Generator,
) -> EmbeddingBuildResult:
    try:
        from sklearn.cross_decomposition import PLSRegression

        n_components = min(effective_dim, x_train.shape[1], max(1, x_train.shape[0] - 1))
        if np.unique(y_train).size < 2:
            raise ValueError("PLS requires non-constant y values")
        pls = PLSRegression(n_components=n_components, scale=True)
        pls.fit(x_train, y_train.reshape(-1, 1))
        matrix = np.asarray(pls.x_rotations_, dtype=float).T
        if matrix.shape[0] < effective_dim:
            extra = rng.normal(size=(effective_dim - matrix.shape[0], x_train.shape[1]))
            matrix = np.vstack([matrix, extra])
        matrix = _orthonormalize_rows(matrix[:effective_dim], rng)
        return EmbeddingBuildResult(
            matrix=matrix,
            method="pls",
            backend="sklearn.cross_decomposition.PLSRegression",
            metadata={"n_components_fit": n_components},
        )
    except Exception as exc:
        fallback = _gaussian_embedding(x_train.shape[1], effective_dim, rng)
        return EmbeddingBuildResult(
            matrix=fallback.matrix,
            method="pls",
            backend="fallback_gaussian_after_pls_failure",
            approximate=True,
            metadata={"fallback_reason": f"{type(exc).__name__}: {exc}"},
        )


def _mgp_embedding(
    x_train: np.ndarray,
    y_train: np.ndarray,
    effective_dim: int,
    rng: np.random.Generator,
    smt_reference_path: str | None,
) -> EmbeddingBuildResult:
    if smt_reference_path:
        path = str(Path(smt_reference_path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from smt.surrogate_models import MGP

        model = MGP(theta0=[1e-2], print_global=False, n_comp=effective_dim)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            warnings.filterwarnings(
                "ignore",
                message="Warning: multiple x input features have the same value.*",
                category=UserWarning,
            )
            model.set_training_values(x_train, y_train)
            model.train()
        matrix = np.asarray(model.embedding["C"], dtype=float).T
        matrix = _orthonormalize_rows(matrix[:effective_dim], rng)
        best_ncomp = _metadata_scalar(getattr(model, "best_ncomp", effective_dim), default=effective_dim)
        return EmbeddingBuildResult(
            matrix=matrix,
            method="mgp",
            backend="smt.surrogate_models.MGP",
            metadata={
                "smt_reference_path": smt_reference_path,
                "best_ncomp": best_ncomp,
            },
        )
    except Exception as exc:
        fallback = _pls_embedding(x_train, y_train, effective_dim, rng)
        metadata = dict(fallback.metadata)
        metadata["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        metadata["replacement"] = fallback.backend
        return EmbeddingBuildResult(
            matrix=fallback.matrix,
            method="mgp",
            backend="fallback_pls_after_mgp_failure",
            approximate=True,
            metadata=metadata,
        )


def _orthonormalize_rows(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    repaired = _repair_degenerate_rows(np.asarray(matrix, dtype=float), rng)
    q, _ = np.linalg.qr(repaired.T, mode="reduced")
    rows = q.T
    if rows.shape[0] < repaired.shape[0]:
        extra = rng.normal(size=(repaired.shape[0] - rows.shape[0], repaired.shape[1]))
        rows = np.vstack([rows, extra])
    return _repair_degenerate_rows(rows[: repaired.shape[0]], rng)


def _repair_degenerate_rows(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    repaired = np.asarray(matrix, dtype=float).copy()
    for row_idx in range(repaired.shape[0]):
        if not np.all(np.isfinite(repaired[row_idx])) or np.linalg.norm(repaired[row_idx]) == 0.0:
            repaired[row_idx] = rng.normal(size=repaired.shape[1])
    return repaired


def _metadata_scalar(value: object, default: int) -> int:
    """Convert SMT metadata values that may be scalars, lists, or arrays."""

    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return int(default)
        return int(arr[0])
    except Exception:
        return int(default)
