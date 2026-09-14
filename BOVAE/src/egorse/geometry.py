"""Embedding geometry for EGORSE Section IV.C-D.

This module implements the paper's B box definition (Eq. 6), gamma_B
constrained backward map (Eq. 7), gamma_W projection fallback (Eq. 10), and
the f^(t), g^(t) projected objective/constraint definitions (Eq. 9 and 11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class GammaResult:
    """Result of a backward projection from reduced u to ambient x."""

    x: np.ndarray
    feasible: bool
    status: str
    residual_norm: float
    backend: str


@dataclass(frozen=True)
class ProjectedEvaluation:
    """One f^(t)(u), g^(t)(u) evaluation in the B hypercube."""

    x: np.ndarray
    u: np.ndarray
    f: float
    g: float
    feasible: bool
    gamma_status: str
    gamma_backend: str


def embedding_box(matrix: np.ndarray) -> np.ndarray:
    """Return B_i = [-sum_j |A_ij|, sum_j |A_ij|] from Eq. (6)."""

    matrix_arr = _as_matrix(matrix)
    row_l1 = np.sum(np.abs(matrix_arr), axis=1)
    return np.column_stack([-row_l1, row_l1])


def gamma_b(
    matrix: np.ndarray,
    u: np.ndarray,
    tol: float = 1e-7,
    backend: str = "auto",
) -> GammaResult:
    """Compute gamma_B(u) from Eq. (7) by constrained QP.

    The preferred backend is CVXOPT when installed, matching the paper's
    implementation details in Section V.A. A SciPy SLSQP fallback is used when
    CVXOPT is unavailable or fails numerically.
    """

    matrix_arr = _as_matrix(matrix)
    u_arr = _as_u(u, matrix_arr.shape[0])
    if backend in {"auto", "cvxopt"}:
        result = _gamma_b_cvxopt(matrix_arr, u_arr, tol)
        if result.feasible or backend == "cvxopt":
            return result
    return _gamma_b_scipy(matrix_arr, u_arr, tol)


def gamma_w(matrix: np.ndarray, u: np.ndarray) -> GammaResult:
    """Compute gamma_W(u) from Eq. (10) by box projection of A^+ u."""

    matrix_arr = _as_matrix(matrix)
    u_arr = _as_u(u, matrix_arr.shape[0])
    x0 = np.linalg.pinv(matrix_arr) @ u_arr
    x = np.clip(x0, -1.0, 1.0)
    residual = float(np.linalg.norm(matrix_arr @ x - u_arr))
    return GammaResult(
        x=x,
        feasible=False,
        status="box_projection",
        residual_norm=residual,
        backend="clip_pinv",
    )


def constraint_value(matrix: np.ndarray, u: np.ndarray, gamma: GammaResult) -> float:
    """Evaluate g^(t)(u) from Eq. (9), with normalized infeasible values.

    The paper states that the constraint is normalized to [-1, 1]. For u not in
    A, we therefore use -mean((u_i / sum_j |A_ij|)^2), which is in [-1, 0] for
    u in B. Feasible points use 1 - ||gamma_B(u)||^2 / d as in Eq. (9).
    """

    matrix_arr = _as_matrix(matrix)
    u_arr = _as_u(u, matrix_arr.shape[0])
    if gamma.feasible:
        return float(1.0 - np.dot(gamma.x, gamma.x) / matrix_arr.shape[1])
    row_l1 = np.sum(np.abs(matrix_arr), axis=1)
    safe = np.where(row_l1 > 0.0, row_l1, 1.0)
    normalized_u = u_arr / safe
    return float(-np.mean(normalized_u**2))


def evaluate_projected(
    matrix: np.ndarray,
    u: np.ndarray,
    objective: Callable[[np.ndarray], float],
    tol: float = 1e-7,
    gamma_backend: str = "auto",
) -> ProjectedEvaluation:
    """Evaluate f^(t)(u), g^(t)(u) from Eq. (11) and Eq. (9)."""

    matrix_arr = _as_matrix(matrix)
    u_arr = _as_u(u, matrix_arr.shape[0])
    backward = gamma_b(matrix_arr, u_arr, tol=tol, backend=gamma_backend)
    if not backward.feasible:
        backward = gamma_w(matrix_arr, u_arr)
    f_val = float(objective(backward.x))
    g_val = constraint_value(matrix_arr, u_arr, backward)
    return ProjectedEvaluation(
        x=backward.x,
        u=u_arr,
        f=f_val,
        g=g_val,
        feasible=bool(g_val >= -tol),
        gamma_status=backward.status,
        gamma_backend=backward.backend,
    )


def positive_constraint_for_known_x(x: np.ndarray) -> float:
    """Eq. (9) feasible branch for an already-known ambient point x."""

    x_arr = np.asarray(x, dtype=float)
    return float(1.0 - np.dot(x_arr, x_arr) / x_arr.size)


def _gamma_b_cvxopt(matrix: np.ndarray, u: np.ndarray, tol: float) -> GammaResult:
    try:
        from cvxopt import matrix as cvx_matrix
        from cvxopt import solvers
    except Exception as exc:
        return GammaResult(
            x=np.empty(matrix.shape[1]),
            feasible=False,
            status=f"cvxopt_unavailable:{type(exc).__name__}",
            residual_norm=float("inf"),
            backend="cvxopt",
        )

    dim = matrix.shape[1]
    x0 = np.linalg.pinv(matrix) @ u
    p = cvx_matrix(2.0 * np.eye(dim))
    q = cvx_matrix(-2.0 * x0)
    g = cvx_matrix(np.vstack([np.eye(dim), -np.eye(dim)]))
    h = cvx_matrix(np.concatenate([np.ones(dim), np.ones(dim)]))
    aeq = cvx_matrix(matrix)
    beq = cvx_matrix(u)
    old_progress = solvers.options.get("show_progress", False)
    solvers.options["show_progress"] = False
    try:
        solution = solvers.qp(p, q, g, h, aeq, beq)
    except Exception as exc:
        solvers.options["show_progress"] = old_progress
        return GammaResult(
            x=np.clip(x0, -1.0, 1.0),
            feasible=False,
            status=f"cvxopt_error:{type(exc).__name__}",
            residual_norm=float("inf"),
            backend="cvxopt",
        )
    solvers.options["show_progress"] = old_progress
    status = str(solution.get("status", "unknown"))
    x = np.asarray(solution["x"], dtype=float).reshape(-1)
    residual = float(np.linalg.norm(matrix @ x - u))
    in_box = bool(np.all(x >= -1.0 - tol) and np.all(x <= 1.0 + tol))
    feasible = status == "optimal" and residual <= tol and in_box
    return GammaResult(
        x=np.clip(x, -1.0, 1.0),
        feasible=feasible,
        status=status,
        residual_norm=residual,
        backend="cvxopt",
    )


def _gamma_b_scipy(matrix: np.ndarray, u: np.ndarray, tol: float) -> GammaResult:
    dim = matrix.shape[1]
    x0 = np.linalg.pinv(matrix) @ u

    def objective(x: np.ndarray) -> float:
        diff = x - x0
        return float(np.dot(diff, diff))

    def jacobian(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - x0)

    constraints = {"type": "eq", "fun": lambda x: matrix @ x - u, "jac": lambda x: matrix}
    result = minimize(
        objective,
        np.clip(x0, -1.0, 1.0),
        jac=jacobian,
        bounds=[(-1.0, 1.0)] * dim,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": tol, "maxiter": 200, "disp": False},
    )
    x = np.asarray(result.x, dtype=float)
    residual = float(np.linalg.norm(matrix @ x - u))
    in_box = bool(np.all(x >= -1.0 - tol) and np.all(x <= 1.0 + tol))
    feasible = bool(result.success and residual <= tol and in_box)
    return GammaResult(
        x=np.clip(x, -1.0, 1.0),
        feasible=feasible,
        status=str(result.message),
        residual_norm=residual,
        backend="scipy_slsqp",
    )


def _as_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix_arr = np.asarray(matrix, dtype=float)
    if matrix_arr.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix_arr.shape[0] <= 0 or matrix_arr.shape[1] <= 0:
        raise ValueError("matrix dimensions must be positive")
    return matrix_arr


def _as_u(u: np.ndarray, effective_dim: int) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float).reshape(-1)
    if u_arr.shape != (effective_dim,):
        raise ValueError(f"u must have shape ({effective_dim},), got {u_arr.shape}")
    return u_arr
