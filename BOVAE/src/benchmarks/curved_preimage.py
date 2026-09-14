"""Deterministic curved-preimage benchmarks for the BO-VAE/EGORSE study.

The online contract is deliberately scalar-only: optimizers receive ambient
points and original minimization values.  Hidden rotations and nonlinear gate
coordinates are retained in immutable artifacts for construction validation
and post-hoc diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from .base import BaseTestFunction


MASTER_CONSTRUCTION_SEED = 20260808
ALPHA_CANDIDATES = (0.5, 0.75, 1.0, 1.25)
BASE_COORDINATE_TRANSFORM_VERSION = "table_a3_full_domain_tanh_v2"
BASE_EFFECTIVE_DIMS = {
    "branin": 2,
    "ackley": 4,
    "rosenbrock": 4,
    "rastrigin": 4,
}
BASE_DOMAINS = {
    "branin": [[-5.0, 10.0], [0.0, 15.0]],
    "ackley": [[-30.0, 30.0]] * 4,
    "rosenbrock": [[-5.0, 10.0]] * 4,
    "rastrigin": [[-5.12, 5.12]] * 4,
}
AMBIENT_DIMS = (10, 100)
ACTIVE_OPTIMUM = np.asarray([0.5, 0.0, 0.0, 0.0], dtype=np.float64)
BRANIN_CONVENTIONAL_MINIMIZERS = np.asarray(
    [
        [-math.pi, 12.275],
        [math.pi, 2.275],
        [3.0 * math.pi, 2.475],
    ],
    dtype=np.float64,
)


def normalized_problem_key(value: str) -> str:
    return str(value).lower().replace("-", "_")


def problem_id(base_function: str, dim: int) -> str:
    base = normalized_problem_key(base_function)
    if base not in BASE_EFFECTIVE_DIMS:
        raise ValueError(f"unsupported curved-preimage base: {base_function}")
    if int(dim) not in AMBIENT_DIMS:
        raise ValueError("curved-preimage ambient dimension must be 10 or 100")
    return f"curved_{base}_d{int(dim)}_de{BASE_EFFECTIVE_DIMS[base]}"


CURVED_PREIMAGE_KEYS = {
    problem_id(base, dim)
    for base in BASE_EFFECTIVE_DIMS
    for dim in AMBIENT_DIMS
}


def is_curved_preimage_key(value: str) -> bool:
    return normalized_problem_key(value) in CURVED_PREIMAGE_KEYS


def construction_seed(base_function: str, dim: int, effective_dim: int) -> tuple[int, str]:
    token = (
        "curved-preimage-active-map|"
        f"{MASTER_CONSTRUCTION_SEED}|{normalized_problem_key(base_function)}|"
        f"D={int(dim)}|de={int(effective_dim)}"
    )
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:32], 16), digest


def _canonical_rotation(generator: np.random.Generator, dim: int) -> np.ndarray:
    raw = np.asarray(generator.normal(size=(dim, dim)), dtype=np.float64, order="C")
    columns, upper = np.linalg.qr(raw)
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    columns = columns * signs.reshape(1, dim)
    return np.asarray(columns.T, dtype=np.float64, order="C")


def _branin_active_minimizers() -> np.ndarray:
    normalized = np.empty_like(BRANIN_CONVENTIONAL_MINIMIZERS)
    normalized[:, 0] = (
        2.0 * (BRANIN_CONVENTIONAL_MINIMIZERS[:, 0] + 5.0) / 15.0 - 1.0
    )
    normalized[:, 1] = 2.0 * BRANIN_CONVENTIONAL_MINIMIZERS[:, 1] / 15.0 - 1.0
    if np.any(np.abs(normalized) >= 1.0):
        raise RuntimeError("Branin minimizer cannot be inverted through tanh")
    return np.arctanh(normalized) / 4.0


def active_minimizers(base_function: str) -> np.ndarray:
    base = normalized_problem_key(base_function)
    if base == "branin":
        return _branin_active_minimizers()
    effective_dim = BASE_EFFECTIVE_DIMS[base]
    return ACTIVE_OPTIMUM[:effective_dim].reshape(1, effective_dim).copy()


def _rosenbrock_offset() -> np.ndarray:
    # Legacy (-2.048, 2.048)^4 map retained for reproducibility/reference:
    # target_unit = 2.0 * (1.0 + 2.048) / 4.096 - 1.0
    target_unit = 2.0 * (1.0 - (-5.0)) / (10.0 - (-5.0)) - 1.0
    rho = np.arctanh(target_unit)
    return ACTIVE_OPTIMUM - rho


def base_coordinates(values: np.ndarray, base_function: str) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64)
    base = normalized_problem_key(base_function)
    if z.ndim != 2 or z.shape[1] != BASE_EFFECTIVE_DIMS[base]:
        raise ValueError("active values have an incompatible shape")
    if base == "ackley":
        # Legacy (-5, 5)^4 map retained for reproducibility/reference:
        # return 5.0 * np.tanh(z - ACTIVE_OPTIMUM.reshape(1, 4))
        return 30.0 * np.tanh(z - ACTIVE_OPTIMUM.reshape(1, 4))
    if base == "rastrigin":
        return 5.12 * np.tanh(z - ACTIVE_OPTIMUM.reshape(1, 4))
    if base == "rosenbrock":
        offset = _rosenbrock_offset().reshape(1, 4)
        # Legacy (-2.048, 2.048)^4 map retained for reproducibility/reference:
        # return -2.048 + 4.096 * (np.tanh(z - offset) + 1.0) / 2.0
        return -5.0 + 15.0 * (np.tanh(z - offset) + 1.0) / 2.0
    if base == "branin":
        transformed = np.empty_like(z)
        transformed[:, 0] = -5.0 + 15.0 * (np.tanh(4.0 * z[:, 0]) + 1.0) / 2.0
        transformed[:, 1] = 15.0 * (np.tanh(4.0 * z[:, 1]) + 1.0) / 2.0
        return transformed
    raise ValueError(f"unsupported base: {base_function}")


def evaluate_base(values: np.ndarray, base_function: str) -> np.ndarray:
    x = base_coordinates(values, base_function)
    base = normalized_problem_key(base_function)
    if base == "ackley":
        first = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(np.square(x), axis=1)))
        second = -np.exp(np.mean(np.cos(2.0 * np.pi * x), axis=1))
        return first + second + 20.0 + np.e
    if base == "rastrigin":
        return 10.0 * x.shape[1] + np.sum(
            np.square(x) - 10.0 * np.cos(2.0 * np.pi * x), axis=1
        )
    if base == "rosenbrock":
        return np.sum(
            np.square(1.0 - x[:, :-1])
            + 100.0 * np.square(x[:, 1:] - np.square(x[:, :-1])),
            axis=1,
        )
    if base == "branin":
        x1 = x[:, 0]
        x2 = x[:, 1]
        a = 5.1 / (4.0 * np.pi**2)
        b = 5.0 / np.pi
        c = 10.0 * (1.0 - 1.0 / (8.0 * np.pi))
        return np.square(x2 - a * np.square(x1) + b * x1 - 6.0) + c * np.cos(x1) + 10.0
    raise ValueError(f"unsupported base: {base_function}")


def build_curved_preimage_artifact(
    base_function: str,
    dim: int,
    *,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base = normalized_problem_key(base_function)
    effective_dim = BASE_EFFECTIVE_DIMS[base]
    seed, full_digest = construction_seed(base, dim, effective_dim)
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    rotation = _canonical_rotation(generator, int(dim))
    matrix = np.asarray(rotation[:effective_dim], dtype=np.float64, order="C")
    complement = np.asarray(rotation[effective_dim:], dtype=np.float64, order="C")
    gate_rows = effective_dim - 1
    gate_dim = int(dim) - effective_dim
    weights = generator.integers(0, 2, size=(gate_rows, gate_dim), dtype=np.int8)
    weights = np.asarray(2 * weights - 1, dtype=np.float64, order="C")
    phases = np.asarray(
        generator.uniform(0.0, 2.0 * np.pi, size=(gate_rows, gate_dim)),
        dtype=np.float64,
        order="C",
    )
    gate_zero = np.tanh(np.sum(weights * np.sin(phases), axis=1) / math.sqrt(gate_dim))
    minimizers = active_minimizers(base)
    witnesses = np.matmul(minimizers, matrix)
    if np.max(np.abs(witnesses)) > 1.0 + 1e-12:
        raise ValueError("known optimum witness is outside the ambient box")
    arrays = {
        "rotation": rotation,
        "matrix": matrix,
        "complement": complement,
        "gate_weights": weights,
        "gate_phases": phases,
        "gate_zero": gate_zero,
        "active_minimizers": minimizers,
        "optimum_witnesses": witnesses,
    }
    minimum = float(evaluate_base(minimizers, base).min())
    metadata: dict[str, Any] = {
        "artifact_type": "curved_preimage_benchmark_v1",
        "problem_id": problem_id(base, dim),
        "base_function": base,
        "D": int(dim),
        "d_e": int(effective_dim),
        "alpha": float(alpha),
        "global_minimum": minimum,
        "ambient_domain": "[-1,1]^D",
        "base_domain": BASE_DOMAINS[base],
        "base_coordinate_transform_version": BASE_COORDINATE_TRANSFORM_VERSION,
        "construction_master_seed": MASTER_CONSTRUCTION_SEED,
        "construction_seed_token_sha256": full_digest,
        "construction_seed_first_128_bits_unsigned": str(seed),
        "prng": "numpy.random.PCG64DXSM",
        "dtype": "float64",
        "array_order": "C",
        "qr": "numpy.linalg.qr; columns sign-normalized to nonnegative diag(R); Q=U.T",
        "draws_ambient_optimum": False,
        "hidden_structure_online_use": False,
        "known_optimum_online_use": False,
        "objective_convention": "original minimization objective",
    }
    return arrays, metadata


@dataclass(frozen=True)
class CurvedPreimageProblem:
    problem_id: str
    base_function: str
    dim: int
    effective_dim: int
    alpha: float
    global_minimum: float
    matrix: np.ndarray
    complement: np.ndarray
    gate_weights: np.ndarray
    gate_phases: np.ndarray
    gate_zero: np.ndarray
    active_minimizers: np.ndarray
    optimum_witnesses: np.ndarray

    @classmethod
    def from_artifact(
        cls,
        artifact_path: Union[str, Path],
        metadata: Union[dict[str, Any], str, Path],
    ) -> "CurvedPreimageProblem":
        if isinstance(metadata, (str, Path)):
            metadata = json.loads(Path(metadata).read_text(encoding="utf-8"))
        with np.load(Path(artifact_path), allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name], dtype=np.float64) for name in payload.files}
        return cls(
            problem_id=str(metadata["problem_id"]),
            base_function=str(metadata["base_function"]),
            dim=int(metadata["D"]),
            effective_dim=int(metadata["d_e"]),
            alpha=float(metadata["alpha"]),
            global_minimum=float(metadata["global_minimum"]),
            matrix=arrays["matrix"],
            complement=arrays["complement"],
            gate_weights=arrays["gate_weights"],
            gate_phases=arrays["gate_phases"],
            gate_zero=arrays["gate_zero"],
            active_minimizers=arrays["active_minimizers"],
            optimum_witnesses=arrays["optimum_witnesses"],
        )

    def __post_init__(self) -> None:
        if self.problem_id not in CURVED_PREIMAGE_KEYS:
            raise ValueError("unknown curved-preimage problem id")
        if self.matrix.shape != (self.effective_dim, self.dim):
            raise ValueError("active matrix has an incompatible shape")
        if self.complement.shape != (self.dim - self.effective_dim, self.dim):
            raise ValueError("complement matrix has an incompatible shape")
        rotation = np.concatenate([self.matrix, self.complement], axis=0)
        if not np.allclose(rotation @ rotation.T, np.eye(self.dim), rtol=0.0, atol=1e-12):
            raise ValueError("artifact rotation is not orthogonal")
        if self.gate_weights.shape != (self.effective_dim - 1, self.dim - self.effective_dim):
            raise ValueError("gate weights have an incompatible shape")
        attained = self.evaluate_batch(self.optimum_witnesses)
        if not np.allclose(attained, self.global_minimum, rtol=0.0, atol=1e-10):
            raise ValueError("known witnesses do not attain the declared minimum")

    @property
    def name(self) -> str:
        return self.problem_id

    @property
    def bounds(self) -> np.ndarray:
        return np.repeat(np.asarray([[-1.0, 1.0]], dtype=np.float64), self.dim, axis=0)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Return a redacted placeholder; optimizers never receive hidden coordinates."""

        values = np.asarray(x)
        return np.zeros(values.shape[:-1] + (self.effective_dim,), dtype=np.float64)

    def active_map_posthoc(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"x must have shape (n, {self.dim})")
        if np.any(values < -1.0 - 1e-12) or np.any(values > 1.0 + 1e-12):
            raise ValueError("x is outside [-1,1]^D")
        active = values @ self.matrix.T
        complement = values @ self.complement.T
        gate_argument = (
            np.sin(complement[:, None, :] * np.pi + self.gate_phases[None, :, :])
            * self.gate_weights[None, :, :]
        ).sum(axis=2) / math.sqrt(self.dim - self.effective_dim)
        gate = np.tanh(gate_argument)
        centered_gate = 0.5 * (gate - self.gate_zero.reshape(1, -1))
        result = active.copy()
        result[:, 1:] += self.alpha * centered_gate
        return result

    def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        return evaluate_base(self.active_map_posthoc(x), self.base_function)

    def evaluate(self, x: np.ndarray) -> float:
        values = np.asarray(x, dtype=np.float64)
        if values.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},)")
        return float(self.evaluate_batch(values.reshape(1, -1))[0])


class CurvedPreimage(BaseTestFunction):
    """Torch objective adapter for the immutable curved-preimage artifacts."""

    is_curved_preimage_benchmark = True

    def __init__(
        self,
        dim: int,
        bounds: torch.Tensor | None = None,
        *,
        artifact_path: str | Path,
        metadata_path: str | Path,
        expected_problem: str | None = None,
    ) -> None:
        metadata_file = Path(metadata_path)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        reference = CurvedPreimageProblem.from_artifact(artifact_path, metadata)
        if reference.dim != int(dim):
            raise ValueError("curved-preimage artifact dimension does not match config")
        if (
            expected_problem is not None
            and reference.problem_id != normalized_problem_key(expected_problem)
        ):
            raise ValueError("curved-preimage problem id does not match config")
        super().__init__(int(dim), bounds)
        expected_bounds = torch.tensor([[-1.0, 1.0]] * self.dim, dtype=torch.float64)
        if self.bounds is None:
            self.bounds = expected_bounds
        else:
            self.bounds = self.bounds.to(dtype=torch.float64)
            if not torch.allclose(
                self.bounds, expected_bounds, rtol=0.0, atol=1e-12
            ):
                raise ValueError("curved-preimage benchmarks use [-1,1]^D")
        self.problem_id = reference.problem_id
        self.base_function = reference.base_function
        self.effective_dim = reference.effective_dim
        self.alpha = reference.alpha
        self.global_minimum = reference.global_minimum
        self.matrix = torch.as_tensor(reference.matrix, dtype=torch.float64)
        self.complement = torch.as_tensor(reference.complement, dtype=torch.float64)
        self.gate_weights = torch.as_tensor(reference.gate_weights, dtype=torch.float64)
        self.gate_phases = torch.as_tensor(reference.gate_phases, dtype=torch.float64)
        self.gate_zero = torch.as_tensor(reference.gate_zero, dtype=torch.float64)
        self.active_minimizers = torch.as_tensor(
            reference.active_minimizers, dtype=torch.float64
        )
        self.optimum_witnesses = torch.as_tensor(
            reference.optimum_witnesses, dtype=torch.float64
        )
        self.artifact_path = str(Path(artifact_path))
        self.metadata_path = str(metadata_file)
        self.optimal_input = None
        self.optimal_value = float(self.global_minimum)
        self.name = self.problem_id

    def active_map_posthoc(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must have shape (n, {self.dim})")
        if torch.any(x < -1.0 - 1e-10) or torch.any(x > 1.0 + 1e-10):
            raise ValueError("x is outside [-1,1]^D")
        active = x.matmul(self.matrix.to(x).T)
        complement = x.matmul(self.complement.to(x).T)
        gate_argument = (
            torch.sin(torch.pi * complement[:, None, :] + self.gate_phases.to(x)[None, :, :])
            * self.gate_weights.to(x)[None, :, :]
        ).sum(dim=2) / math.sqrt(self.dim - self.effective_dim)
        centered_gate = 0.5 * (
            torch.tanh(gate_argument) - self.gate_zero.to(x).reshape(1, -1)
        )
        transformed = active.clone()
        transformed[:, 1:] += float(self.alpha) * centered_gate
        return transformed

    def base_coordinates(self, z: torch.Tensor) -> torch.Tensor:
        optimum = torch.tensor(
            [0.5, 0.0, 0.0, 0.0], device=z.device, dtype=z.dtype
        )
        if self.base_function == "ackley":
            return 30.0 * torch.tanh(z - optimum)
        if self.base_function == "rastrigin":
            return 5.12 * torch.tanh(z - optimum)
        if self.base_function == "rosenbrock":
            target_unit = 2.0 * (1.0 - (-5.0)) / 15.0 - 1.0
            rho = torch.atanh(
                torch.tensor(target_unit, device=z.device, dtype=z.dtype)
            )
            return -5.0 + 15.0 * (torch.tanh(z - (optimum - rho)) + 1.0) / 2.0
        if self.base_function == "branin":
            transformed = torch.empty_like(z)
            transformed[:, 0] = -5.0 + 15.0 * (
                torch.tanh(4.0 * z[:, 0]) + 1.0
            ) / 2.0
            transformed[:, 1] = 15.0 * (
                torch.tanh(4.0 * z[:, 1]) + 1.0
            ) / 2.0
            return transformed
        raise RuntimeError(f"unsupported base: {self.base_function}")

    def original_objective(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_coordinates(self.active_map_posthoc(x))
        if self.base_function == "ackley":
            values = (
                -20.0 * torch.exp(-0.2 * torch.sqrt(torch.mean(base.square(), dim=1)))
                - torch.exp(torch.mean(torch.cos(2.0 * torch.pi * base), dim=1))
                + 20.0
                + torch.exp(torch.ones((), device=x.device, dtype=x.dtype))
            )
        elif self.base_function == "rastrigin":
            values = 10.0 * self.effective_dim + torch.sum(
                base.square() - 10.0 * torch.cos(2.0 * torch.pi * base), dim=1
            )
        elif self.base_function == "rosenbrock":
            values = torch.sum(
                (1.0 - base[:, :-1]).square()
                + 100.0 * (base[:, 1:] - base[:, :-1].square()).square(),
                dim=1,
            )
        elif self.base_function == "branin":
            x1, x2 = base[:, 0], base[:, 1]
            a = 5.1 / (4.0 * torch.pi**2)
            b = 5.0 / torch.pi
            c = 10.0 * (1.0 - 1.0 / (8.0 * torch.pi))
            values = (
                (x2 - a * x1.square() + b * x1 - 6.0).square()
                + c * torch.cos(x1)
                + 10.0
            )
        else:
            raise RuntimeError(f"unsupported base: {self.base_function}")
        return values.reshape(-1, 1)

    def func(self, x: torch.Tensor) -> torch.Tensor:
        return -self.original_objective(x)

    def metadata(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "base_function": self.base_function,
            "D": self.dim,
            "d_e": self.effective_dim,
            "alpha": self.alpha,
            "global_minimum": self.global_minimum,
            "base_coordinate_transform_version": BASE_COORDINATE_TRANSFORM_VERSION,
            "artifact_path": self.artifact_path,
            "metadata_path": self.metadata_path,
            "hidden_transform_online_use": False,
            "known_optimum_online_use": False,
            "draws_ambient_optimum": False,
            "objective_convention": "func=-f_original",
        }
