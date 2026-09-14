"""Common optimizer interface used by EGORSE and future BO-VAE comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class OptimizationHistory:
    """Serializable history returned by shared benchmark optimizers."""

    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def best_valid_value(self) -> float | None:
        valid = [
            float(row["f"])
            for row in self.records
            if bool(row.get("feasibility", True)) and np.isfinite(float(row["f"]))
        ]
        return min(valid) if valid else None


class BaseOptimizer(Protocol):
    """Optimizer contract: run(problem, budget, init_X, seed) -> history."""

    def run(
        self,
        problem: Any,
        budget: int,
        init_X: np.ndarray,
        seed: int,
    ) -> OptimizationHistory:
        """Run the optimizer on a minimization problem."""
