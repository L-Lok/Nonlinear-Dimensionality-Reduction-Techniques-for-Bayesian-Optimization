"""Shared contracts for sequential domain-reduction implementations."""

from __future__ import annotations

from typing import Any, Protocol

import torch


class SDRMethod(Protocol):
    dim: int
    budget: int
    period: int
    current_bounds: torch.Tensor
    update_count: int

    def reset(self) -> None: ...

    def record_candidate_boundary(
        self, candidate_unit: torch.Tensor, tol: float = 0.05
    ) -> bool: ...

    def update(
        self,
        *,
        train_unit_x: torch.Tensor,
        train_internal_y: torch.Tensor,
        best_original: float,
        iteration: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...

    def metadata(self) -> dict[str, Any]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


SDR_METHODS = {"none", "original"}


def validate_sdr_method(value: str) -> str:
    value = str(value)
    if value not in SDR_METHODS:
        raise ValueError(f"sdr_method must be one of {sorted(SDR_METHODS)}")
    return value
