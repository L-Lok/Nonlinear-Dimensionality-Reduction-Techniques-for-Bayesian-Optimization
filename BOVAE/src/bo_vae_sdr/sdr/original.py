"""Pipeline adapter for the original sequential domain-reduction method."""

from __future__ import annotations

from typing import Any

import torch

from ..pipeline.transforms import assert_unit_box, tensor_to_list, unit_bounds
from .original_transformer import SequentialDomainReductionTransformer


METHOD_NAME = "original"


def original_sdr_provenance() -> dict[str, Any]:
    """Describe the public SDR implementation and its coordinate convention."""

    return {
        "sdr_method": METHOD_NAME,
        "core_class": "SequentialDomainReductionTransformer",
        "coordinate_contract": (
            "The adapter passes global unit latent coordinates and d x 2 unit "
            "bounds to the original sequential domain-reduction transformer."
        ),
    }


class OriginalSDR:
    """Apply the original SDR contraction within the BO latent unit box."""

    def __init__(
        self,
        dim: int,
        budget: int,
        *,
        period: int = 5,
        minimum_window_unit: float = 0.05,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.dim = int(dim)
        self.budget = int(budget)
        self.period = int(period)
        self.minimum_window_unit = float(minimum_window_unit)
        self.device = torch.device(device)
        self.dtype = dtype
        self.current_bounds = unit_bounds(
            self.dim, device=self.device, dtype=self.dtype
        )
        self.incumbent_history: list[torch.Tensor] = []
        self.boundary_history: list[bool] = []
        self.best_original_history: list[float] = []
        self.update_count = 0
        self._new_transformer()

    def _new_transformer(self) -> None:
        self.transformer = SequentialDomainReductionTransformer(
            minimum_window=self.minimum_window_unit
        )
        self.transformer.initialize(
            original_bounds=unit_bounds(
                self.dim, device=self.device, dtype=self.dtype
            )
        )

    def reset(self) -> None:
        self.current_bounds = unit_bounds(
            self.dim, device=self.device, dtype=self.dtype
        )
        self.incumbent_history = []
        self.boundary_history = []
        self.best_original_history = []
        self.update_count = 0
        self._new_transformer()

    def record_candidate_boundary(
        self, candidate_unit: torch.Tensor, tol: float = 0.05
    ) -> bool:
        candidate = candidate_unit.detach().reshape(-1, self.dim).to(
            self.current_bounds
        )
        lower = self.current_bounds[:, 0]
        upper = self.current_bounds[:, 1]
        near = bool(
            ((candidate <= lower + tol) | (candidate >= upper - tol)).any().item()
        )
        self.boundary_history.append(near)
        return near

    def should_update(self, iteration: int) -> bool:
        return int(iteration) % self.period == 0

    def update(
        self,
        *,
        train_unit_x: torch.Tensor,
        train_internal_y: torch.Tensor,
        best_original: float,
        iteration: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        train_x = train_unit_x.detach().to(device=self.device, dtype=self.dtype)
        train_y = train_internal_y.detach().to(train_x)
        assert_unit_box(self.current_bounds, name="original SDR current_bounds")
        if train_x.ndim != 2 or train_x.shape[1] != self.dim:
            raise ValueError("train_unit_x has incompatible shape for original SDR")
        if torch.any(train_x < -1e-8) or torch.any(train_x > 1.0 + 1e-8):
            raise ValueError("original SDR received non-unit latent data")

        incumbent = train_x[int(torch.argmax(train_y.reshape(-1)).item())]
        self.incumbent_history.append(incumbent.detach().clone())
        self.best_original_history.append(float(best_original))
        if not self.should_update(iteration):
            return self.current_bounds, {
                "updated": False,
                "reason": "not_due",
                "iteration": int(iteration),
                "sdr_method": METHOD_NAME,
                "current_bounds": tensor_to_list(self.current_bounds),
            }

        previous_bounds = self.current_bounds.detach().clone()
        self.current_bounds = self.transformer.transform(
            train_x=train_x, train_y=train_y
        ).detach().clone()
        assert_unit_box(self.current_bounds, name="original SDR new_bounds")
        self.update_count += 1
        return self.current_bounds, {
            "updated": True,
            "iteration": int(iteration),
            "sdr_method": METHOD_NAME,
            "current_bounds": tensor_to_list(self.current_bounds),
            "previous_bounds": tensor_to_list(previous_bounds),
            "incumbent_unit": tensor_to_list(incumbent.reshape(1, -1)),
            "radius": tensor_to_list(self.transformer.r),
            "contraction_rate": tensor_to_list(self.transformer.contraction_rate),
            "minimum_window_unit": self.minimum_window_unit,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "sdr_method": METHOD_NAME,
            "dim": self.dim,
            "budget": self.budget,
            "period": self.period,
            "minimum_window_unit": self.minimum_window_unit,
            "current_bounds": self.current_bounds.detach().cpu(),
            "update_count": self.update_count,
            "incumbent_history": [
                item.detach().cpu() for item in self.incumbent_history
            ],
            "boundary_history": list(self.boundary_history),
            "best_original_history": list(self.best_original_history),
            "transformer": {
                name: getattr(self.transformer, name).detach().cpu()
                for name in (
                    "original_bounds",
                    "previous_optimal",
                    "current_optimal",
                    "previous_d",
                    "current_d",
                    "c",
                    "c_hat",
                    "gamma",
                    "contraction_rate",
                    "r",
                )
            }
            | {
                "bounds": [
                    item.detach().cpu() for item in self.transformer.bounds
                ]
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("sdr_method") != METHOD_NAME:
            raise ValueError(
                f"original SDR state has wrong method: {state.get('sdr_method')}"
            )
        self.period = int(state.get("period", self.period))
        self.minimum_window_unit = float(
            state.get("minimum_window_unit", self.minimum_window_unit)
        )
        self.current_bounds = state["current_bounds"].to(
            device=self.device, dtype=self.dtype
        )
        self.update_count = int(state.get("update_count", 0))
        self.incumbent_history = [
            item.to(device=self.device, dtype=self.dtype)
            for item in state.get("incumbent_history", [])
        ]
        self.boundary_history = [
            bool(item) for item in state.get("boundary_history", [])
        ]
        self.best_original_history = [
            float(item) for item in state.get("best_original_history", [])
        ]
        self._new_transformer()
        transformer_state = state.get("transformer", {})
        for name in (
            "original_bounds",
            "previous_optimal",
            "current_optimal",
            "previous_d",
            "current_d",
            "c",
            "c_hat",
            "gamma",
            "contraction_rate",
            "r",
        ):
            if name in transformer_state:
                setattr(
                    self.transformer,
                    name,
                    transformer_state[name].to(device=self.device, dtype=self.dtype),
                )
        if "bounds" in transformer_state:
            self.transformer.bounds = [
                item.to(device=self.device, dtype=self.dtype)
                for item in transformer_state["bounds"]
            ]

    def metadata(self) -> dict[str, Any]:
        return {
            **original_sdr_provenance(),
            "period": self.period,
            "minimum_window_unit": self.minimum_window_unit,
            "defaults": {
                "gamma_osc": self.transformer.gamma_osc,
                "gamma_pan": self.transformer.gamma_pan,
                "eta": self.transformer.eta,
            },
        }


__all__ = ["OriginalSDR", "original_sdr_provenance"]
