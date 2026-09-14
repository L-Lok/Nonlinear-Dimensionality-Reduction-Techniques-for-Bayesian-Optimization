"""Original sequential domain-reduction contraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from warnings import warn

import torch


class SequentialDomainReductionTransformer:
    """Reduce a box using the Stander--Craig sequential scheme.

    Parameters
    ----------
    gamma_osc:
        Damping factor for oscillating incumbent motion.
    gamma_pan:
        Scaling factor for consistent incumbent motion.
    eta:
        Base contraction factor.
    minimum_window:
        Minimum width in each coordinate, supplied as one scalar or one value
        per dimension.
    """

    def __init__(
        self,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9,
        minimum_window: float | Sequence[float] | Mapping[str, float] = 0.0,
    ) -> None:
        self.gamma_osc = float(gamma_osc)
        self.gamma_pan = float(gamma_pan)
        self.eta = float(eta)
        if isinstance(minimum_window, Mapping):
            self.minimum_window_value: float | list[float] = [
                float(value) for _, value in sorted(minimum_window.items())
            ]
        elif isinstance(minimum_window, Sequence) and not isinstance(
            minimum_window, (str, bytes)
        ):
            self.minimum_window_value = [float(value) for value in minimum_window]
        elif isinstance(minimum_window, torch.Tensor) and minimum_window.ndim > 0:
            self.minimum_window_value = [float(value) for value in minimum_window]
        else:
            self.minimum_window_value = float(minimum_window)

    def initialize(self, original_bounds: torch.Tensor) -> None:
        """Initialize the transformer on a ``dimension x 2`` box."""

        if original_bounds.ndim != 2 or original_bounds.shape[1] != 2:
            raise ValueError("original_bounds must have shape dimension x 2")
        self.original_bounds = original_bounds
        self.bounds = [self.original_bounds]
        if isinstance(self.minimum_window_value, list):
            if len(self.minimum_window_value) != len(original_bounds):
                raise ValueError("minimum_window must have one value per dimension")
            self.minimum_window = torch.as_tensor(
                self.minimum_window_value,
                dtype=original_bounds.dtype,
                device=original_bounds.device,
            )
        else:
            self.minimum_window = torch.full(
                (len(original_bounds),),
                self.minimum_window_value,
                dtype=original_bounds.dtype,
                device=original_bounds.device,
            )
        self._window_bounds_compatibility(self.original_bounds)

        center = torch.mean(self.original_bounds, dim=1)
        self.previous_optimal = center.clone()
        self.current_optimal = center.clone()
        self.r = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        self.previous_d = 2.0 * (
            self.current_optimal - self.previous_optimal
        ) / self.r
        self.current_d = 2.0 * (
            self.current_optimal - self.previous_optimal
        ) / self.r
        self._update_contraction_parameters()
        self.r = self.contraction_rate * self.r

    def _update_contraction_parameters(self) -> None:
        self.c = self.current_d * self.previous_d
        self.c_hat = torch.sqrt(torch.abs(self.c)) * torch.sign(self.c)
        self.gamma = 0.5 * (
            self.gamma_pan * (1.0 + self.c_hat)
            + self.gamma_osc * (1.0 - self.c_hat)
        )
        self.contraction_rate = self.eta + torch.abs(self.current_d) * (
            self.gamma - self.eta
        )

    def _update(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d
        self.current_optimal = train_x[torch.argmax(train_y)]
        self.current_d = 2.0 * (
            self.current_optimal - self.previous_optimal
        ) / self.r
        self._update_contraction_parameters()
        self.r = self.contraction_rate * self.r

    def _trim(
        self, new_bounds: torch.Tensor, global_bounds: torch.Tensor
    ) -> torch.Tensor:
        """Clip a proposed box and enforce its minimum coordinate widths."""

        new_bounds = torch.sort(new_bounds).values.detach().clone()
        global_bounds = global_bounds.detach().clone()
        for index, coordinate_bounds in enumerate(new_bounds):
            if coordinate_bounds[0] < global_bounds[index, 0]:
                coordinate_bounds[0] = global_bounds[index, 0]
            if coordinate_bounds[1] > global_bounds[index, 1]:
                coordinate_bounds[1] = global_bounds[index, 1]
            if coordinate_bounds[0] > global_bounds[index, 1]:
                coordinate_bounds[0] = global_bounds[index, 0]
                warn(
                    "SDR lower bound exceeded the global upper bound and was reset.",
                    stacklevel=2,
                )
            if coordinate_bounds[1] < global_bounds[index, 0]:
                coordinate_bounds[1] = global_bounds[index, 1]
                warn(
                    "SDR upper bound fell below the global lower bound and was reset.",
                    stacklevel=2,
                )

        for index, coordinate_bounds in enumerate(new_bounds):
            current_width = abs(coordinate_bounds[0] - coordinate_bounds[1])
            if current_width >= self.minimum_window[index]:
                continue
            half_deficit = (self.minimum_window[index] - current_width) / 2.0
            available_left = abs(
                global_bounds[index, 0] - coordinate_bounds[0]
            )
            available_right = abs(
                global_bounds[index, 1] - coordinate_bounds[1]
            )
            expand_left = min(half_deficit, available_left)
            expand_right = min(half_deficit, available_right)
            coordinate_bounds[0] -= expand_left + max(
                half_deficit - expand_right, 0
            )
            coordinate_bounds[1] += expand_right + max(
                half_deficit - expand_left, 0
            )
        return new_bounds

    def _window_bounds_compatibility(self, global_bounds: torch.Tensor) -> None:
        for index, coordinate_bounds in enumerate(global_bounds):
            global_width = abs(coordinate_bounds[1] - coordinate_bounds[0])
            if global_width < self.minimum_window[index]:
                raise ValueError(
                    "global bounds are incompatible with the minimum SDR window"
                )

    def transform(
        self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> torch.Tensor:
        """Update the contraction state and return the next search bounds."""

        self._update(train_x=train_x, train_y=train_y)
        new_bounds = torch.stack(
            [
                self.current_optimal - 0.5 * self.r,
                self.current_optimal + 0.5 * self.r,
            ]
        ).T
        new_bounds = self._trim(new_bounds, self.original_bounds)
        self.bounds.append(new_bounds)
        return new_bounds


__all__ = ["SequentialDomainReductionTransformer"]
