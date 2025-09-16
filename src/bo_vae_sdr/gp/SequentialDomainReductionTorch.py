from typing import Dict, List, Optional, Union
from warnings import warn

import numpy as np
import torch


class DomainTransformer:
    """Base class."""

    def __init__(self, **kwargs) -> None:
        """To override with specific implementation."""
        pass

    def initialize(self, **kwargs) -> None:
        """To override with specific implementation."""
        raise NotImplementedError

    def transform(self, **kwargs) -> np.ndarray:
        """To override with specific implementation."""
        raise NotImplementedError


class SequentialDomainReductionTransformer(DomainTransformer):
    """Reduce the searchable space.

    A sequential domain reduction transformer based on the work by Stander, N. and Craig, K:
    "On the robustness of a simple domain reduction scheme for simulation-based optimization"

    Parameters
    ----------
    gamma_osc : float, default=0.7
        Parameter used to scale (typically dampen) oscillations.

    gamma_pan : float, default=1.0
        Parameter used to scale (typically unitary) panning.

    eta : float, default=0.9
        Zooming parameter used to shrink the region of interest.

    minimum_window : float or np.ndarray or dict, default=0.0
        Minimum window size for each parameter. If a float is provided, the same value is used for all parameters.
    """

    def __init__(
        self,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9,
        minimum_window: Optional[
            Union[List[float], float, Dict[str, float]]
        ] = 0.0,
        # original_bounds: torch.Tensor = None
    ) -> None:
        self.gamma_osc = gamma_osc
        self.gamma_pan = gamma_pan
        self.eta = eta
        if isinstance(minimum_window, dict):
            self.minimum_window_value = [
                item[1] for item in sorted(minimum_window.items(), key=lambda x: x[0])
            ]
        else:
            self.minimum_window_value = minimum_window

    def initialize(self, original_bounds) -> None:
        """Initialize all of the parameters.

        Parameters
        ----------
        original_bounds : The initial bounds, a d x 2  tensor
            Initial Domain this DomainTransformer operates on.
        """
        # Set the original bounds
        self.original_bounds = original_bounds
        self.bounds = [self.original_bounds]

        # Set the minimum window to an array of length bounds
        if isinstance(self.minimum_window_value, list) or isinstance(
            self.minimum_window_value, torch.Tensor
        ):
            assert len(self.minimum_window_value) == len(original_bounds)
            self.minimum_window = self.minimum_window_value
        else:
            self.minimum_window = [
                self.minimum_window_value] * len(original_bounds)

        # print("self.original_bounds", self.original_bounds)

        # Set initial values
        # print(self.original_bounds.dtype)
        self.previous_optimal = torch.mean(self.original_bounds, dim=1)
        self.current_optimal = torch.mean(self.original_bounds, dim=1)
        self.r = self.original_bounds[:, 1] - self.original_bounds[:, 0]

        #############################################################################
        # debug use
        # print("initialised self.current_optimal", self.current_optimal)

        # print("initialised self.previous_optimal:", self.previous_optimal)

        # print("initialised self.r:", self.r)
        #############################################################################

        self.previous_d = 2.0 * \
            (self.current_optimal - self.previous_optimal) / self.r

        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d
        self.c_hat = torch.sqrt(torch.abs(self.c)) * torch.sign(self.c)

        self.gamma = 0.5 * (
            self.gamma_pan * (1.0 + self.c_hat) +
            self.gamma_osc * (1.0 - self.c_hat)
        )

        self.contraction_rate = self.eta + torch.abs(self.current_d) * (
            self.gamma - self.eta
        )

        self.r = self.contraction_rate * self.r

        # check if the minimum window fits in the original bounds
        self._window_bounds_compatibility(self.original_bounds)

    def _update(self, train_x, train_y) -> None:
        """Update contraction rate, window size, and window center.

        Parameters
        ----------
        train_y: The current training dataset for y (function values)
        train_x: The current training dataset for x

        Note:
        Only update the bounds after the new candidate is obtained !
        """
        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        # TODO (CHECKED!)
        # the x's coord corresponding the best f value found so far
        self.current_optimal = train_x[torch.argmax(train_y)]
        ###########################################################
        # debug use:
        # print("self.current_optimal in _update:", self.current_optimal)
        # print("self.previous_optimal in _update:", self.previous_optimal)
        ###########################################################
        self.current_d = 2.0 * (self.current_optimal -
                                self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = torch.sqrt(torch.abs(self.c)) * torch.sign(self.c)

        self.gamma = 0.5 * (
            self.gamma_pan * (1.0 + self.c_hat) +
            self.gamma_osc * (1.0 - self.c_hat)
        )

        self.contraction_rate = self.eta + torch.abs(self.current_d) * (
            self.gamma - self.eta
        )

        self.r = self.contraction_rate * self.r

    def _trim(
        self, new_bounds: torch.Tensor, global_bounds: torch.Tensor
    ) -> torch.Tensor:
        """
        Adjust the new_bounds and verify that they adhere to global_bounds and minimum_window.

        Parameters
        ----------
        new_bounds : np.ndarray / Tensor, Size d x 2
            The proposed new_bounds that (may) need adjustment.

        global_bounds : np.ndarray / Tensor, Size d x 2
            The maximum allowable bounds for each parameter.

        Returns
        -------
        new_bounds : np.ndarray / Tensor
            The adjusted bounds after enforcing constraints.
        """
        # sort bounds
        new_bounds = torch.sort(new_bounds).values.detach().clone()
        global_bounds = global_bounds.detach().clone()

        # print("sorted new bounds in _trim:", new_bounds)

        # Validate each parameter's bounds against the global_bounds
        for i, pbounds in enumerate(new_bounds):
            # If the one of the bounds is outside the global bounds, reset the bound to the global bound
            # This is expected to happen when the window is near the global bounds, no warning is issued
            # print("pbounds", pbounds, global_bounds[i])
            if pbounds[0] < global_bounds[i, 0]:
                # print("FIXING", pbounds[0], global_bounds[i, 0])
                pbounds[0] = global_bounds[i, 0]

            if pbounds[1] > global_bounds[i, 1]:
                # print("FIXING")
                pbounds[1] = global_bounds[i, 1]

            # If a lower bound is greater than the associated global upper bound, reset it to the global lower bound
            if pbounds[0] > global_bounds[i, 1]:
                pbounds[0] = global_bounds[i, 0]
                warn(
                    "\nDomain Reduction Warning:\n"
                    + "A parameter's lower bound is greater than the global upper bound."
                    + "The offensive boundary has been reset."
                    + "Be cautious of subsequent reductions.",
                    stacklevel=2,
                )

            # If an upper bound is less than the associated global lower bound, reset it to the global upper bound
            if pbounds[1] < global_bounds[i, 0]:
                pbounds[1] = global_bounds[i, 1]
                warn(
                    "\nDomain Reduction Warning:\n"
                    + "A parameter's lower bound is greater than the global upper bound."
                    + "The offensive boundary has been reset."
                    + "Be cautious of subsequent reductions.",
                    stacklevel=2,
                )

        # print("new_bounds here", new_bounds)

        # Adjust new_bounds to ensure they respect the minimum window width for each parameter
        for i, pbounds in enumerate(new_bounds):
            current_window_width = abs(pbounds[0] - pbounds[1])

            # If the window width is less than the minimum allowable width, adjust it
            # Note that when minimum_window < width of the global bounds one side always has more space than required
            # print("minimum_window", self.minimum_window)
            if current_window_width < self.minimum_window[i]:
                width_deficit = (
                    self.minimum_window[i] - current_window_width) / 2.0
                available_left_space = abs(global_bounds[i, 0] - pbounds[0])
                available_right_space = abs(global_bounds[i, 1] - pbounds[1])

                # determine how much to expand on the left and right
                expand_left = min(width_deficit, available_left_space)
                expand_right = min(width_deficit, available_right_space)

                # calculate the deficit on each side
                expand_left_deficit = width_deficit - expand_left
                expand_right_deficit = width_deficit - expand_right

                # shift the deficit to the side with more space
                adjust_left = expand_left + max(expand_right_deficit, 0)
                adjust_right = expand_right + max(expand_left_deficit, 0)

                # adjust the bounds
                pbounds[0] -= adjust_left
                pbounds[1] += adjust_right
        # DEBUGGING
        #############################################################################
        # print("new_bounds.T", new_bounds.T)
        return new_bounds

    def _window_bounds_compatibility(self, global_bounds: np.ndarray):
        """Check if global bounds are compatible with the minimum window sizes.

        Parameters
        ----------
        global_bounds : np.ndarray
            The maximum allowable bounds for each parameter.

        Raises
        ------
        ValueError
            If global bounds are not compatible with the minimum window size.
        """
        for i, entry in enumerate(global_bounds):
            global_window_width = abs(entry[1] - entry[0])
            if global_window_width < self.minimum_window[i]:
                raise ValueError(
                    "Global bounds are not compatible with the minimum window size."
                )

    def _create_bounds(self, parameters: dict, bounds: np.ndarray) -> dict:
        """Create a dictionary of bounds for each parameter.

        Parameters
        ----------
        parameters : dict
            The parameters for which to create the bounds.

        bounds : np.ndarray
            The bounds for each parameter.
        """
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    def transform(self, train_x, train_y) -> torch.Tensor:
        """Transform the bounds of the target space.

        Parameters
        ----------
        target_space : TargetSpace
            TargetSpace this DomainTransformer operates on.

        Returns
        -------
        dict
            The new bounds of each parameter.
        """
        self._update(train_x=train_x, train_y=train_y)

        # print("r after _update:", self.r)

        new_bounds = torch.stack(
            [self.current_optimal - 0.5 * self.r,
                self.current_optimal + 0.5 * self.r]
        ).T
        # print("new bounds in transform fn:", new_bounds)

        new_bounds = self._trim(new_bounds, self.original_bounds)

        # print("new bounds after _trim:", new_bounds)

        self.bounds.append(new_bounds)
        return new_bounds
