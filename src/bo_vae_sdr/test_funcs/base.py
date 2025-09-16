"""
Abstract class for test functions.
"""

import torch


def check_dimension(func):
    """
    Decorator to check the dimension of the input tensor.
    """

    def wrapper(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim == 2 and x.shape[1] == self.dim, \
            "Input tensor has incorrect dimension." \
            f"Expected {(x.shape[0], self.dim)}, got {x.shape}"
        return func(self, x)
    return wrapper


class BaseTestFunction:
    """
    Base class for test functions.
    """

    def __init__(self, dim, bounds=None):
        assert dim > 0 and isinstance(
            dim, int), "Dimension must be a positive integer."

        self.dim = dim
        self.optimal_input = None
        self.optimal_value = None

        if bounds is None:
            self.bounds = None
        else:
            assert bounds.dim() == 2, "bounds must be a 2D tensor."
            assert bounds.shape[1] == 2, "bounds must have 2 rows."
            assert bounds.shape[0] == self.dim, \
                "bounds must have the same dimension as the test function."
            self.bounds = bounds
        self.name = "BaseTestFunction"

    def bounds_suffix(self) -> str:
        # check if bounds is same for all dimensions
        if torch.all(self.bounds[:, 0] == self.bounds[0, 0]) and \
                torch.all(self.bounds[:, 1] == self.bounds[0, 1]):
            self.bounds = torch.tensor(
                [self.bounds[0, 0], self.bounds[0, 1]]).repeat(self.dim, 1)
            return f"bounds_{self.bounds[0, 0]}_{self.bounds[0, 1]}"
        else:
            bounds_str = "".join(
                [
                    f"_{self.bounds[i, 0]}_{self.bounds[i, 1]}"
                    for i in range(self.dim)
                ]
            )
            return f"bounds_{bounds_str}"

    def __call__(self, x) -> torch.Tensor:
        """
        Alias for func method.
        Evaluates the test function at the given point x.
        """
        return None

    @check_dimension
    def func(self, x) -> None:
        """
        Evaluates the test function at the given point x.
        """
        return x


class ShiftedTestFunction(BaseTestFunction):
    """
    A class for shifted inputs and outputs function
    Used as a sanity check for the optimization algorithms.    
    """

    def __init__(
        self,
        test_func: BaseTestFunction,
        shift_x: torch.Tensor | None = None,
        shift_y: float = 0.0,
    ):
        """
        Parameters:
            test_func (BaseTestFunction): The test function.
            shift_x (torch.Tensor): The shift for the input. dim x 1 shape.
            shift_y (float): The shift for the output.
        """
        self.test_func = test_func
        super().__init__(
            self.test_func.dim,
            self.test_func.bounds
        )

        if shift_x is None:
            shift_x = torch.zeros((self.dim, 1))

        assert shift_x.dim() == 2, "shift_x must be a 2D tensor."
        assert shift_x.shape[1] == 1, "shift_x must be a column vector."
        assert shift_x.shape[0] == test_func.dim, \
            "shift_x must have the same dimension as the test function."

        self.shift_x = shift_x
        self.shift_y = shift_y

        self.dim = test_func.dim
        self.optimal_input = test_func.optimal_input + self.shift_x
        self.optimal_value = test_func.optimal_value + self.shift_y

        self.name = f"{self.name}_shift_x_{shift_x.detach().squeeze(1).numpy()}_shift_y_{shift_y}"

    def func(self, x) -> None:
        return self.test_func.func(x + self.shift_x) + self.shift_y


class NoisyTestFunction(BaseTestFunction):
    """
    A class for noisy outputs function
    """

    def __init__(
        self,
        test_func: BaseTestFunction,
        noise_sigma_y: float = 1e-2,
    ):
        """
        Parameters:
            test_func (BaseTestFunction): The test function.
            noise_sigma_y (float): The noise level (Gaussian distribution) for the output.
        """
        self.test_func = test_func
        super().__init__(
            self.test_func.dim,
            self.test_func.bounds
        )

        self.dim = test_func.dim
        self.optimal_input = test_func.optimal_input
        self.optimal_value = test_func.optimal_value

        self.noise_sigma_y = noise_sigma_y

        self.name = f"{test_func.name}_noisy_{noise_sigma_y}"

    def func(self, x) -> None:
        func_value = self.test_func.func(x)
        func_noise = self.noise_sigma_y * torch.randn(func_value.shape).to(func_value)
        return func_value + func_noise
