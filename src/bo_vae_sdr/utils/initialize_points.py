"""
Initialize a set of points at random within the given bounds.
"""
import torch


def initialize_points_uniformly(
    n_points: int,
    dim: int,
    bounds: torch.Tensor,
):
    """
    Initialize a set of points uniformly at random within the given bounds.

    Args:
        n_points (int): Number of points to generate.
        dim (int): Dimension of the points.
        bounds (torch.Tensor): Bounds for each dimension, shape (dim, 2).

    Returns:
        torch.Tensor: A tensor of shape (n_points, dim) containing the points.
    """
    assert bounds.dim() == 2, "bounds must be a 2D tensor."
    assert bounds.shape[0] == dim, "bounds must have the same dimension as the points."
    assert bounds.shape[1] == 2, "bounds must have 2 rows in the second dimension."

    return torch.rand(n_points, dim).to(bounds.device) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
