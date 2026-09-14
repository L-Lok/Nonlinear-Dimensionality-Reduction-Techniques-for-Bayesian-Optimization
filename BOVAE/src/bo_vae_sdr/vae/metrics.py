"""Deep-metric-learning loss used by the manuscript BO-VAE method."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import Tensor


class TripletLossTorch:
    """Soft triplet loss constructed from normalized objective distances."""

    def __init__(
        self,
        threshold: float,
        eta: float = 0.2,
    ) -> None:
        self.threshold = float(threshold)
        if eta <= 0:
            raise ValueError("eta must be positive")
        self.eta = float(eta)

    def triplet_loss_values(
        self,
        positive_embedding_distances: Tensor,
        negative_embedding_distances: Tensor,
        positive_target_distances: Tensor,
        negative_target_distances: Tensor,
    ) -> Tensor:
        """Return the manuscript loss for broadcast-compatible triplet distances.

        This elementwise form is also used by the visualization notebook so the
        plotted surface and the loss optimized during DML retraining cannot
        silently diverge.
        """
        scores = positive_embedding_distances - negative_embedding_distances
        per_triplet = torch.logaddexp(torch.zeros_like(scores), scores)
        per_triplet = (
            per_triplet
            * self.smooth_indicator(self.threshold - positive_target_distances)
            / self.smooth_indicator(self.threshold)
            * self.smooth_indicator(negative_target_distances - self.threshold)
            / self.smooth_indicator(1.0 - self.threshold)
        )
        valid_triplet = (positive_target_distances < self.threshold) & (
            negative_target_distances >= self.threshold
        )
        return torch.where(valid_triplet, per_triplet, torch.zeros_like(per_triplet))

    def build_loss_matrix(
        self,
        embs: Tensor,
        ys: Tensor,
    ) -> Tensor:
        embedding_distances = torch.cdist(embs, embs, p=2)
        objective_distances = torch.cdist(ys.reshape(-1, 1), ys.reshape(-1, 1), p=1)
        positives = embedding_distances.where(
            objective_distances < self.threshold,
            torch.zeros((), dtype=embs.dtype, device=embs.device),
        )
        negatives = embedding_distances.where(
            objective_distances >= self.threshold,
            torch.zeros((), dtype=embs.dtype, device=embs.device),
        )
        loss = 0.0 * embs.sum()
        denominator = torch.zeros((), dtype=embs.dtype, device=embs.device)
        for index in range(embs.shape[0]):
            positive = positives[index][positives[index] > 0]
            negative = negatives[index][negatives[index] > 0]
            positive_y = objective_distances[index][positives[index] > 0]
            negative_y = objective_distances[index][negatives[index] > 0]
            pairs = torch.cartesian_prod(positive, negative)
            pairs_y = torch.cartesian_prod(positive_y, negative_y)
            per_triplet = self.triplet_loss_values(
                positive_embedding_distances=pairs[:, 0],
                negative_embedding_distances=pairs[:, 1],
                positive_target_distances=pairs_y[:, 0],
                negative_target_distances=pairs_y[:, 1],
            )
            denominator = denominator + (per_triplet > 0).to(embs.dtype).sum()
            loss = loss + per_triplet.sum()
        return loss / denominator.clamp_min(1.0)

    def smooth_indicator(self, value: Union[Tensor, float]) -> Union[Tensor, float]:
        if isinstance(value, float):
            return np.tanh(value / (2.0 * float(self.eta)))
        return torch.tanh(value / (2.0 * float(self.eta)))

    def __call__(
        self,
        embs: Tensor,
        ys: Tensor,
    ) -> Tensor:
        return self.build_loss_matrix(embs, ys)
