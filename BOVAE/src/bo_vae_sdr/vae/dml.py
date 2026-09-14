from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TargetNormalizer:
    """Min-max normalizer for DML objective values.

    The revised manuscript constructs DML positives/negatives after rescaling
    objective values to [0, 1].  BO-VAE applies this to original minimization
    values, then passes the normalized tensor into the triplet loss.
    """

    minimum: float
    maximum: float
    eps: float = 1e-12

    @classmethod
    def fit(cls, y: torch.Tensor, eps: float = 1e-12) -> "TargetNormalizer":
        flat = y.detach().reshape(-1).to(dtype=torch.float64)
        if flat.numel() == 0:
            raise ValueError("cannot fit DML target normalizer on empty tensor")
        return cls(minimum=float(flat.min().item()), maximum=float(flat.max().item()), eps=eps)

    @property
    def scale(self) -> float:
        return max(self.maximum - self.minimum, self.eps)

    @property
    def is_constant(self) -> bool:
        return self.maximum - self.minimum <= self.eps

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        y64 = y.to(dtype=torch.float64)
        if self.is_constant:
            return torch.zeros_like(y64)
        return ((y64 - self.minimum) / self.scale).clamp(0.0, 1.0)

    def inverse(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm.to(dtype=torch.float64) * self.scale + self.minimum

    def metadata(self) -> dict[str, Any]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "scale": self.scale,
            "constant": self.is_constant,
            "eps": self.eps,
        }


def normalize_dml_targets(y_original: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, TargetNormalizer]:
    normalizer = TargetNormalizer.fit(y_original, eps=eps)
    return normalizer.transform(y_original), normalizer


def dml_triplet_stats(y_normalized: torch.Tensor, threshold: float) -> dict[str, Any]:
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("DML threshold must be in (0, 1)")
    y = y_normalized.detach().reshape(-1, 1).to(dtype=torch.float64)
    n = int(y.shape[0])
    if n == 0:
        return {
            "n_points": 0,
            "positive_pairs": 0,
            "negative_pairs": 0,
            "valid_triplets": 0,
            "anchors_with_positive": 0,
            "anchors_with_negative": 0,
            "anchors_with_triplet": 0,
        }
    distance = torch.cdist(y, y, p=1)
    eye = torch.eye(n, dtype=torch.bool, device=y.device)
    positive = (distance < float(threshold)) & ~eye
    negative = distance >= float(threshold)
    positives_per_anchor = positive.sum(dim=1)
    negatives_per_anchor = negative.sum(dim=1)
    valid_triplets = positives_per_anchor * negatives_per_anchor
    return {
        "n_points": n,
        "positive_pairs": int(positive.sum().item()),
        "negative_pairs": int(negative.sum().item()),
        "valid_triplets": int(valid_triplets.sum().item()),
        "anchors_with_positive": int((positives_per_anchor > 0).sum().item()),
        "anchors_with_negative": int((negatives_per_anchor > 0).sum().item()),
        "anchors_with_triplet": int((valid_triplets > 0).sum().item()),
        "threshold": float(threshold),
        "target_min": float(y.min().item()),
        "target_max": float(y.max().item()),
    }


def resolve_dml_threshold(
    y_normalized: torch.Tensor,
    threshold: float,
) -> tuple[float, dict[str, Any]]:
    """Validate and report the fixed manuscript DML triplet threshold."""

    del y_normalized
    requested = float(threshold)
    if not 0.0 < requested < 1.0:
        raise ValueError("DML threshold must be in (0, 1)")
    metadata = {
        "threshold_strategy": "fixed",
        "requested_threshold": requested,
        "effective_threshold": requested,
    }
    return requested, metadata


def manuscript_triplet_config(threshold: float = 0.01, eta: float = 0.2) -> dict[str, Any]:
    return {
        "type": "triplet",
        "threshold": float(threshold),
        "eta": float(eta),
    }
