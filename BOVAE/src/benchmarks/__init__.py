"""Canonical benchmark package for the maintained codebase."""

from .ackley import Ackley
from .base import BaseTestFunction, sample_uniform_design
from .canonical_levy import CanonicalLevy
from .canonical_styblinski_tang import CanonicalStyblinskiTang
from .curved_preimage import (
    CurvedPreimage,
    CurvedPreimageProblem,
    build_curved_preimage_artifact,
)
from .rastrigin import Rastrigin
from .rosenbrock import Rosenbrock

__all__ = [
    "Ackley",
    "BaseTestFunction",
    "CanonicalLevy",
    "CanonicalStyblinskiTang",
    "CurvedPreimage",
    "CurvedPreimageProblem",
    "Rastrigin",
    "Rosenbrock",
    "build_curved_preimage_artifact",
    "sample_uniform_design",
]
