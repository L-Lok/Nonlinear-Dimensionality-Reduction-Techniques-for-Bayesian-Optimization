"""Experiment bookkeeping shared by retained EGORSE cells."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from benchmarks.base import sample_uniform_design
from egorse.embeddings import VARIANT_METHODS


TRACKED_PACKAGES = [
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "nlopt",
    "cvxopt",
    "smt",
    "pyoptsparse",
    "sb-arch-opt",
    "segomoe",
]

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]


@dataclass(frozen=True)
class BudgetSpec:
    """Paper-style EGORSE evaluation budget."""

    variant: str
    effective_dim: int
    max_nb_it: int
    max_nb_it_sub: int

    @property
    def method_count(self) -> int:
        return len(VARIANT_METHODS[self.variant])

    @property
    def outer_iterations(self) -> int:
        if self.method_count == 1:
            return 2 * self.max_nb_it
        return self.max_nb_it

    @property
    def enrichment_evaluations(self) -> int:
        return self.outer_iterations * self.method_count * self.max_nb_it_sub


def parse_doe_size(spec: int | str, dim: int) -> int:
    """Parse Section V.C initial DoE sizes: 5, d, and 2d."""

    if isinstance(spec, int):
        return spec
    text = str(spec).strip().lower()
    if text == "d":
        return dim
    if text == "2d":
        return 2 * dim
    return int(text)


def initial_design_for_run(dim: int, doe_size: int, seed: int) -> np.ndarray:
    """Generate the reusable initial DoE for a problem x DoE x run cell."""

    return sample_uniform_design(dim, doe_size, seed)


def package_versions() -> dict[str, str]:
    """Collect package versions for per-evaluation metadata."""

    versions: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = _module_version(package)
        if version in {None, "0.0.0"}:
            version = _module_version(package)
        versions[package] = str(version)
    for env_name in THREAD_ENV_VARS:
        versions[f"env_{env_name}"] = os.environ.get(env_name, "")
    return versions


def _module_version(package: str) -> str:
    module_name = {"scikit-learn": "sklearn"}.get(package, package)
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return "not-installed"
    return str(getattr(module, "__version__", "unknown"))


def json_dumps_compact(value: object) -> str:
    """Stable compact JSON for CSV fields."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def records_to_csv_rows(
    records: Iterable[dict[str, object]],
    package_versions_json: str,
) -> list[dict[str, object]]:
    """Normalize optimizer records to the required raw CSV schema."""

    rows: list[dict[str, object]] = []
    for record in records:
        row = dict(record)
        row["x"] = json_dumps_compact(row.pop("x"))
        row["u"] = json_dumps_compact(row.pop("u"))
        if "subspace_u" in row:
            row["subspace_u"] = json_dumps_compact(row["subspace_u"])
        row["package_versions"] = package_versions_json
        if "embedding_metadata" in row:
            row["embedding_metadata"] = json_dumps_compact(row["embedding_metadata"])
        if "acquisition_metadata" in row:
            row["acquisition_metadata"] = json_dumps_compact(row["acquisition_metadata"])
        if "backend_versions" in row:
            row["backend_versions"] = json_dumps_compact(row["backend_versions"])
        if "backend_commits" in row:
            row["backend_commits"] = json_dumps_compact(row["backend_commits"])
        rows.append(row)
    return rows
