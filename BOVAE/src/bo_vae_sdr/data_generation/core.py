"""Reproducible data materialization for the public manuscript workflows."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.curved_preimage import (
    CurvedPreimageProblem,
    build_curved_preimage_artifact,
)

from ..pipeline.core import canonical_problem_key
from ..pipeline.surrogate import make_problem
from ..pipeline.transforms import BoxTransform, map_between_boxes
from ..vae.artifacts import VAETrainingConfig, generate_pretraining_data


def read_json(path: Path | str) -> dict[str, Any]:
    """Read one JSON object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_torch(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _prepare_outputs(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        listed = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to replace existing output(s): {listed}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def resolve_workspace_path(value: str | Path, workspace_root: Path) -> Path:
    """Resolve paths exactly like publication configs rooted above ``BOVAE``."""

    path = Path(value)
    return path if path.is_absolute() else Path(workspace_root) / path


def derived_initial_design_seed(
    *,
    problem: str,
    dim: int,
    seed: int,
    namespace: str | None,
) -> tuple[int, str]:
    """Return a deterministic generator seed and its human-readable source."""

    if namespace is None:
        return int(seed), str(int(seed))
    source = f"{namespace}|{problem}|D={int(dim)}|seed={int(seed)}"
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1), source


def materialize_vae_data(
    config: VAETrainingConfig,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate and save the train/validation tensors used for VAE pretraining."""

    output_dir = Path(output_dir)
    train_path = output_dir / "train_data.pt"
    validation_path = output_dir / "validation_data.pt"
    config_path = output_dir / "vae_config.json"
    metadata_path = output_dir / "data_metadata.json"
    _prepare_outputs(
        [train_path, validation_path, config_path, metadata_path],
        overwrite=overwrite,
    )

    train_x, validation_x, metadata = generate_pretraining_data(config)
    dtype = config.dtype_obj()
    _save_torch(
        train_path,
        (train_x, torch.ones((train_x.shape[0], 1), dtype=dtype)),
    )
    _save_torch(
        validation_path,
        (validation_x, torch.ones((validation_x.shape[0], 1), dtype=dtype)),
    )
    _write_json(config_path, asdict(config))
    _write_json(metadata_path, metadata)
    return {
        "train_data": str(train_path),
        "validation_data": str(validation_path),
        "vae_config": str(config_path),
        "data_metadata": str(metadata_path),
        "train_size": int(train_x.shape[0]),
        "validation_size": int(validation_x.shape[0]),
    }


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be 'float32' or 'float64'")


def _configured_bounds(
    payload: dict[str, Any],
    *,
    dim: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    lower = payload.get("problem_bound_lower")
    upper = payload.get("problem_bound_upper")
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError("problem bounds must provide both lower and upper values")
    return torch.tensor([[float(lower), float(upper)]] * dim, dtype=dtype)


def _training_data_design(
    payload: dict[str, Any],
    *,
    workspace_root: Path,
    objective_bounds: torch.Tensor,
    dim: int,
    size: int,
    generator_seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    checkpoint = resolve_workspace_path(payload["vae_checkpoint"], workspace_root)
    train_path = checkpoint / "train_data.pt"
    if not train_path.is_file():
        raise FileNotFoundError(f"missing VAE training data: {train_path}")
    stored = torch.load(train_path, map_location="cpu", weights_only=False)
    train_x = stored[0] if isinstance(stored, (tuple, list)) else stored
    train_x = torch.as_tensor(train_x, dtype=dtype)
    if train_x.ndim != 2 or train_x.shape[1] != dim or train_x.shape[0] < size:
        raise ValueError("VAE training data cannot supply the requested initial design")
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    indices = torch.randperm(train_x.shape[0], generator=generator)[:size]
    selected = train_x[indices].contiguous()
    vae_bounds = torch.tensor(
        [
            [
                float(payload.get("vae_input_lower", -5.0)),
                float(payload.get("vae_input_upper", 5.0)),
            ]
        ]
        * dim,
        dtype=dtype,
    )
    return map_between_boxes(
        selected,
        BoxTransform(vae_bounds, name="vae_input"),
        BoxTransform(objective_bounds, name="objective"),
        clip_source_unit=True,
    )


def materialize_initial_design(
    config_path: Path | str,
    output_path: Path | str,
    *,
    workspace_root: Path | str,
    seed: int | None = None,
    seed_namespace: str | None = None,
    source: str = "uniform",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate one matched BO-VAE/ambient/EGORSE initial design."""

    payload = read_json(config_path)
    problem_key = str(payload["problem"])
    dim = int(payload["dim"])
    size = int(payload["initial_points"])
    run_seed = int(payload["seed"] if seed is None else seed)
    dtype = _dtype_from_name(str(payload.get("dtype", "float64")))
    workspace_root = Path(workspace_root)
    output_path = Path(output_path)
    if output_path.suffix != ".pt":
        raise ValueError("initial-design output must have a .pt suffix")
    npz_path = output_path.with_suffix(".npz")
    metadata_path = output_path.with_suffix(".json")
    _prepare_outputs([output_path, npz_path, metadata_path], overwrite=overwrite)

    artifact_value = payload.get("curved_preimage_artifact_path")
    metadata_value = payload.get("curved_preimage_metadata_path")
    artifact_path = (
        str(resolve_workspace_path(artifact_value, workspace_root))
        if artifact_value
        else None
    )
    curved_metadata_path = (
        str(resolve_workspace_path(metadata_value, workspace_root))
        if metadata_value
        else None
    )
    declared_bounds = _configured_bounds(payload, dim=dim, dtype=dtype)
    objective = make_problem(
        problem_key,
        dim,
        bounds=declared_bounds,
        curved_preimage_artifact_path=artifact_path,
        curved_preimage_metadata_path=curved_metadata_path,
    )
    objective_bounds = objective.bounds.to(dtype=dtype, device="cpu")
    generator_seed, seed_source = derived_initial_design_seed(
        problem=problem_key,
        dim=dim,
        seed=run_seed,
        namespace=seed_namespace,
    )
    if source == "uniform":
        generator = torch.Generator(device="cpu").manual_seed(generator_seed)
        unit = torch.rand((size, dim), generator=generator, dtype=dtype)
        x_objective = BoxTransform(
            objective_bounds, name="objective"
        ).from_unit(unit)
    elif source == "vae-training-data":
        x_objective = _training_data_design(
            payload,
            workspace_root=workspace_root,
            objective_bounds=objective_bounds,
            dim=dim,
            size=size,
            generator_seed=generator_seed,
            dtype=dtype,
        )
    else:
        raise ValueError("source must be 'uniform' or 'vae-training-data'")

    y_internal = objective.func(x_objective).detach().cpu().reshape(-1, 1)
    y_original = -y_internal
    if not torch.isfinite(y_original).all():
        raise ValueError("generated objective values are not finite")
    design = {
        "artifact_type": "manuscript_initial_design_v1",
        "problem_key": canonical_problem_key(problem_key, dim),
        "D": dim,
        "seed": run_seed,
        "generator_seed": generator_seed,
        "generator_seed_source": seed_source,
        "generator": "torch.Generator(device='cpu')",
        "source": source,
        "dtype": str(payload.get("dtype", "float64")),
        "bounds": objective_bounds,
        "x_obj": x_objective.detach().cpu(),
        "y_original": y_original,
        "y_internal": y_internal,
        "objective_values_verified": True,
        "objective_convention": "original minimization; internal=-original",
    }
    _save_torch(output_path, design)
    _save_npz(
        npz_path,
        {
            "x_obj": design["x_obj"].numpy(),
            "y_original": y_original.numpy(),
            "y_internal": y_internal.numpy(),
            "evaluation_elapsed_seconds": np.zeros(size, dtype=np.float64),
        },
    )
    sidecar = {
        key: value
        for key, value in design.items()
        if not isinstance(value, torch.Tensor)
    }
    sidecar.update(
        {
            "pt_path": str(output_path),
            "npz_path": str(npz_path),
        }
    )
    _write_json(metadata_path, sidecar)
    return sidecar


def materialize_curved_preimage_artifact(
    *,
    base_function: str,
    dim: int,
    alpha: float,
    output_dir: Path | str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build, save, and reload one deterministic curved-preimage problem."""

    output_dir = Path(output_dir)
    artifact_path = output_dir / "problem.npz"
    metadata_path = output_dir / "metadata.json"
    _prepare_outputs([artifact_path, metadata_path], overwrite=overwrite)
    arrays, metadata = build_curved_preimage_artifact(
        base_function,
        int(dim),
        alpha=float(alpha),
    )
    _save_npz(artifact_path, arrays)
    _write_json(metadata_path, metadata)
    reference = CurvedPreimageProblem.from_artifact(artifact_path, metadata_path)
    return {
        "problem_id": reference.problem_id,
        "artifact": str(artifact_path),
        "metadata": str(metadata_path),
        "D": reference.dim,
        "d_e": reference.effective_dim,
        "alpha": reference.alpha,
    }


__all__ = [
    "derived_initial_design_seed",
    "materialize_curved_preimage_artifact",
    "materialize_initial_design",
    "materialize_vae_data",
    "read_json",
    "resolve_workspace_path",
]
