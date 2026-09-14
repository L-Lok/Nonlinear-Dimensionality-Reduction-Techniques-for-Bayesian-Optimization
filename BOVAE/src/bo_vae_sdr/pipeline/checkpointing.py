"""Small, portable helpers for pipeline checkpoint files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically save a tensor checkpoint in a CPU-portable form."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, *, map_location: torch.device | str = "cpu"
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


__all__ = ["load_checkpoint", "save_checkpoint"]
