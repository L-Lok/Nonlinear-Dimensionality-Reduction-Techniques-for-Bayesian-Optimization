#!/usr/bin/env python3
"""Generate deterministic VAE pretraining and validation tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import materialize_vae_data
from ..vae import VAETrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the train_data.pt and validation_data.pt used by VAE pretraining."
    )
    parser.add_argument("config", type=Path, help="VAETrainingConfig JSON file")
    parser.add_argument("output", type=Path, help="output checkpoint/data directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the four generated data/config files if they exist",
    )
    args = parser.parse_args()
    result = materialize_vae_data(
        VAETrainingConfig.from_json(args.config),
        args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
