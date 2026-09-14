"""Command-line pretraining entry point for manuscript VAE checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import VAETrainingConfig, train_vae


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to vae_config.json")
    parser.add_argument("output", type=Path, help="Checkpoint output directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = VAETrainingConfig.from_json(args.config)
    result = train_vae(config, args.output, device=args.device, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
