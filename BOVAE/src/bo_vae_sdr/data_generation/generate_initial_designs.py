#!/usr/bin/env python3
"""Generate matched initial designs for publication pipeline configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import materialize_initial_design, read_json, resolve_workspace_path

BOVAE_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = BOVAE_ROOT.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate .pt and .npz initial designs from one publication PipelineConfig."
        )
    )
    parser.add_argument("config", type=Path, help="pipeline example/config JSON")
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="run seed; repeat to generate matched designs for several seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for seed_<seed>.pt/.npz/.json (default: configured path)",
    )
    parser.add_argument(
        "--source",
        choices=("uniform", "vae-training-data"),
        default="uniform",
        help="sample the objective box or select from checkpoint train_data.pt",
    )
    parser.add_argument(
        "--seed-namespace",
        help=(
            "optional deterministic namespace; use "
            "'manuscript-full-rank-cn-reference|20260812' for newly generated "
            "full-rank/Figure-1 designs"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    configured_output = resolve_workspace_path(
        config["initial_design_path"], WORKSPACE_ROOT
    )
    seeds = args.seeds or [int(config["seed"])]
    output_dir = args.output_dir or configured_output.parent
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    results = []
    for seed in seeds:
        output_path = output_dir / f"seed_{int(seed)}.pt"
        results.append(
            materialize_initial_design(
                args.config,
                output_path,
                workspace_root=WORKSPACE_ROOT,
                seed=seed,
                seed_namespace=args.seed_namespace,
                source=args.source,
                overwrite=args.overwrite,
            )
        )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
