#!/usr/bin/env python3
"""Generate deterministic curved-preimage benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.curved_preimage import problem_id

from .core import materialize_curved_preimage_artifact, read_json


BOVAE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STUDY = (
    BOVAE_ROOT / "studies/bovae_vs_egorse_curved_preimage_d10_d100"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the curved-preimage geometry consumed by BO-VAE and EGORSE."
    )
    parser.add_argument(
        "--study-config",
        type=Path,
        default=DEFAULT_STUDY / "configs/study.json",
        help="study JSON supplying the default problem grid and fixed alpha",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_STUDY / "inputs/problem_artifacts",
    )
    parser.add_argument(
        "--base",
        action="append",
        choices=("ackley", "branin", "rastrigin", "rosenbrock"),
        help="restrict generation to one or more base functions",
    )
    parser.add_argument(
        "--dim",
        action="append",
        type=int,
        choices=(10, 100),
        help="restrict generation to one or both ambient dimensions",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="override fixed_alpha from the study configuration",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    study = read_json(args.study_config)
    alpha = float(study["fixed_alpha"] if args.alpha is None else args.alpha)
    allowed_bases = set(args.base or [])
    allowed_dims = set(args.dim or [])
    specs = [
        row
        for row in study["problems"]
        if (not allowed_bases or str(row["base_function"]) in allowed_bases)
        and (not allowed_dims or int(row["D"]) in allowed_dims)
    ]
    if not specs:
        raise ValueError("the requested base/dimension filters select no problems")

    results = []
    for row in specs:
        base = str(row["base_function"])
        dim = int(row["D"])
        expected_id = problem_id(base, dim)
        if str(row["problem_id"]) != expected_id:
            raise ValueError(f"study problem id does not match {base}/D{dim}")
        results.append(
            materialize_curved_preimage_artifact(
                base_function=base,
                dim=dim,
                alpha=alpha,
                output_dir=args.output_root / expected_id,
                overwrite=args.overwrite,
            )
        )
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
