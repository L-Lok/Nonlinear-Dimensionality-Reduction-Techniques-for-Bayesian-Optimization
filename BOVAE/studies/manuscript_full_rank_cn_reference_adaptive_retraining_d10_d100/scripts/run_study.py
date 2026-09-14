#!/usr/bin/env python3
"""Run, aggregate, plot, or verify full-rank adaptive retraining."""

from pathlib import Path

from bo_vae_sdr.study_tools import cli


if __name__ == "__main__":
    raise SystemExit(cli(Path(__file__).resolve().parents[1], expected_kernel="matern52"))
