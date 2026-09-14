#!/usr/bin/env python3
"""Run or resume the retained curved-preimage EGORSE comparison cells."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.curved_preimage import CurvedPreimageProblem
from egorse.acquisition import ApproximateConstrainedEIConfig
from egorse.optimizer import EGORSEConfig, EGORSEOptimizer


STUDY = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def cells() -> list[dict[str, Any]]:
    study = read_json(STUDY / "configs/study.json")
    rows = []
    for problem in study["problems"]:
        for method in study["egorse_methods"]:
            for seed in study["egorse_seeds"]:
                rows.append({**problem, "method": method, "seed": int(seed)})
    return rows


def optimizer(cell: dict[str, Any]) -> EGORSEOptimizer:
    settings = read_json(STUDY / "configs/egorse.json")
    effective_dim = int(cell["d_e"]) * (2 if cell["method"] == "egorse_2de" else 1)
    acquisition = ApproximateConstrainedEIConfig(
        n_candidates=int(settings["n_candidates"]),
        n_local_starts=int(settings["n_local_starts"]),
        gp_restarts=int(settings["gp_restarts"]),
        constraint_strategy=settings["constraint_strategy"],
        global_optimizer=settings["global_optimizer"],
        isres_maxeval=int(settings["isres_maxeval"]),
        local_refiner=settings["local_refiner"],
        surrogate_backend=settings["surrogate_backend"],
        smt_theta0=float(settings["smt_theta0"]),
        smt_n_start=int(settings["smt_n_start"]),
        smt_nugget_sequence=tuple(settings["smt_nugget_sequence"]),
        duplicate_tolerance=float(settings["duplicate_tolerance"]),
    )
    return EGORSEOptimizer(
        EGORSEConfig(
            variant=settings["variant"],
            effective_dim=effective_dim,
            max_nb_it=int(settings["max_nb_it"]),
            max_nb_it_sub=int(settings["max_nb_it_sub_factor"]) * effective_dim,
            backend_mode=settings["backend_mode"],
            backend_exact=bool(settings["backend_exact"]),
            backend_fallback_reason=settings["backend_fallback_reason"],
            allow_approximate_backend=bool(settings["allow_approximate_backend"]),
            gamma_tol=float(settings["gamma_tol"]),
            acquisition=acquisition,
        )
    )


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    problem_id = cell["problem_id"]
    output = STUDY / "results/egorse" / problem_id / cell["method"] / f"seed_{cell['seed']}"
    if (output / "summary.json").is_file():
        return {"cell": f"{problem_id}:{cell['method']}:{cell['seed']}", "status": "completed"}
    artifact = STUDY / "inputs/problem_artifacts" / problem_id / "problem.npz"
    metadata = STUDY / "inputs/problem_artifacts" / problem_id / "metadata.json"
    initial = np.load(STUDY / "inputs/initial_designs" / problem_id / f"seed_{cell['seed']}.npz")
    problem = CurvedPreimageProblem.from_artifact(artifact, metadata)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pkl"
    resume_state = None
    resume_records = None
    if checkpoint.is_file():
        with checkpoint.open("rb") as handle:
            resume_state = pickle.load(handle)
        resume_records = list(resume_state["history_records"])

    def save_state(state: dict[str, object]) -> None:
        temporary = checkpoint.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, checkpoint)

    history = optimizer(cell).run(
        problem,
        budget=800,
        init_X=np.asarray(initial["x_obj"], dtype=float),
        init_Y=np.asarray(initial["y_original"], dtype=float),
        init_wall_times=np.asarray(initial["evaluation_elapsed_seconds"], dtype=float),
        seed=int(cell["seed"]),
        state_callback=save_state,
        resume_state=resume_state,
        resume_records=resume_records,
    )
    with (output / "evaluation_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in history.records:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    summary = {
        "problem": problem_id,
        "method": cell["method"],
        "seed": cell["seed"],
        "n_observations": len(history.records),
        "final_best_original": history.best_valid_value(),
        "surrogate_backend": read_json(STUDY / "configs/egorse.json")["surrogate_backend"],
    }
    write_json(output / "summary.json", summary)
    return {"cell": f"{problem_id}:{cell['method']}:{cell['seed']}", "status": "ran"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    selected = [row for row in cells() if args.selector is None or args.selector in json.dumps(row)]
    if args.max_runs is not None:
        selected = selected[: args.max_runs]
    print(json.dumps([run_cell(row) for row in selected], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
