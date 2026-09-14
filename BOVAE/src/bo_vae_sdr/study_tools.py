"""Common orchestration, validation, aggregation, and plotting for studies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import PipelineConfig, run_pipeline


PIPELINE_FIELDS = {field.name for field in fields(PipelineConfig)}
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repository_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else WORKSPACE_ROOT / path)


def portable_path(value: str | Path) -> str:
    """Represent repository paths without embedding a workstation prefix."""

    path = Path(value)
    absolute = path if path.is_absolute() else WORKSPACE_ROOT / path
    try:
        return absolute.resolve().relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def pipeline_config(path: Path) -> PipelineConfig:
    """Load a current or retained ambient configuration."""

    payload = read_json(path)
    if "mode" not in payload:
        payload["mode"] = (
            "ambient_bo" if payload.get("sdr_method", "none") == "none" else "ambient_bo_sdr"
        )
        payload["dim"] = int(payload.pop("D"))
        payload["vae_checkpoint"] = ""
        payload["latent_dim"] = int(payload.get("latent_dim", payload["dim"]))
        if "lower" in payload:
            payload["problem_bound_lower"] = float(payload["lower"])
        if "upper" in payload:
            payload["problem_bound_upper"] = float(payload["upper"])
    if payload["mode"] in {"fixed_no_sdr", "retrain_dml"}:
        payload["sdr_method"] = "none"
    for key, value in list(payload.items()):
        if value and isinstance(value, str) and (
            key in {"output_dir", "vae_checkpoint"} or key.endswith("_path")
        ):
            payload[key] = _resolve_repository_path(value)
    return PipelineConfig(**{key: value for key, value in payload.items() if key in PIPELINE_FIELDS})


def config_paths(study: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted((study / "configs").rglob("*.json")):
        payload = read_json(path)
        if "mode" in payload or "D" in payload and "output_dir" in payload:
            paths.append(path)
    return paths


def example_config_path(study: Path, config_path: Path | str | None = None) -> Path:
    """Return the checked, user-selectable pipeline configuration for a study."""

    path = study / "configs/example.json" if config_path is None else Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"pipeline configuration does not exist: {path}")
    return path


def result_summary(output: Path, config: PipelineConfig) -> dict[str, Any] | None:
    """Read a full run summary or reconstruct one from a retained profile trace."""

    summary_path = output / "summary.json"
    if summary_path.is_file():
        return read_json(summary_path)
    profile_path = output / "profile_trace.json"
    if not profile_path.is_file():
        return None
    profile = read_json(profile_path)
    trace = profile.get("trajectory", profile.get("trace", []))
    if not trace:
        return None
    final = trace[-1]
    enrichment = final.get("enrichment_evaluation")
    evaluations = final.get("evaluation_count")
    complete = (
        enrichment is not None and int(enrichment) >= int(config.budget)
    ) or (
        evaluations is not None
        and int(evaluations) >= int(config.initial_points + config.budget)
    )
    if not complete:
        return None
    best_original = float(final["best_original"])
    return {
        "artifact_type": "retained_compact_profile_summary",
        "n_observations": int(evaluations or config.initial_points + config.budget),
        "final_best_internal": -best_original,
        "final_best_original": best_original,
        "elapsed_seconds": final.get("elapsed_seconds", final.get("total_time_seconds")),
    }


def run_config(
    study: Path,
    *,
    config_path: Path | str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run one explicit configuration instead of a generated study matrix."""

    path = example_config_path(study, config_path)
    config = pipeline_config(path)
    output = Path(config.output_dir)
    if result_summary(output, config) is not None:
        return {
            "config": portable_path(path),
            "output": portable_path(output),
            "status": "completed",
        }
    config = replace(config, resume=bool(resume and (output / "checkpoint.pt").is_file()))
    summary = run_pipeline(config)
    return {
        "config": portable_path(path),
        "output": portable_path(output),
        "status": "ran",
        "summary": summary,
    }


def verify_study(
    study: Path,
    *,
    expected_kernel: str | None,
    config_path: Path | str | None = None,
    check_artifacts: bool = False,
) -> dict[str, Any]:
    required = ("configs", "scripts")
    if check_artifacts:
        required += ("inputs", "results", "plot_data", "figures", "reports")
    missing_layout = [name for name in required if not (study / name).is_dir()]
    if missing_layout:
        raise FileNotFoundError(f"missing study directories: {missing_layout}")
    configs = [example_config_path(study, config_path)]
    kernels: set[str] = set()
    missing_inputs: list[str] = []
    missing_outputs: list[str] = []
    for path in configs:
        config = pipeline_config(path)
        kernels.add(config.gp_kernel)
        for value in (
            config.initial_design_path,
            config.vae_checkpoint if not config.mode.startswith("ambient_") else None,
            config.curved_preimage_artifact_path,
            config.curved_preimage_metadata_path,
        ):
            if value and not Path(value).exists():
                missing_inputs.append(str(value))
        if result_summary(Path(config.output_dir), config) is None:
            missing_outputs.append(config.output_dir)
    if expected_kernel is not None and kernels != {expected_kernel}:
        raise ValueError(f"expected kernel {expected_kernel}, found {sorted(kernels)}")
    if check_artifacts and missing_inputs:
        raise FileNotFoundError(f"missing configured inputs: {missing_inputs[:5]}")
    forbidden = ("audit_logs", "audit_pretrained_vae")
    bad_references = [
        str(path)
        for path in (study / "configs").rglob("*.json")
        if any(token in path.read_text(encoding="utf-8") for token in forbidden)
    ]
    if bad_references:
        raise ValueError(f"configs retain obsolete external references: {bad_references[:5]}")
    plot_files = sorted((study / "plot_data").glob("*.csv"))
    empty_plot_files = [str(path) for path in plot_files if path.stat().st_size == 0]
    if empty_plot_files:
        raise ValueError(f"empty plot data: {empty_plot_files}")
    egorse_results = sorted((study / "results/egorse").rglob("summary.json"))
    return {
        "study": study.name,
        "config_count": len(configs),
        "kernels": sorted(kernels),
        "completed_config_count": len(configs) - len(missing_outputs),
        "missing_output_count": len(missing_outputs),
        "missing_input_count": len(missing_inputs),
        "artifact_check": check_artifacts,
        "plot_data_count": len(plot_files),
        "egorse_result_count": len(egorse_results),
    }


def aggregate_study(
    study: Path,
    *,
    config_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in [example_config_path(study, config_path)]:
        config = pipeline_config(path)
        output = Path(config.output_dir)
        summary = result_summary(output, config)
        if summary is None:
            continue
        rows.append(
            {
                "config": portable_path(path),
                "output": portable_path(output),
                "problem": config.problem,
                "D": config.dim,
                "d": None if config.mode.startswith("ambient_") else config.latent_dim,
                "mode": config.mode,
                "seed": config.seed,
                "gp_kernel": config.gp_kernel,
                "sdr_method": config.sdr_method,
                "n_observations": summary.get("n_observations"),
                "final_best_original": summary.get("final_best_original"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            }
        )
    for summary_path in sorted((study / "results/egorse").rglob("summary.json")):
        summary = read_json(summary_path)
        backend = summary.get("backend", {})
        rows.append(
            {
                "config": "",
                "output": portable_path(summary_path.parent),
                "problem": summary.get("problem_id"),
                "D": summary.get("D"),
                "d": summary.get("optimizer_effective_dim", summary.get("d_e")),
                "mode": summary.get("method", "egorse"),
                "seed": summary.get("seed"),
                "gp_kernel": backend.get("kernel", "squared_exponential"),
                "sdr_method": "none",
                "n_observations": summary.get("n_observations"),
                "final_best_original": summary.get("final_best_original"),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            }
        )
    destination = study / "reports" / "run_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["config", "output", "problem", "D", "d", "mode", "seed"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_json(study / "reports" / "run_summary.json", rows)
    return rows


COLORS = {
    "fixed_sdr": "#0072B2",
    "fixed_no_sdr": "#7F7F7F",
    "retrain_sdr": "#D55E00",
    "retrain_dml_sdr": "#009E73",
    "ambient_sdr_original": "#6B5B95",
    "ambient_plain_bo": "#7F7F7F",
    "egorse_de": "#CC79A7",
    "egorse_2de": "#E69F00",
    "fixed": "#0072B2",
    "retrain": "#D55E00",
    "dml": "#009E73",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _ribbon(ax: plt.Axes, frame: pd.DataFrame, x: str, y: str, sd: str | None, group: str) -> None:
    for key, values in frame.groupby(group, sort=False):
        values = values.sort_values(x)
        label = str(values["method_label"].iloc[0]) if "method_label" in values else str(values.get("display_label", pd.Series([key])).iloc[0])
        color = COLORS.get(str(key))
        xv = values[x].to_numpy(float)
        yv = np.maximum(values[y].to_numpy(float), 1e-12)
        ax.plot(xv, yv, label=label, color=color)
        if sd and sd in values:
            sv = values[sd].fillna(0.0).to_numpy(float)
            ax.fill_between(xv, np.maximum(yv - sv, 1e-12), yv + sv, alpha=0.18, color=color)
    ax.grid(True, which="both", alpha=0.25)


def plot_study(study: Path) -> list[Path]:
    key = study.name
    figures = study / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if key == "manuscript_figure1_ackley_bo_sdr_ablation_d10":
        frame = pd.read_csv(study / "plot_data/figure1_ackley_bo_sdr_ablation_d10.csv")
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        _ribbon(ax, frame, "enrichment_evaluation", "mean_simple_regret", "sample_sd_simple_regret", "method")
        ax.set_yscale("log"); ax.set_xlabel("Enrichment evaluations"); ax.set_ylabel("Simple regret"); ax.legend()
        path = figures / "figure1_ackley_bo_sdr_ablation_d10"; _save(fig, path); outputs += [path.with_suffix(".png"), path.with_suffix(".pdf")]
    elif key == "manuscript_full_rank_cn_reference_sdr_ablation_d10_d100":
        frame = pd.read_csv(study / "plot_data/fixed_bovae_sdr_ablation_simple_regret_d10_d100.csv")
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for ax, ((problem, dim), values) in zip(axes.flat, frame.groupby(["problem", "D"], sort=False)):
            _ribbon(ax, values, "iteration", "mean_simple_regret", "sample_std_simple_regret", "method")
            ax.set_yscale("log"); ax.set_title(f"{str(problem).title()}, D={dim}")
        axes[-1, 0].set_xlabel("Enrichment evaluations"); axes[-1, 1].set_xlabel("Enrichment evaluations"); axes[0, 0].legend()
        path = figures / "fixed_bovae_sdr_ablation_simple_regret_d10_d100"; _save(fig, path); outputs += [path.with_suffix(".png"), path.with_suffix(".pdf")]
    elif key == "manuscript_full_rank_cn_reference_adaptive_retraining_d10_d100":
        for source, stem, x, xlabel in (
            ("three_bovae_simple_regret_by_iterations.csv", "three_bovae_simple_regret_by_iterations", "iteration", "Enrichment evaluations"),
            ("three_bovae_simple_regret_by_total_time.csv", "three_bovae_simple_regret_by_total_time", "total_time_minutes", "Total time (minutes)"),
        ):
            frame = pd.read_csv(study / "plot_data" / source)
            groups = list(frame.groupby(["D", "problem"], sort=False))
            fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
            for ax, ((dim, problem), values) in zip(axes.flat, groups):
                _ribbon(ax, values, x, "mean_simple_regret", "sample_std_simple_regret", "method")
                ax.set_yscale("log"); ax.set_title(f"{str(problem).title()}, D={dim}"); ax.set_xlabel(xlabel)
            axes[0, 0].legend(fontsize=8)
            path = figures / stem; _save(fig, path); outputs += [path.with_suffix(".png"), path.with_suffix(".pdf")]
    elif key == "manuscript_full_rank_cn_reference_profiles_d10_d100":
        for source, stem, x, xlabel in (
            ("performance_profiles_tau_1e1_1e3_1e5.csv", "performance_profiles_tau_1e1_1e3_1e5", "x", "Performance ratio"),
            ("data_profiles_tau_1e1_1e3_1e5.csv", "data_profiles_tau_1e1_1e3_1e5", "x", "Evaluations / (D + 1)"),
        ):
            frame = pd.read_csv(study / "plot_data" / source)
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
            for ax, ((dim, tau), values) in zip(axes.flat, frame.groupby(["D", "tau"], sort=False)):
                y = "proportion_solved"
                for method, rows in values.groupby("method", sort=False):
                    rows = rows.sort_values(x)
                    ax.plot(rows[x], rows[y], label=str(method), color=COLORS.get(str(method)))
                ax.set_title(f"D={dim}, tau={tau:g}"); ax.set_xlabel(xlabel); ax.set_ylim(-0.02, 1.02); ax.grid(True, alpha=0.25)
            axes[0, 0].legend(fontsize=7)
            path = figures / stem; _save(fig, path); outputs += [path.with_suffix(".png"), path.with_suffix(".pdf")]
    elif key == "latent_dimension_unweighted_figure_revision":
        frame = pd.read_csv(study / "plot_data/top5_mean_simple_regret_by_latent_dimension.csv")
        for dim, values in frame.groupby("ambient_dim"):
            problems = list(values["problem"].drop_duplicates())
            fig, axes = plt.subplots(1, len(problems), figsize=(4.2 * len(problems), 3.8), squeeze=False)
            for ax, problem in zip(axes.flat, problems):
                subset = values[values["problem"] == problem]
                _ribbon(ax, subset, "latent_dim", "mean_simple_regret", "sample_std_simple_regret", "pipeline")
                ax.set_yscale("log"); ax.set_title(str(problem).title()); ax.set_xlabel("Latent dimension")
            axes[0, 0].legend(fontsize=8)
            path = figures / f"D{int(dim)}_top5_mean_simple_regret_by_latent_dimension"; _save(fig, path); outputs += [path.with_suffix(".png"), path.with_suffix(".pdf")]
        floor = pd.read_csv(study / "plot_data/representation_floor_vs_simple_regret.csv")
        for dim, values in floor.groupby("ambient_dim"):
            fig, ax = plt.subplots(figsize=(6.2, 4.5))
            for pipeline, rows in values.groupby("pipeline"):
                ax.scatter(rows["best_ever_support_gap_mean"], rows["initial_inclusive_simple_regret_mean"], label=str(pipeline), color=COLORS.get(str(pipeline)))
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("VAE representation floor"); ax.set_ylabel("Simple regret"); ax.grid(True, which="both", alpha=0.25); ax.legend()
            path = figures / f"D{int(dim)}_vae_representation_floor_vs_simple_regret_log"; _save(fig, path); outputs += [path.with_suffix(".png"), path.with_suffix(".pdf")]
    elif key == "bovae_vs_egorse_curved_preimage_d10_d100":
        anytime = pd.read_csv(study / "plot_data/anytime_convergence_top5_bovae_vs_5seed_egorse.csv")
        problems = list(anytime.groupby(["D", "base_function"], sort=False))
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
        for ax, ((dim, problem), values) in zip(axes.flat, problems):
            values = values.assign(simple_regret=np.maximum(values["incumbent_original"] - values["known_minimum"], 1e-12))
            aggregate = values.groupby(["method", "display_label", "enrichment_evaluation"], as_index=False)["simple_regret"].agg(["mean", "std"]).reset_index()
            aggregate = aggregate.rename(columns={"mean": "mean_simple_regret", "std": "sample_std_simple_regret"})
            _ribbon(ax, aggregate, "enrichment_evaluation", "mean_simple_regret", "sample_std_simple_regret", "method")
            ax.set_yscale("log"); ax.set_title(f"{str(problem).title()}, D={dim}")
        axes[0, 0].legend(fontsize=7)
        path = figures / "anytime_convergence_top5_bovae_vs_5seed_egorse"; _save(fig, path); outputs += [path.with_suffix(".png"), path.with_suffix(".pdf")]
        for source, stem, value, ylabel in (
            ("final_results_top5_bovae_vs_5seed_egorse.csv", "final_results_top5_bovae_vs_5seed_egorse", "simple_regret", "Final simple regret"),
            ("runtime_top5_bovae_vs_5seed_egorse.csv", "runtime_top5_bovae_vs_5seed_egorse", "runtime_seconds", "Runtime (seconds)"),
        ):
            frame = pd.read_csv(study / "plot_data" / source)
            aggregate = frame.groupby(["D", "base_function", "method", "display_label"], as_index=False)[value].mean()
            fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
            for ax, ((dim, problem), rows) in zip(axes.flat, aggregate.groupby(["D", "base_function"], sort=False)):
                ax.bar(np.arange(len(rows)), rows[value], color=[COLORS.get(str(method), "#777777") for method in rows["method"]])
                ax.set_xticks(np.arange(len(rows)), rows["method"], rotation=35, ha="right", fontsize=7); ax.set_title(f"{str(problem).title()}, D={dim}"); ax.set_ylabel(ylabel)
                if value == "simple_regret": ax.set_yscale("log")
            path = figures / stem; _save(fig, path); outputs += [path.with_suffix(".png"), path.with_suffix(".pdf")]
        acceptance = pd.read_csv(study / "plot_data/retraining_proposal_acceptance_diagnostic.csv")
        acceptance = acceptance[acceptance["retraining_proposals"] > 0].copy()
        acceptance["acceptance_rate"] = acceptance["retraining_acceptances"] / acceptance["retraining_proposals"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method, rows in acceptance.groupby("method"):
            ax.scatter(rows["retraining_proposals"], rows["acceptance_rate"], label=str(method), alpha=0.65, color=COLORS.get(str(method)))
        ax.set_xlabel("Retraining proposals"); ax.set_ylabel("Acceptance rate"); ax.grid(True, alpha=0.25); ax.legend()
        path = figures / "retraining_proposal_acceptance_diagnostic"; _save(fig, path); outputs += [path.with_suffix(".png"), path.with_suffix(".pdf")]
    else:
        raise ValueError(f"no maintained plot recipe for {key}")
    return outputs


def cli(study: Path, *, expected_kernel: str | None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "aggregate", "plot", "verify"))
    parser.add_argument(
        "--config",
        type=Path,
        help="pipeline JSON to use (default: this study's configs/example.json)",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--check-artifacts",
        action="store_true",
        help="during verification, also require the separately published artifacts",
    )
    args = parser.parse_args()
    if args.action == "run":
        print(
            json.dumps(
                run_config(study, config_path=args.config, resume=not args.no_resume),
                indent=2,
                default=str,
            )
        )
    elif args.action == "aggregate":
        print(
            json.dumps(
                {"rows": len(aggregate_study(study, config_path=args.config))},
                indent=2,
            )
        )
    elif args.action == "plot":
        print(json.dumps({"figures": [str(path) for path in plot_study(study)]}, indent=2))
    else:
        print(
            json.dumps(
                verify_study(
                    study,
                    expected_kernel=expected_kernel,
                    config_path=args.config,
                    check_artifacts=args.check_artifacts,
                ),
                indent=2,
            )
        )
    return 0


__all__ = [
    "aggregate_study",
    "cli",
    "config_paths",
    "example_config_path",
    "pipeline_config",
    "plot_study",
    "result_summary",
    "run_config",
    "verify_study",
]
