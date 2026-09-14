import json
from pathlib import Path

from bo_vae_sdr.study_tools import config_paths, pipeline_config


BOVAE_ROOT = Path(__file__).resolve().parents[1]
STUDIES = BOVAE_ROOT / "studies"
EXPECTED_KERNELS = {
    "bovae_vs_egorse_curved_preimage_d10_d100": "matern52",
    "latent_dimension_unweighted_figure_revision": "matern52",
    "manuscript_figure1_ackley_bo_sdr_ablation_d10": "matern52",
    "manuscript_full_rank_cn_reference_adaptive_retraining_d10_d100": "matern52",
    "manuscript_full_rank_cn_reference_profiles_d10_d100": "matern52",
    "manuscript_full_rank_cn_reference_sdr_ablation_d10_d100": "matern52",
}


def test_each_study_has_one_loadable_bovae_example() -> None:
    study_names = {path.name for path in STUDIES.iterdir() if path.is_dir()}
    assert study_names == set(EXPECTED_KERNELS)
    for study_name, kernel in EXPECTED_KERNELS.items():
        study = STUDIES / study_name
        example = study / "configs/example.json"
        assert config_paths(study) == [example]
        config = pipeline_config(example)
        assert config.gp_kernel == kernel
        assert config.sdr_method in {"none", "original"}


def test_egorse_control_configs_remain_separate() -> None:
    config_dir = (
        STUDIES / "bovae_vs_egorse_curved_preimage_d10_d100" / "configs"
    )
    json_names = {path.name for path in config_dir.glob("*.json")}
    assert json_names == {"example.json", "egorse.json", "study.json"}
    egorse = json.loads((config_dir / "egorse.json").read_text(encoding="utf-8"))
    assert egorse["primary_kernel"] == "squared_exponential"
    assert egorse["fallback_kernel"] == "matern52"


def test_latent_dimension_example_uses_fixed_period_without_acceptance_gate() -> None:
    example = json.loads(
        (
            STUDIES
            / "latent_dimension_unweighted_figure_revision"
            / "configs/example.json"
        ).read_text(encoding="utf-8")
    )
    assert example["retrain_schedule_kind"] == "fixed"
    assert example["retrain_period"] == 100
    assert not any(key.startswith("retrain_acceptance_") for key in example)


def test_study_configs_do_not_publish_top_five_selection_metadata() -> None:
    forbidden_keys = {"plot_caption", "top_five_rule", "selection_rule"}
    for path in STUDIES.glob("*/configs/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert forbidden_keys.isdisjoint(payload)
        serialized = json.dumps(payload).lower()
        assert "top five" not in serialized
        assert "top-five" not in serialized


def test_fixed_sdr_ablation_example_has_no_retraining_or_dml_fields() -> None:
    example = json.loads(
        (
            STUDIES
            / "manuscript_full_rank_cn_reference_sdr_ablation_d10_d100"
            / "configs/example.json"
        ).read_text(encoding="utf-8")
    )
    assert example["mode"] == "fixed_sdr"
    assert not any(key.startswith("retrain_") for key in example)
    assert not any(key.startswith("dml_") for key in example)
    assert "beta_metric_loss" not in example
