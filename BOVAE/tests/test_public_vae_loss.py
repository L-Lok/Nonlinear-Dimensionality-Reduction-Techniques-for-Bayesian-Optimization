import pytest
import torch
from types import SimpleNamespace

from bo_vae_sdr.pipeline.core import RETRAIN_ACCEPTANCE_METRICS
from bo_vae_sdr.pipeline.engine import PipelineRunner
from bo_vae_sdr.vae.artifacts import VAETrainingConfig, _vae_loss_terms, make_model
from bo_vae_sdr.vae.dml import manuscript_triplet_config, normalize_dml_targets
from bo_vae_sdr.vae.metrics import TripletLossTorch


def _model_and_data():
    config = VAETrainingConfig(
        vae_id="test",
        ambient_dim=3,
        latent_dim=2,
        encoder_layer_dims=[3, 2],
        decoder_layer_dims=[2, 3],
    )
    model = make_model(config).to(dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, 36, dtype=torch.float64).reshape(12, 3)
    return config, model, x


def test_public_vae_loss_is_reconstruction_plus_kl() -> None:
    _, model, x = _model_and_data()
    torch.manual_seed(9)
    loss, terms = _vae_loss_terms(model, x, beta=0.75)
    assert "dml_triplet_loss" not in terms
    assert float(loss.detach()) == pytest.approx(
        terms["reconstruction_loss"] + 0.75 * terms["kl_loss"]
    )


def test_retraining_acceptance_uses_only_the_selected_vae_diagnostic() -> None:
    assert RETRAIN_ACCEPTANCE_METRICS == {
        "always_accept": (),
        "reconstruction_not_worse": ("reconstruction_loss",),
        "kl_not_worse": ("kl_loss",),
        "vae_loss_not_worse": ("vae_loss",),
    }
    runner = object.__new__(PipelineRunner)
    before = {
        "observed": {
            "reconstruction_loss": 1.0,
            "kl_loss": 1.0,
            "vae_loss": 2.0,
        }
    }
    for policy, selected in RETRAIN_ACCEPTANCE_METRICS.items():
        runner.config = SimpleNamespace(
            retrain_acceptance_policy=policy,
            retrain_acceptance_loss_rel_tol=0.0,
            retrain_acceptance_loss_abs_tol=0.0,
            retrain_acceptance_reconstruction_rel_tol=0.0,
            retrain_acceptance_reconstruction_abs_tol=0.0,
            retrain_acceptance_kl_rel_tol=0.0,
            retrain_acceptance_kl_abs_tol=0.0,
        )
        after_values = {
            "reconstruction_loss": 10.0,
            "kl_loss": 10.0,
            "vae_loss": 10.0,
        }
        for metric in selected:
            after_values[metric] = before["observed"][metric]
        accepted, failures = runner.assess_retrain_acceptance(
            before, {"observed": after_values}
        )
        assert accepted
        assert failures == []
        if selected:
            rejected_values = dict(before["observed"])
            rejected_values[selected[0]] *= 2.0
            accepted, failures = runner.assess_retrain_acceptance(
                before, {"observed": rejected_values}
            )
            assert not accepted
            assert [failure["metric"] for failure in failures] == [selected[0]]


def test_dml_adds_triplet_loss_on_normalized_targets() -> None:
    _, model, x = _model_and_data()
    y, normalizer = normalize_dml_targets(
        torch.linspace(2.0, 8.0, x.shape[0], dtype=torch.float64).reshape(-1, 1)
    )
    assert normalizer.minimum == 2.0
    assert normalizer.maximum == 8.0
    torch.manual_seed(9)
    loss, terms = _vae_loss_terms(
        model,
        x,
        beta=1.0,
        y_metric=y,
        dml_triplet_config=manuscript_triplet_config(0.25),
        dml_loss_weight=0.5,
    )
    assert terms["dml_triplet_loss"] > 0.0
    assert float(loss.detach()) == pytest.approx(
        terms["reconstruction_loss"]
        + terms["kl_loss"]
        + 0.5 * terms["dml_triplet_loss"]
    )


def test_triplet_visualization_uses_training_loss_values() -> None:
    metric = TripletLossTorch(threshold=0.25, eta=0.05)
    positive_embedding = torch.tensor(0.6, dtype=torch.float64)
    negative_embedding = torch.tensor([0.2, 1.0], dtype=torch.float64)
    positive_target = torch.tensor(0.1, dtype=torch.float64)
    negative_target = torch.tensor([0.2, 0.8], dtype=torch.float64)

    values = metric.triplet_loss_values(
        positive_embedding_distances=positive_embedding,
        negative_embedding_distances=negative_embedding,
        positive_target_distances=positive_target,
        negative_target_distances=negative_target,
    )
    expected_valid = (
        torch.logaddexp(
            torch.tensor(0.0, dtype=torch.float64),
            positive_embedding - negative_embedding[1],
        )
        * metric.smooth_indicator(metric.threshold - positive_target)
        / metric.smooth_indicator(metric.threshold)
        * metric.smooth_indicator(negative_target[1] - metric.threshold)
        / metric.smooth_indicator(1.0 - metric.threshold)
    )

    assert values[0] == 0.0
    assert values[1] == pytest.approx(float(expected_valid))
