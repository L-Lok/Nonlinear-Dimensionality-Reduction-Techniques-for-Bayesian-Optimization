import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.curved_preimage import CurvedPreimageProblem
from bo_vae_sdr.data_generation import (
    derived_initial_design_seed,
    materialize_curved_preimage_artifact,
    materialize_initial_design,
    materialize_vae_data,
)
from bo_vae_sdr.vae import VAETrainingConfig


def test_materialize_vae_data_is_deterministic(tmp_path: Path) -> None:
    config = VAETrainingConfig(
        vae_id="data-smoke",
        ambient_dim=3,
        latent_dim=2,
        encoder_layer_dims=[3, 2],
        decoder_layer_dims=[2, 3],
        num_samples=20,
        validation_fraction=0.25,
        data_distribution="uniform_box",
        train_bounds_lower=-1.0,
        train_bounds_upper=1.0,
        seed=73,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    result = materialize_vae_data(config, first)
    materialize_vae_data(config, second)

    train_first = torch.load(
        first / "train_data.pt", map_location="cpu", weights_only=False
    )[0]
    train_second = torch.load(
        second / "train_data.pt", map_location="cpu", weights_only=False
    )[0]
    validation = torch.load(
        first / "validation_data.pt", map_location="cpu", weights_only=False
    )[0]
    assert result["train_size"] == 15
    assert result["validation_size"] == 5
    assert torch.equal(train_first, train_second)
    assert validation.shape == (5, 3)
    assert json.loads((first / "vae_config.json").read_text())["vae_id"] == "data-smoke"
    with pytest.raises(FileExistsError):
        materialize_vae_data(config, first)


def test_full_rank_namespace_seed_is_stable() -> None:
    seed, source = derived_initial_design_seed(
        problem="ackley",
        dim=10,
        seed=20260819,
        namespace="manuscript-full-rank-cn-reference|20260812",
    )
    assert seed == 8171132734082894764
    assert source.endswith("|ackley|D=10|seed=20260819")


def test_initial_design_writes_pipeline_and_egorse_formats(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "problem": "ackley",
                "dim": 3,
                "initial_points": 4,
                "seed": 19,
                "dtype": "float64",
                "problem_bound_lower": -2.0,
                "problem_bound_upper": 2.0,
                "initial_design_path": "unused.pt",
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "seed_19.pt"
    materialize_initial_design(
        config_path,
        output_path,
        workspace_root=tmp_path,
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    assert payload["x_obj"].shape == (4, 3)
    assert torch.equal(payload["y_internal"], -payload["y_original"])
    assert payload["objective_values_verified"] is True
    with np.load(output_path.with_suffix(".npz")) as arrays:
        assert np.array_equal(arrays["x_obj"], payload["x_obj"].numpy())
        assert np.array_equal(
            arrays["y_original"], payload["y_original"].numpy()
        )
        assert arrays["evaluation_elapsed_seconds"].shape == (4,)


def test_initial_design_can_select_checkpoint_training_data(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    train_x = torch.linspace(-1.0, 1.0, 30, dtype=torch.float64).reshape(10, 3)
    torch.save((train_x, torch.ones(10, 1)), checkpoint / "train_data.pt")
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "problem": "ackley",
                "dim": 3,
                "initial_points": 4,
                "seed": 23,
                "dtype": "float64",
                "problem_bound_lower": -2.0,
                "problem_bound_upper": 2.0,
                "vae_input_lower": -1.0,
                "vae_input_upper": 1.0,
                "vae_checkpoint": str(checkpoint),
                "initial_design_path": "unused.pt",
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "from_training_data.pt"
    materialize_initial_design(
        config_path,
        output_path,
        workspace_root=tmp_path,
        source="vae-training-data",
    )

    generator = torch.Generator(device="cpu").manual_seed(23)
    selected = train_x[torch.randperm(10, generator=generator)[:4]]
    expected = 2.0 * selected
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    torch.testing.assert_close(payload["x_obj"], expected, rtol=0.0, atol=1e-15)


def test_curved_artifact_and_initial_design_round_trip(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "curved_ackley_d10_de4"
    result = materialize_curved_preimage_artifact(
        base_function="ackley",
        dim=10,
        alpha=1.0,
        output_dir=artifact_dir,
    )
    reference = CurvedPreimageProblem.from_artifact(
        artifact_dir / "problem.npz", artifact_dir / "metadata.json"
    )
    assert result["problem_id"] == reference.problem_id == "curved_ackley_d10_de4"

    config_path = tmp_path / "curved_config.json"
    config_path.write_text(
        json.dumps(
            {
                "problem": reference.problem_id,
                "dim": 10,
                "initial_points": 10,
                "seed": 1207310,
                "dtype": "float64",
                "curved_preimage_artifact_path": str(artifact_dir / "problem.npz"),
                "curved_preimage_metadata_path": str(artifact_dir / "metadata.json"),
                "initial_design_path": "unused.pt",
            }
        ),
        encoding="utf-8",
    )
    design_path = tmp_path / "seed_1207310.pt"
    materialize_initial_design(
        config_path,
        design_path,
        workspace_root=tmp_path,
    )
    design = torch.load(design_path, map_location="cpu", weights_only=False)
    expected = reference.evaluate_batch(design["x_obj"].numpy()).reshape(-1, 1)
    assert np.allclose(design["y_original"].numpy(), expected, rtol=0.0, atol=1e-10)
