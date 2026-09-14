from pathlib import Path

import torch
import bo_vae_sdr.sdr.original as original_adapter
import bo_vae_sdr.sdr.original_transformer as original_transformer

from bo_vae_sdr.sdr import OriginalSDR, SDR_METHODS


def test_only_original_sdr_is_public() -> None:
    assert SDR_METHODS == {"none", "original"}


def test_original_sdr_source_contains_only_public_contraction_state() -> None:
    source = "\n".join(
        Path(module.__file__).read_text(encoding="utf-8")
        for module in (original_adapter, original_transformer)
    )
    reserved_terms = (
        "local_" + "refit",
        "probe_" + "count",
        "probe_" + "evaluation",
        "contraction_" + "scale",
        "reset_" + "window",
    )
    assert all(term not in source for term in reserved_terms)


def _exercise(adapter):
    generator = torch.Generator(device="cpu").manual_seed(20260907)
    points = torch.rand((64, 3), generator=generator, dtype=torch.float64)
    targets = -torch.sum((points - torch.tensor([0.2, 0.55, 0.8])) ** 2, dim=1, keepdim=True)
    updates = []
    for iteration in range(41):
        adapter.record_candidate_boundary(points[iteration % len(points)])
        bounds, trace = adapter.update(
            train_unit_x=points[: max(12, iteration + 1)],
            train_internal_y=targets[: max(12, iteration + 1)],
            best_original=float(-targets[: max(12, iteration + 1)].max()),
            iteration=iteration,
        )
        if trace.get("updated"):
            updates.append((iteration, bounds.clone()))
    return updates


def test_original_sdr_matches_prefactor_golden() -> None:
    updates = _exercise(OriginalSDR(3, 40, period=5))
    expected = torch.tensor(
        [
            [0.0, 0.2796233671214136],
            [0.18745429057038196, 0.5220173011566482],
            [0.5060996874461885, 0.8377098251164397],
        ],
        dtype=torch.float64,
    )
    assert [iteration for iteration, _ in updates] == list(range(0, 41, 5))
    assert torch.allclose(updates[-1][1], expected, atol=1e-15, rtol=0.0)
