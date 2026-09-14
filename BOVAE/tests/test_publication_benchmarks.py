from pathlib import Path

import benchmarks
from bo_vae_sdr.pipeline.surrogate import make_problem


EXPECTED_MODULES = {
    "__init__.py",
    "ackley.py",
    "base.py",
    "canonical_levy.py",
    "canonical_styblinski_tang.py",
    "curved_preimage.py",
    "rastrigin.py",
    "rosenbrock.py",
}


def test_publication_contains_only_manuscript_benchmark_modules() -> None:
    package = Path(benchmarks.__file__).resolve().parent
    assert {path.name for path in package.glob("*.py")} == EXPECTED_MODULES
    assert set(benchmarks.__all__) == {
        "Ackley",
        "BaseTestFunction",
        "CanonicalLevy",
        "CanonicalStyblinskiTang",
        "CurvedPreimage",
        "CurvedPreimageProblem",
        "Rastrigin",
        "Rosenbrock",
        "build_curved_preimage_artifact",
        "sample_uniform_design",
    }


def test_all_full_rank_manuscript_problem_keys_construct() -> None:
    for key in (
        "ackley",
        "rosenbrock",
        "rastrigin",
        "canonical_levy",
        "canonical_styblinski_tang",
    ):
        problem = make_problem(key, dim=10)
        assert problem.dim == 10
