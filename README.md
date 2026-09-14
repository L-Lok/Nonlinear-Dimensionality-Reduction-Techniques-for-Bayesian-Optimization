# Nonlinear Dimensionality Reduction Techniques for Bayesian Optimisation

This repository deloys the algorithms from the paper Nonlinear Dimensionality Reduction Techniques for Bayesian Optimisation.

Run all commands in this README from the repository root, the directory that
contains `BOVAE`, `requirements.txt`, and `Makefile`.

## Workspace structure

```text
.
├── BOVAE/
│   ├── src/
│   │   ├── bo_vae_sdr/
│   │   │   ├── pipeline/          # BO and BO-VAE execution pipeline
│   │   │   ├── vae/               # VAE training, loading, and retraining
│   │   │   ├── sdr/               # original SDR implementation
│   │   │   ├── data_generation/   # data and initial-design generators
│   │   │   ├── pretrained_vae/    # example VAE config/checkpoint location
│   │   │   └── study_tools.py     # shared run, verify, aggregate, and plot CLI
│   │   ├── benchmarks/            # manuscript benchmark functions
│   │   ├── egorse/                # EGORSE comparison implementation
│   │   └── common/                # shared EGORSE utilities
│   ├── studies/
│   │   └── <study>/
│   │       ├── configs/            # example pipeline and study controls
│   │       └── scripts/            # study entry points
│   └── tests/                      # source-level regression tests
├── requirements.txt
├── requirements-lock.txt
├── requirements-egorse-optional.txt
├── requirements-egorse-optional-lock.txt
└── Makefile
```

The six study directories are:

- `manuscript_figure1_ackley_bo_sdr_ablation_d10`
- `manuscript_full_rank_cn_reference_adaptive_retraining_d10_d100`
- `manuscript_full_rank_cn_reference_profiles_d10_d100`
- `manuscript_full_rank_cn_reference_sdr_ablation_d10_d100`
- `latent_dimension_unweighted_figure_revision`
- `bovae_vs_egorse_curved_preimage_d10_d100`

Generated or separately supplied files are placed under the relevant study in
`inputs/`, `checkpoints/`, `results/`, `plot_data/`, `figures/`, and
`reports/`. The example JSON files contain the expected paths.

## Install the environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `requirements-lock.txt` instead of `requirements.txt` to install the
pinned core versions:

```bash
python -m pip install -r requirements-lock.txt
```

Install the optional EGORSE numerical backends when running that comparison:

```bash
python -m pip install -r requirements-egorse-optional.txt
```

The corresponding pinned optional versions are in
`requirements-egorse-optional-lock.txt`.

Configure the source and Matplotlib paths for the current shell:

```bash
export PYTHONPATH=BOVAE/src
export MPLCONFIGDIR=/tmp/bovae-matplotlib
```

## Prepare VAE checkpoints

Train the supplied D10/d6 example VAE into the path used by the shared
full-rank examples:

```bash
python -m bo_vae_sdr.vae.cli \
  BOVAE/src/bo_vae_sdr/pretrained_vae/vae_config.example.json \
  BOVAE/src/bo_vae_sdr/pretrained_vae/correlated_normal_reference/D10_d6 \
  --device cuda
```

For another VAE, copy and edit `vae_config.example.json`, choose an output
directory, and update the study pipeline config's `vae_checkpoint` and
`vae_state_file` fields. Use `--device cpu` when CUDA is unavailable. Add
`--overwrite` only when an existing checkpoint directory should be replaced.

To generate VAE training and validation tensors without training a model:

```bash
python -m bo_vae_sdr.data_generation.generate_vae_data \
  BOVAE/src/bo_vae_sdr/pretrained_vae/vae_config.example.json \
  BOVAE/src/bo_vae_sdr/pretrained_vae/correlated_normal_reference/D10_d6
```

## Prepare study inputs

Generate the initial design named by a study configuration:

```bash
python -m bo_vae_sdr.data_generation.generate_initial_designs \
  BOVAE/studies/<study>/configs/example.json
```

For a design selected from a VAE checkpoint's training data, add:

```bash
--source vae-training-data
```

For the curved-preimage study, generate its problem artifacts before its
initial designs:

```bash
python -m bo_vae_sdr.data_generation.generate_curved_preimage_artifacts

python -m bo_vae_sdr.data_generation.generate_initial_designs \
  BOVAE/studies/bovae_vs_egorse_curved_preimage_d10_d100/configs/example.json
```

If archived inputs or checkpoints are available, place them at the paths in
the selected configuration instead of regenerating them.

## Run a study

Every study exposes the same four actions through `scripts/run_study.py`:

```bash
python BOVAE/studies/<study>/scripts/run_study.py verify
python BOVAE/studies/<study>/scripts/run_study.py run
python BOVAE/studies/<study>/scripts/run_study.py aggregate
python BOVAE/studies/<study>/scripts/run_study.py plot
```

`run` resumes from `results/.../checkpoint.pt` when one exists. Disable resume
with `--no-resume`. To use a modified configuration, pass it explicitly:

```bash
python BOVAE/studies/<study>/scripts/run_study.py run \
  --config path/to/config.json
```

`verify` checks the source layout and configuration. After installing all
inputs and results, request the complete artifact check with:

```bash
python BOVAE/studies/<study>/scripts/run_study.py verify --check-artifacts
```

Aggregation writes `run_summary.csv` and `run_summary.json` under `reports/`.
Plotting reads the study CSV files under `plot_data/` and writes figures under
`figures/`.

## Run the pipeline directly

```python
from bo_vae_sdr.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig.from_json(
    "BOVAE/studies/<study>/configs/example.json"
)
summary = run_pipeline(config)
print(summary)
```

## Run EGORSE

After installing the optional dependencies and preparing curved-preimage
artifacts and `.npz` initial designs, run all configured EGORSE cells with:

```bash
python \
  BOVAE/studies/bovae_vs_egorse_curved_preimage_d10_d100/scripts/run_egorse.py
```

Limit or select cells when needed:

```bash
python \
  BOVAE/studies/bovae_vs_egorse_curved_preimage_d10_d100/scripts/run_egorse.py \
  --selector curved_ackley_d10_de4 --max-runs 1
```

## Check the workspace

```bash
make test
make import-check
make config-check
```
