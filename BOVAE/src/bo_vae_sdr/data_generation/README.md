# Data generation

These entry points create publication-compatible inputs from the maintained
data distributions and benchmark definitions. Run them from the directory
containing `BOVAE` with `PYTHONPATH=BOVAE/src`.

## VAE pretraining data

`generate_vae_data.py` writes `train_data.pt`, `validation_data.pt`,
`vae_config.json`, and `data_metadata.json`. The supplied configuration is the
representative manuscript D10/d6 correlated-normal VAE.

```bash
PYTHONPATH=BOVAE/src python -m bo_vae_sdr.data_generation.generate_vae_data \
  BOVAE/src/bo_vae_sdr/pretrained_vae/vae_config.example.json \
  BOVAE/src/bo_vae_sdr/pretrained_vae/correlated_normal_reference/D10_d6
```

To generate the same data and train the model in one command, use the existing
pretraining CLI instead:

```bash
PYTHONPATH=BOVAE/src python -m bo_vae_sdr.vae.cli \
  BOVAE/src/bo_vae_sdr/pretrained_vae/vae_config.example.json \
  BOVAE/src/bo_vae_sdr/pretrained_vae/correlated_normal_reference/D10_d6 \
  --device cuda
```

## Initial designs

`generate_initial_designs.py` reads a pipeline configuration and writes the
same design in three forms: a `.pt` file for the BO/BO-VAE pipelines, an `.npz`
file for EGORSE, and a small JSON description. The default source is uniform
sampling over the configured objective box.

```bash
PYTHONPATH=BOVAE/src python -m bo_vae_sdr.data_generation.generate_initial_designs \
  BOVAE/studies/manuscript_figure1_ackley_bo_sdr_ablation_d10/configs/example.json
```

Repeat `--seed` to build a matched set for several runs. All methods that are
being compared must point to the same generated file. For cells whose starting
points should be selected from VAE pretraining data, use
`--source vae-training-data` after installing or training the checkpoint.

The optional namespace below follows the deterministic convention used when
new full-rank manuscript designs were generated:

```bash
--seed-namespace 'manuscript-full-rank-cn-reference|20260812'
```

Published optimization traces must continue to use their corresponding
archived initial designs. Generating a new design, even with the same BO seed,
defines a new run and will generally produce a different trajectory.

## Curved-preimage problems

The following command materializes all eight problem geometries declared by
the retained study. `--base` and `--dim` may be repeated to generate a subset.

```bash
PYTHONPATH=BOVAE/src python -m \
  bo_vae_sdr.data_generation.generate_curved_preimage_artifacts
```

After generating the geometry, generate each shared BO-VAE/EGORSE design from
the corresponding pipeline config:

```bash
PYTHONPATH=BOVAE/src python -m bo_vae_sdr.data_generation.generate_initial_designs \
  BOVAE/studies/bovae_vs_egorse_curved_preimage_d10_d100/configs/example.json
```

The scripts refuse to replace existing files unless `--overwrite` is given.
