# Latent-dimension figure revision

The publication bundle supplies one representative D10/d6 normalized-DML
retraining configuration with fixed retraining period `q = 100` and direct
acceptance of each retraining step. The complete latent-dimension grid is
specified in the manuscript rather than duplicated as thousands of generated
JSON files. All BO surrogates in this study use Matérn-5/2 and LogEI.

```bash
export PYTHONPATH=BOVAE/src MPLCONFIGDIR=/tmp/bovae-matplotlib
python BOVAE/studies/latent_dimension_unweighted_figure_revision/scripts/run_study.py verify
python BOVAE/studies/latent_dimension_unweighted_figure_revision/scripts/run_study.py run
```

Pass `--config path/to/config.json` to run a modified copy of the example. See
`configs/README.md` for the fields that define a cell. After installing or
training its VAE, a new matched design can be generated with
`python -m bo_vae_sdr.data_generation.generate_initial_designs CONFIG --source
vae-training-data`.
Exact published trajectories and plots require their archived input and result
artifacts.
