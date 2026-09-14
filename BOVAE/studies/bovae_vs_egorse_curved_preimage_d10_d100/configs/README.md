# Example configuration

`example.json` is one runnable D10 curved-Ackley BO-VAE example using ordinary
VAE retraining with the original SDR.

For another manuscript cell, update `problem`, `dim`, `latent_dim`, `seed`,
the three input/checkpoint paths, and `output_dir`. Keep `gp_kernel` set to
`matern52` for BO-VAE. Select `fixed_sdr`, `retrain_sdr`, or `retrain_dml_sdr` with
`mode`; DML also requires a positive `beta_metric_loss`. The manuscript gives
the complete problem, dimension, method, and seed matrix.

The retained EGORSE execution settings are in `study.json` and `egorse.json`;
`scripts/run_egorse.py` uses those two files to construct its comparison
cells. Its primary SMT backend uses the squared-exponential kernel,
and the documented sklearn fallback uses Matérn-5/2. Large checkpoint,
archived initial-design, and result artifacts are distributed separately from
this source repository. New curved-problem and matched initial-design inputs
can be generated with the entry points documented in
`BOVAE/src/bo_vae_sdr/data_generation/README.md`.
