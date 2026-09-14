# Example configuration

`example.json` is one D10/d6 Ackley cell using ordinary adaptive VAE
retraining, Matérn-5/2, LogEI, and the original SDR.

For another cell, update `problem`, its bounds, `dim`, `latent_dim`, `seed`,
`vae_checkpoint`, `initial_design_path`, and `output_dir`. Select `fixed_sdr`,
`retrain_sdr`, `retrain_dml_sdr`, or `fixed_no_sdr` with `mode`; the no-SDR
mode also requires `sdr_method: "none"`. DML requires a positive
`beta_metric_loss`. The manuscript provides the full benchmark, dimension,
method, seed, and hyperparameter matrix.

Shared correlated-normal VAE checkpoints and archived matched initial designs
are distributed separately for exact trace reproduction. The clean VAE-data
and initial-design generators are documented in
`BOVAE/src/bo_vae_sdr/data_generation/README.md`.

The example accepts each completed retraining step directly. To enable a
loss-based gate, set `retrain_acceptance_policy` to
`reconstruction_not_worse`, `kl_not_worse`, or `vae_loss_not_worse`; these
check reconstruction, KL, or reconstruction-plus-KL respectively.
