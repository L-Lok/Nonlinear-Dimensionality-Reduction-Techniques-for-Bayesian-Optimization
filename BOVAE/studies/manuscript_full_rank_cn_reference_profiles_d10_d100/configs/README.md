# Example configuration

`example.json` is one D10/d6 Ackley BO-VAE cell using ordinary adaptive
retraining, Matérn-5/2, LogEI, and the original SDR.

Update `problem`, its bounds, `dim`, `latent_dim`, `seed`, `vae_checkpoint`,
`initial_design_path`, and `output_dir` for another profile cell. BO-VAE modes
are `fixed_sdr`, `retrain_sdr`, and `retrain_dml_sdr`. Ambient profile cells
use `ambient_bo_sdr`, an empty `vae_checkpoint`, and `latent_dim` equal to
`dim`. The manuscript defines the complete five-problem, two-dimension method
matrix and the performance/data-profile construction.

Shared checkpoints, archived matched initial designs, and profile result data
are distributed separately for exact trace reproduction. New matched designs
can be generated with `bo_vae_sdr.data_generation.generate_initial_designs`.

The example accepts each completed retraining step directly. Optional public
acceptance gates are limited to reconstruction, KL, or total VAE loss through
`reconstruction_not_worse`, `kl_not_worse`, and `vae_loss_not_worse`.
