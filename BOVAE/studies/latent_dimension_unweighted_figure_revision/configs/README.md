# Example configuration

`example.json` is one D10/d6 Ackley cell using normalized-DML VAE retraining,
Matérn-5/2, LogEI, and the original SDR. Retraining follows the fixed
manuscript period `q = 100`; proposed retraining steps are accepted directly,
without a reconstruction-based acceptance policy.

To create another cell, update `problem`, `dim`, `latent_dim`, `seed`,
`vae_checkpoint`, `initial_design_path`, and `output_dir`. Use `fixed_sdr`,
`retrain_sdr`, or `retrain_dml_sdr` for `mode`, and set the DML weight and
threshold only for the DML method. Keep `retrain_schedule_kind` set to `fixed`
and `retrain_period` set to `100` for retrained cells. The manuscript is the
authoritative source for the latent-dimension grid, seeds, and reported
hyperparameters.

The checkpoint families and archived matched initial designs are large
artifacts and are distributed separately from this source repository. New
designs can be selected from a trained checkpoint's `train_data.pt` with
`python -m bo_vae_sdr.data_generation.generate_initial_designs CONFIG --source
vae-training-data`.
