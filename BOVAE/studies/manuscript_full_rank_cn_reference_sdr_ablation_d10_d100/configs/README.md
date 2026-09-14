# Example configuration

`example.json` is the with-original-SDR arm for one D10/d6 Ackley fixed-VAE
cell. It uses Matérn-5/2 and LogEI. Because both ablation arms keep the VAE
fixed, the example intentionally contains no retraining, retraining-acceptance,
or DML hyperparameters.

For the no-SDR arm, change `mode` to `fixed_no_sdr`, set `sdr_method` to
`none`, and choose a distinct `output_dir`. Update `problem`, its bounds,
`dim`, `latent_dim`, `seed`, `vae_checkpoint`, and `initial_design_path` for
other manuscript cells. The manuscript records the full D10/d6 and D100/d50
matrix for Ackley and Rosenbrock.

Shared correlated-normal VAE checkpoints and archived matched initial designs
are distributed separately for exact trace reproduction. New matched designs
can be generated with `bo_vae_sdr.data_generation.generate_initial_designs`.
