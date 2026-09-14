# Example configuration

`example.json` is the BO-with-original-SDR arm of the D10 Ackley ablation. It
uses Matérn-5/2 and LogEI and does not use a VAE.

For the no-SDR arm, change `mode` to `ambient_bo`, set `sdr_method` to `none`,
and choose a distinct `output_dir`. Change `seed` and `initial_design_path`
together to preserve matched initialization. The manuscript records the full
seed set and reporting protocol.

The archived initial-design tensors are distributed separately for exact trace
reproduction. `bo_vae_sdr.data_generation.generate_initial_designs` creates
new matched designs for independent reproduction runs.
