# Manuscript full-rank profiles

`configs/example.json` demonstrates one D10/d6 Ackley BO-VAE profile cell with
ordinary adaptive retraining, Matérn-5/2, LogEI, and the original SDR. The
manuscript defines the complete ambient and BO-VAE profile matrix over five
benchmarks and two dimension settings.

```bash
export PYTHONPATH=BOVAE/src MPLCONFIGDIR=/tmp/bovae-matplotlib
python BOVAE/studies/manuscript_full_rank_cn_reference_profiles_d10_d100/scripts/run_study.py verify
python BOVAE/studies/manuscript_full_rank_cn_reference_profiles_d10_d100/scripts/run_study.py run
```

Pass `--config path/to/config.json` to run a modified copy. Shared checkpoints,
archived initial designs, profile data, and figures are distributed separately;
aggregation and plotting require those artifacts. New matched initial designs
can be produced with `bo_vae_sdr.data_generation.generate_initial_designs`.
