# Manuscript full-rank adaptive retraining

`configs/example.json` demonstrates one D10/d6 Ackley run with ordinary
adaptive VAE retraining, Matérn-5/2, LogEI, and the original SDR. The manuscript
defines the complete fixed, retrained, normalized-DML, and no-SDR comparison
across D10/d6 and D100/d50.

```bash
export PYTHONPATH=BOVAE/src MPLCONFIGDIR=/tmp/bovae-matplotlib
python BOVAE/studies/manuscript_full_rank_cn_reference_adaptive_retraining_d10_d100/scripts/run_study.py verify
python BOVAE/studies/manuscript_full_rank_cn_reference_adaptive_retraining_d10_d100/scripts/run_study.py run
```

Pass `--config path/to/config.json` to run a modified copy. See
`configs/README.md` for method and path changes. New VAE data and matched
initial designs can be produced with the entry points documented in
`BOVAE/src/bo_vae_sdr/data_generation/README.md`. The archived checkpoints,
inputs, results, and figures remain necessary for reproducing the published
trajectories exactly.
