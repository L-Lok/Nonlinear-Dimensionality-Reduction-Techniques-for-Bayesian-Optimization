# Manuscript full-rank SDR ablation

`configs/example.json` demonstrates the with-original-SDR arm for one D10/d6
Ackley fixed-VAE cell. Both ablation arms use Matérn-5/2 and LogEI. The
manuscript defines the complete D10/d6 and D100/d50 comparison for Ackley and
Rosenbrock.

```bash
export PYTHONPATH=BOVAE/src MPLCONFIGDIR=/tmp/bovae-matplotlib
python BOVAE/studies/manuscript_full_rank_cn_reference_sdr_ablation_d10_d100/scripts/run_study.py verify
python BOVAE/studies/manuscript_full_rank_cn_reference_sdr_ablation_d10_d100/scripts/run_study.py run
```

Pass `--config path/to/config.json` to run a modified copy. See
`configs/README.md` for the no-SDR changes. Shared checkpoints, matched initial
designs, results, and figures are distributed separately for exact manuscript
trace reproduction. New matched designs can be generated with
`bo_vae_sdr.data_generation.generate_initial_designs` and shared by both
ablation arms.
