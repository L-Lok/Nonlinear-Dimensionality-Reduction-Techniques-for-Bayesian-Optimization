# Manuscript Figure 1: ambient BO/SDR ablation

This study compares ambient BO with and without the original SDR on
10-dimensional Ackley. Both arms use a Matérn-5/2 Gaussian process and LogEI;
neither uses a VAE. `configs/example.json` demonstrates the SDR arm, and
`configs/README.md` explains the two-field change for plain BO.

```bash
export PYTHONPATH=BOVAE/src MPLCONFIGDIR=/tmp/bovae-matplotlib
python -m bo_vae_sdr.data_generation.generate_initial_designs \
  BOVAE/studies/manuscript_figure1_ackley_bo_sdr_ablation_d10/configs/example.json
python BOVAE/studies/manuscript_figure1_ackley_bo_sdr_ablation_d10/scripts/run_study.py verify
python BOVAE/studies/manuscript_figure1_ackley_bo_sdr_ablation_d10/scripts/run_study.py run
```

Pass `--config path/to/config.json` to run a modified example. Point both arms
to the same generated design for a matched comparison. The archived designs
and figure data remain necessary for reproducing the published traces exactly.
