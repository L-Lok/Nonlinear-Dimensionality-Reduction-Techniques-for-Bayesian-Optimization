# Curved-preimage BO-VAE versus EGORSE

The BO-VAE study uses a Matérn-5/2 Gaussian-process kernel, LogEI, and the original
SDR. EGORSE uses the squared-exponential SMT KRG backend when available; its
documented sklearn fallback uses Matérn-5/2.

The publication bundle supplies one representative BO-VAE configuration in
`configs/example.json`. The existing `configs/study.json` and
`configs/egorse.json` continue to define the EGORSE execution matrix. See
`configs/README.md` for the BO-VAE fields to change when constructing other
cells from the manuscript protocol.

```bash
export PYTHONPATH=BOVAE/src MPLCONFIGDIR=/tmp/bovae-matplotlib
python -m bo_vae_sdr.data_generation.generate_curved_preimage_artifacts
python -m bo_vae_sdr.data_generation.generate_initial_designs \
  BOVAE/studies/bovae_vs_egorse_curved_preimage_d10_d100/configs/example.json
python BOVAE/studies/bovae_vs_egorse_curved_preimage_d10_d100/scripts/run_study.py verify
python BOVAE/studies/bovae_vs_egorse_curved_preimage_d10_d100/scripts/run_study.py run
python BOVAE/studies/bovae_vs_egorse_curved_preimage_d10_d100/scripts/run_egorse.py
```

Pass `--config path/to/config.json` to the BO-VAE runner to use a modified
copy. Add `--check-artifacts` to `verify` after generating the problem and
initial-design inputs and installing or training the required VAE checkpoint.
Archived inputs remain necessary when reproducing the published trajectories
exactly.
