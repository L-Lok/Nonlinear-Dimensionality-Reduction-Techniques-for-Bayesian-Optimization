# Shared pretrained VAEs

`correlated_normal_reference/D10_d6` and `D100_d50` name the common reference
checkpoints used by the adaptive-retraining, profile, and SDR-ablation studies.
Their model states, VAE configurations, training traces, and latent-anchor data
are part of the separately distributed artifact archive, not this source-only
publication bundle.

The public VAE uses SiLU hidden activations. Decoder output is `linear` for the
shared reference models and may be `scaled_tanh` for the bounded
latent-dimension and curved-preimage models. Load installed artifacts with
`bo_vae_sdr.vae.load_pretrained_vae`, or recreate one with
`python -m bo_vae_sdr.vae.cli` and its VAE configuration. A representative
D10/d6 configuration and a data-only generator are documented in
`BOVAE/src/bo_vae_sdr/data_generation/README.md`.
