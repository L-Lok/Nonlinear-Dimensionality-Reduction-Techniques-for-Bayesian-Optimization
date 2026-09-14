# Publication bundle manifest

The bundle is intentionally limited to publication source and textual
configuration data.

Included file classes:

- Python source under the maintained `BOVAE/src` packages.
- Python study entry points for all six retained studies.
- One public BO-VAE `configs/example.json` per retained study.
- The curved-preimage study's retained EGORSE `study.json` and `egorse.json`
  control files.
- Publication input generators under `BOVAE/src/bo_vae_sdr/data_generation`.
- Data-independent regression tests for configurations, benchmarks, data
  generation, VAE losses, and original SDR.
- Core and optional-EGORSE dependency files, pinned version files, build
  helpers, and documentation at the repository root.

Excluded file classes:

- `.pt`, `.pth`, `.npz`, and other model/data tensors.
- `inputs`, `checkpoints`, `results`, `plot_data`, `figures`, and `reports`.
- Internal development material, integrity manifests, execution-state files,
  and caches.
- Exploratory scripts and excluded studies.

This separation keeps the GitHub repository reviewable while allowing the
large immutable artifact archive to be released independently.

The manuscript, rather than generated per-seed JSON files, is the authoritative
record of the complete experimental matrices and hyperparameter tables.
