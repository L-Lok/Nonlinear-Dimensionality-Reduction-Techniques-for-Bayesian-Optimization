.PHONY: test import-check config-check

PYTHON ?= python3
ENV = PYTHONPATH=BOVAE/src MPLCONFIGDIR=/tmp/bovae-matplotlib PYTHONDONTWRITEBYTECODE=1

test:
	$(ENV) $(PYTHON) -m pytest -q -p no:cacheprovider BOVAE/tests

import-check:
	$(ENV) $(PYTHON) -c "import benchmarks, bo_vae_sdr, bo_vae_sdr.pipeline, bo_vae_sdr.vae, egorse"

config-check:
	$(ENV) $(PYTHON) -c "import json, pathlib; from bo_vae_sdr.study_tools import pipeline_config; root=pathlib.Path('BOVAE/studies'); examples=sorted(root.glob('*/configs/example.json')); assert len(examples)==6, len(examples); [pipeline_config(path) for path in examples]; extras=sorted((root/'bovae_vs_egorse_curved_preimage_d10_d100'/'configs').glob('*.json')); [json.loads(path.read_text(encoding='utf-8')) for path in extras]; print('validated 6 BO-VAE examples and retained EGORSE controls')"
