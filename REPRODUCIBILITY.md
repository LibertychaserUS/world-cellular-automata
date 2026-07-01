# Reproducibility

This repository is a curated source export. Heavy artifacts, raw datasets, and
private queue logs are intentionally excluded.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev,real-data,viz]
```

## Smoke Tests

```bash
python -m pytest tests/test_model_shapes.py tests/test_field_model_smoke.py tests/test_no_label_leakage.py
python -m pytest tests/test_field_horizon_stratified_eval.py tests/test_eval_field_sentinels.py
```

## Control Plane

```bash
python scripts/wca_exp_control.py plan path/to/manifest.json
python scripts/wca_exp_control.py validate path/to/manifest.json
python scripts/wca_exp_control.py submit path/to/manifest.json
python scripts/wca_exp_control.py status path/to/manifest.json
python scripts/wca_exp_control.py fetch path/to/manifest.json
python scripts/wca_exp_control.py inventory path/to/manifest.json
python scripts/wca_exp_control.py report path/to/manifest.json
```

Remote machine details are intentionally excluded from the public repository.
Run local gates before any paid remote submission.

## TypeScript Shadow Control Plane

```bash
cd tools/wca-exp-control-ts
npm install
npm run shadow:dry-run -- fixtures/sample-plan.json --pretty
npm run shadow:planner-compare -- fixtures/profile-backed-plan.json --profile-rows fixtures/profile-rows.json --pretty
```

## Artifact Policy

Use the report and artifact repositories for PDFs, large result tables,
prediction panels, and Zenodo handoff packages.
