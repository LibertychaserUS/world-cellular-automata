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
```

When TypeScript shadow control-plane source files are needed:

```bash
cd tools/wca-exp-control-ts
npm install
npm install
npm run build --if-present
```

## Artifact Policy

Use the report and artifact repositories for PDFs, large result tables,
prediction panels, and Zenodo handoff packages.
