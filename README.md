# World Cellular Automata

World Cellular Automata (WCA) is a research codebase for recursive
world-state prediction. The protected baseline represents a world state as
node vectors, expands it into node-indexed local worlds, evolves those local
worlds through pair interactions, and recomposes evolved centers into the next
world state.

```text
H_t [B,N,D] -> L_t [B,N,N,D] -> local-world pair evolution -> H_{t+1} [B,N,D]
```

## What This Repository Contains

- WCA model code under `src/wca/`.
- Training and evaluation entrypoints for public smoke and field tasks.
- Strict local evaluation and audit utilities.
- Public experiment configs for reproducible follow-up.
- Tests for model shapes, schema boundaries, no-label-leakage, evaluation, and
  sentinel gates.

## What This Repository Does Not Contain

Raw artifacts, private VPS logs, remote machine configs, checkpoints, and large
prediction dumps are intentionally excluded. Report PDFs and figures are
released separately in `world-cellular-automata-technical-report`. Heavy result
tables or prediction panels should use a dataset/artifact repository.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev,real-data,viz]
python -m pytest tests/test_model_shapes.py tests/test_field_model_smoke.py tests/test_no_label_leakage.py
```

For the TypeScript shadow control plane:

```bash
cd tools/wca-exp-control-ts
npm install
npm test -- --runInBand
```

## Evidence Boundary

Current public evidence should be read as a staged research record:

- strongest formal row: token-level PDEBench reaction-diffusion h8x2 rollout;
- attribution work: learnable-interface and no-recursion controls;
- secondary evidence: maze potential fields and WeatherBench-style transfer;
- diagnostic boundary: larger-token PatchMean interfaces under current capacity.

Do not interpret patch-repeated qualitative panels as native full-field
superiority claims.
