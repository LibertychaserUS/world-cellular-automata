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

## Related Public Repositories

- Technical report, PDFs, LaTeX, figures, tables, and release package:
  <https://github.com/LibertychaserUS/world-cellular-automata-technical-report>
- Hugging Face report/artifact mirror:
  <https://huggingface.co/datasets/Chaser111/world-cellular-automata-technical-report>

## What This Repository Does Not Contain

Raw artifacts, private VPS logs, remote machine configs, checkpoints, and large
prediction dumps are intentionally excluded. Report PDFs and figures are
released separately in `world-cellular-automata-technical-report`. Heavy result
tables or prediction panels should use a dataset/artifact repository.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev,real-data,viz]
```

Use `.[dev]` for code/test work only. Add `real-data` for PDEBench or
WeatherBench-style data preparation, and `viz` for figure/table generation.

## Quick Verification

Run the small local checks before trusting any experiment output:

```bash
python -m pytest tests/test_model_shapes.py tests/test_field_model_smoke.py tests/test_no_label_leakage.py
python -m pytest tests/test_field_horizon_stratified_eval.py tests/test_eval_field_sentinels.py
```

The exported test suite is intentionally lightweight; it checks tensor shape
contracts, schema separation, horizon-stratified evaluation, sentinel behavior,
and dense/native mock equivalence without requiring private artifacts.

## Smoke Training And Evaluation

The synthetic heat smoke config exercises the public field-prediction training
path without downloading large datasets:

```bash
python scripts/train_field.py --config configs/experiments/field_synthetic_heat_smoke.yaml
```

If a run produces an output directory, evaluate it with the horizon-stratified
evaluator using the corresponding config/checkpoint paths:

```bash
python scripts/eval_field_horizon_stratified.py --help
python scripts/eval_field_sentinels.py --help
```

For maze smoke work:

```bash
python scripts/generate_maze_pool.py --help
python scripts/train_maze.py --help
python scripts/eval_maze.py --help
```

## Experiment Control Plane

The public release includes the Python control-plane entrypoint and a
TypeScript shadow control-plane prototype. The control plane is designed around
manifested experiments rather than ad hoc shell commands.

Typical lifecycle:

```bash
python scripts/wca_exp_control.py plan path/to/manifest.json
python scripts/wca_exp_control.py validate path/to/manifest.json
python scripts/wca_exp_control.py submit path/to/manifest.json
python scripts/wca_exp_control.py status path/to/manifest.json
python scripts/wca_exp_control.py fetch path/to/manifest.json
python scripts/wca_exp_control.py inventory path/to/manifest.json
python scripts/wca_exp_control.py report path/to/manifest.json
```

Remote execution requires machine-specific SSH configuration that is not part
of the public repository. Before submitting to paid hardware, local static,
compile, preflight, and contract gates should pass.

For the TypeScript shadow control plane:

```bash
cd tools/wca-exp-control-ts
npm install
npm run shadow:dry-run -- fixtures/sample-plan.json --pretty
npm run shadow:planner-compare -- fixtures/profile-backed-plan.json --profile-rows fixtures/profile-rows.json --pretty
```

It is not yet the canonical launcher; it is a typed replacement path for future
control-plane hardening.

## Reproducing The Main Evidence Path

The public config names encode the staged evidence tree:

- V25: recursion-depth ladder and h8x2 rollout evidence.
- V25c/V25e: attribution and guardrail controls.
- V26/V27/V28: N=256 capacity, token-equivalent baselines, and interface
  attribution diagnostics.

To reproduce a formal line, use the exact config/manifest, keep the declared
split/seed/eval-index plan fixed, and report both numerical metrics and gate
status. Historical raw anchors and token-level fair comparisons should not be
mixed into one formal table without stating the interface contract.

## Evidence Boundary

Current public evidence should be read as a staged research record:

- strongest formal row: token-level PDEBench reaction-diffusion h8x2 rollout;
- attribution work: learnable-interface and no-recursion controls;
- secondary evidence: maze potential fields and WeatherBench-style transfer;
- diagnostic boundary: larger-token PatchMean interfaces under current capacity.

Do not interpret patch-repeated qualitative panels as native full-field
superiority claims.

## Reporting

The report repository contains the human-readable technical report and figure
package. This source repository contains the executable code path. When citing
or reproducing results, cite both the code commit and the report package
manifest.
