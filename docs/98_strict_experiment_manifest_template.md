# Strict Experiment Manifest Template

Status: active template.

This template exists because strict experiments must not be hand-assembled from
scratch. Every formal WCA experiment should start from this contract and then
specialize the model matrix, run directories, and queue ids.

## Purpose

The control plane must fail before GPU work when artifact ownership, eval plans,
label/token contracts, or report contracts are ambiguous.

The common failure mode is:

```text
formal horizon eval writes eval_plan.json
sentinel eval writes another eval_plan.json in the same directory
strict report reads the wrong plan
```

That is not a model failure. It is an artifact ownership failure.

## Required Layers

### 1. Manifest Layer

Each strict manifest must define:

- `protocol.claim_id`
- `protocol.changed_variable`
- `protocol.hypothesis`
- `protocol.falsification_condition`
- `eval.horizons`
- `eval.eval_samples`
- `eval.field_split`
- `checkpoint.selection_rule`
- `model_matrix`
- `artifacts.required_paths`
- `report_contract.required_outputs`
- `report_contract.artifact_owners`
- a single formal label/eval-token contract per strict comparison table.

For V25+ strict submit manifests, `report_contract.required_outputs` and
`report_contract.artifact_owners` are a hard control-plane gate, not a
recommendation. Every owner path must also be listed in
`artifacts.required_paths`, so fetch/inventory/report all reason over the same
artifact set.

### 2. Producer Layer

Each job may write only its owned outputs.

Recommended ownership:

| Artifact | Owner |
|---|---|
| `preflight.json` | preflight job |
| `eval_plan.json` | formal horizon eval job |
| `per_sample_rows.csv` | formal horizon eval job |
| `results_by_horizon.csv` | formal horizon eval job |
| `results_by_horizon.md` | formal horizon eval job |
| `summary.json` | formal horizon eval job |
| `sentinel_eval_plan.json` | sentinel job |
| `sentinel_results.csv` | sentinel job |
| `sentinel_summary.json` | sentinel job |
| `results.md` | strict report job |

`eval_field_sentinels.py` must never own or write formal `eval_plan.json`.

### 3. Consumer Layer

`report_v20_pdebench.py` consumes:

- `results_by_horizon.csv`
- `summary.json`
- `eval_plan.json`
- `per_sample_rows.csv`
- `sentinel_summary.json` and `sentinel_results.csv` when sentinel guardrails
  are part of the report contract
- `dataset_audit.json`
- `cache_manifest.json`

The report script validates. It does not repair, infer, or silently accept
incompatible artifact contracts.

When sentinel artifacts are part of the contract, the report must distinguish
hard anti-cheat gates from robustness diagnostics. Label leakage and required
horizon-conditioning sentinels are hard gates. Input-shuffle degradation is a
required robustness diagnostic unless the experiment manifest declares a
model-family-specific hard threshold. The report must expose diagnostic counts,
affected model families, and ratio summaries rather than hiding non-degraded
rows.

### 4. Gate Layer

Before submit, `scripts/wca_exp_control.py submit` runs the local pre-cloud gate
automatically. This is a hard blocker: submit fails before any remote API call if
no safe gate jobs are present, or if any selected static/compile/preflight/
contract-check job fails. The local gate must execute the same generated-queue
gate commands and artifact contract as the cloud queue wherever possible. Missing
Python dependencies are not a reason to skip the gate: install them into the
local `.venv`, then rerun. Paid GPU queues must not be used as syntax,
dependency, report-template, or contract-test probes.

Minimum local control-plane gates:

```bash
.venv/bin/python scripts/wca_exp_control.py validate <manifest.json>
.venv/bin/python scripts/wca_exp_control.py plan <manifest.json>
```

After `plan`, the generated queue's pre-train/pre-eval gate jobs must be
discoverable by the control plane. For strict queues, this normally includes
static tests, `py_compile`, preflight, and contract-check jobs. Safe gate jobs
can be marked explicitly with `local_precloud_gate=true`; otherwise the control
plane selects ids beginning with `strict_`, `compile_`, or `preflight`, plus ids
containing `static_tests`. It also selects `contract_check*` only until the first
producer job (`train_*`, `eval_*`, `report_*`, `audit_*`, or `profile_*`) appears
in queue order, because later contract checks may depend on artifacts produced by
the same queue. If a gate needs remote-only artifacts, fetch those artifacts
first or split the gate into a local dependency/schema check plus an explicit
remote artifact preflight. Do not hide that split in the report.

For strict PDEBench report-only recovery, the minimum static gates are:

```bash
.venv/bin/python -m pytest \
  tests/test_field_horizon_stratified_eval.py \
  tests/test_eval_field_sentinels.py \
  tests/test_v20_pdebench_report.py \
  tests/test_wca_exp_control.py

.venv/bin/python -m py_compile \
  scripts/eval_field_horizon_stratified.py \
  scripts/eval_field_sentinels.py \
  scripts/report_v20_pdebench.py \
  scripts/wca_exp_control.py
```

For horizon-stratified field eval, strict comparison tables must also pass a
CPU-only contract check before GPU model evaluation:

```bash
.venv/bin/python scripts/eval_field_horizon_stratified.py \
  <strict-run-dirs...> \
  --horizons 1,2,4,8 \
  --eval-samples 64 \
  --eval-batch-size 1 \
  --eval-batches 64 \
  --field-split test \
  --contract-check-only
```

This check must reject:

- raw-field anchors mixed with token-level WCA/token baselines in the same strict
  table;
- different grid/patch/token geometry in the same strict table;
- different dataset path, output dimension, split windows, input steps, or
  stride in the same strict table.

Identical start-index hashes are not sufficient for fairness. Persistence MSE is
computed over the configured token labels, so patch-8 and patch-16 runs can have
the same physical start indices but different persistence baselines.

## Template Snippet

```json
{
  "report_contract": {
    "formal_status": "formal",
    "primary_metric": "horizon_stratified_final_mse_and_relative_l2",
    "required_outputs": [
      "artifacts/reports/<experiment>/eval_plan.json",
      "artifacts/reports/<experiment>/per_sample_rows.csv",
      "artifacts/reports/<experiment>/results_by_horizon.csv",
      "artifacts/reports/<experiment>/summary.json",
      "artifacts/reports/<experiment>/sentinel_eval_plan.json",
      "artifacts/reports/<experiment>/sentinel_results.csv",
      "artifacts/reports/<experiment>/sentinel_summary.json",
      "artifacts/reports/<experiment>/results.md"
    ],
    "artifact_owners": [
      {
        "path": "artifacts/reports/<experiment>/eval_plan.json",
        "owner": "eval_<experiment>_horizon_stratified"
      },
      {
        "path": "artifacts/reports/<experiment>/sentinel_eval_plan.json",
        "owner": "eval_<experiment>_sentinels"
      },
      {
        "path": "artifacts/reports/<experiment>/results.md",
        "owner": "report_<experiment>_strict"
      }
    ]
  }
}
```

## Acceptance Criteria

A strict experiment template is acceptable only when:

- every formal artifact has exactly one owner;
- owner ids exist in `jobs`;
- no sentinel job owns `eval_plan.json`;
- `eval_plan.json` and `sentinel_eval_plan.json` are separate files;
- if sentinel guardrails are declared, report generation consumes
  `sentinel_summary.json` and `sentinel_results.csv`;
- sentinel checkpoint kinds must cover the formal checkpoint kinds;
- strict report succeeds from fetched artifacts without rewriting raw results;
- inventory reports no missing required paths.
- strict horizon tables contain exactly one label/eval-token contract.

## Evidence Status

This is a system-debt guardrail. It does not change WCA model claims. It prevents
false evidence caused by artifact collision or report-contract drift.
