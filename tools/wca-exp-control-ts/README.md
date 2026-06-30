# WCA Experiment Control TypeScript Scaffold

This is the first safe TypeScript shadow slice of the WCA experiment control
plane. It is intentionally isolated from the Python package, training code,
queue execution, artifact fetching, and report generation.

This package is non-production shadow mode only. It must not be used for active
experiment execution, queue submission, fetch, or report paths until after the
current V24 results are fetched, reviewed, and the Python-vs-TypeScript parity
requirements are explicitly accepted.

Interpret all outputs through `execution_state`. In this package, planner
results are `shadow_only`; readiness reports also stay `shadow_only` in this
Slice I1 package. `ok=true` means the local report or dry-run generated
successfully. It does not mean the command may execute on a VPS, and
`active_takeover_allowed` must remain false.

Current scope:

- contract types for `Manifest`, `JobProfile`, `GpuSlot`, `PlannerPolicy`, and `PlannerResult`;
- read-only strict control-manifest inspection for existing Python-owned manifests;
- read-only TS/Python manifest validation parity reporting;
- read-only TS/Python artifact inventory parity reporting;
- read-only TS/Python formal evidence stub report-gate parity reporting;
- shadow resource lock contract types for `ResourceLockRequest`, `ResourceLease`,
  `ResourceLockGrant`, and `ResourceLockConflict`;
- deterministic dry-run slot planning with the `priority_lpt` strategy;
- deterministic dry-run multi-resource placement with the
  `resource_best_fit_v1` strategy;
- local planner strategy comparison across supported shadow strategies using
  the same plan spec and resource profile rows;
- resource profile rows with explicit units for memory, time, throughput, GPU
  utilization, cache wait, and OOM status;
- read-only extraction of resource profile rows from `summary.json` and
  `train_log.csv`;
- read-only resource profile library audit for coverage, duplicate IDs, OOM
  rates, and freshness-gated formal requirements;
- read-only queue status parsing from already-fetched local `status.jsonl` and
  `queue_summary.json` artifacts;
- read-only queue status parity against a Python local artifact oracle;
- fail-closed migration readiness reporting that aggregates manifest,
  inventory, report-gate, and queue-status parity while keeping active takeover
  disabled until live remote/fetch/lease gates exist;
- JSON output containing `planned_jobs`, `rejected_jobs`, and `scheduler_decision`;
- JSON output containing shadow-only resource lock requests, grants, and
  conflicts for GPU windows, writable paths, and cache write paths;
- pure runtime lease TTL/conflict evaluation for future remote-safe runner
  design;
- local shadow runtime lease store validation for duplicate lease IDs, malformed
  resource leases, and owner/TTL shape;
- local shadow fetch package validation for report-first package files and raw
  archive byte/SHA integrity claims;
- local shadow remote-status snapshot validation for future live read-only
  status adapter output;
- local shadow WCA kernel equivalence gate validation for future
  PyTorch C++/CUDA optimization admission artifacts;
- local mock active-control lifecycle generation under shadow-only directories,
  producing deterministic queue artifacts for queue-status/parity gates while
  keeping active takeover disabled;
- local fixture and Node built-in tests.

Out of scope:

- active V24 execution;
- remote submission;
- queue mutation;
- artifact fetching;
- archive extraction;
- SSH or live remote status reads;
- formal report generation or mutation;
- model or training changes;
- replacing `scripts/wca_exp_control.py`.
- active resource locks, remote leases, or any lease enforcement outside this
  local dry-run process.
- watchdog or remote process liveness integration.
- compiling or running PyTorch C++/CUDA extensions.
- active-control execution beyond the local mock lifecycle.

The CLI explicitly rejects active commands such as `run`, `submit`, `fetch`,
`cancel`, and remote mutation/status aliases. Use the Python control plane for
active queues until a migration gate report approves takeover.

The resource lock and lease records are contract scaffolding only. Runtime lease
evaluation is pure local logic. They are not connected to Python, SSH, remote
runners, process managers, file-system locks, or GPU allocation APIs. A
`ResourceLockGrant` means only that the local shadow planner found no modeled
conflict for that request in the dry-run result.

Run from this directory:

```bash
npm test
npm run shadow:acceptance
npm run shadow:dry-run -- fixtures/sample-plan.json --pretty
npm run shadow:dry-run -- fixtures/profile-backed-plan.json --profile-rows fixtures/profile-rows.json --pretty
npm run shadow:planner-compare -- fixtures/profile-backed-plan.json --profile-rows fixtures/profile-rows.json --pretty
node --experimental-strip-types src/cli.ts inspect-manifest ../../configs/control/v24/westb_pdebench_v24_multidirection_bigtrain_manifest.json --pretty
npm run shadow:parity -- ../../configs/control/v21/westb_pdebench_v21_strict_reacdiff_r3_manifest.json ../../configs/control/v22/wca_scaling_and_efficiency_manifest.json ../../configs/control/v23/westb_pdebench_v23a_mixed_mainline_continue_manifest.json ../../configs/control/v24/westb_pdebench_v24_multidirection_bigtrain_manifest.json --repo-root ../.. --output /tmp/wca-exp-control-ts/v21-v24-parity.json
npm run shadow:inventory-parity -- ../../configs/control/v21/westb_pdebench_v21_strict_reacdiff_r3_manifest.json ../../configs/control/v22/wca_scaling_and_efficiency_manifest.json ../../configs/control/v23/westb_pdebench_v23a_mixed_mainline_continue_manifest.json ../../configs/control/v24/westb_pdebench_v24_multidirection_bigtrain_manifest.json --repo-root ../.. --output /tmp/wca-exp-control-ts/v21-v24-inventory-parity.json
npm run shadow:report-gate-parity -- ../../configs/control/v21/westb_pdebench_v21_strict_reacdiff_r3_manifest.json ../../configs/control/v22/wca_scaling_and_efficiency_manifest.json ../../configs/control/v23/westb_pdebench_v23a_mixed_mainline_continue_manifest.json ../../configs/control/v24/westb_pdebench_v24_multidirection_bigtrain_manifest.json --repo-root ../.. --output /tmp/wca-exp-control-ts/v21-v24-report-gate-parity.json
npm run shadow:extract-profiles -- ../../runs/westb_v22/scaling --repo-root ../.. --gpu-model "RTX PRO 6000" --output /tmp/wca-exp-control-ts/v22-resource-profiles.json
npm run shadow:audit-profiles -- /tmp/wca-exp-control-ts/v22-resource-profiles.json --pretty
npm run shadow:audit-profiles -- fixtures/profile-rows.json --requirements fixtures/profile-audit-requirements.json --pretty
npm run shadow:lease-store-validate -- /tmp/wca-exp-control-ts/runtime-leases.json --pretty
npm run shadow:fetch-package-validate -- /tmp/wca-exp-control-ts/report-first-package.json --pretty
npm run shadow:remote-status-snapshot-validate -- /tmp/wca-exp-control-ts/remote-status-snapshot.json --pretty
npm run shadow:kernel-equivalence-validate -- /tmp/wca-exp-control-ts/optimization_gate.json --pretty
npm run shadow:mock-active-control -- fixtures/mock-active-control-success.json --pretty
npm run shadow:queue-status -- /tmp/wca-exp-control-ts/mock-active-control/success/queue --pretty
npm run shadow:queue-status-parity -- /tmp/wca-exp-control-ts/mock-active-control/success/queue --pretty
npm run shadow:mock-active-control -- fixtures/mock-active-control-failure.json --pretty
npm run shadow:queue-status -- ../../artifacts/queues/westb_pdebench_v21_strict_reacdiff_r3 --pretty
npm run shadow:queue-status-parity -- ../../artifacts/queues/westb_pdebench_v21_strict_reacdiff_r3 --pretty
npm run shadow:migration-readiness -- ../../configs/control/v21/westb_pdebench_v21_strict_reacdiff_r3_manifest.json ../../configs/control/v22/wca_scaling_and_efficiency_manifest.json ../../configs/control/v23/westb_pdebench_v23a_mixed_mainline_continue_manifest.json ../../configs/control/v24/westb_pdebench_v24_multidirection_bigtrain_manifest.json --repo-root ../.. --queue-dir ../../artifacts/queues/westb_pdebench_v21_strict_reacdiff_r3 --lease-store /tmp/wca-exp-control-ts/runtime-leases.json --fetch-package /tmp/wca-exp-control-ts/report-first-package.json --remote-status-snapshot /tmp/wca-exp-control-ts/remote-status-snapshot.json --pretty
```

The scripts use Node's built-in TypeScript stripping and do not require
installing npm dependencies. No root package scripts are added, and this local
package is not imported by existing Python scripts.

`npm run shadow:acceptance` is the Slice I1 deterministic local acceptance
loop from `docs/91_infra_subagent_execution_plan.md`. It runs the Node
acceptance harness in `tests/shadow_acceptance.test.ts` and covers:

- planner dry-run with profile-backed unit fields and `execution_state=shadow_only`;
- planner comparison across supported shadow strategies;
- profile audit success plus fail-closed behavior for a missing required
  profile row;
- queue-status parsing from already-fetched local artifacts;
- migration readiness with `active_takeover_allowed=false` and
  `recommendation=keep_python_active`;
- kernel-equivalence artifact validation from a local fixture file.

This acceptance command performs no SSH, remote status polling, submit, fetch,
cancel, queue mutation, model execution, training, or eval work.

The planner summary includes unit-bearing fields such as
`expected_total_slot_seconds`, `expected_idle_slot_seconds`,
`expected_mean_gpu_utilization`, and `expected_min_memory_headroom_mb`. These
fields are required before any CP-SAT/MILP optimizer or learned resource
predictor can be trusted.

`resource_best_fit_v1` is a local-only deterministic heuristic. It keeps the
same role/duration job order as `priority_lpt`, but scores feasible GPU groups
by projected finish time, memory waste, slot readiness, and GPU id. It is useful
for avoiding obvious memory fragmentation in heterogeneous GPU pools, but it is
not a proof of global optimality.

`planner-compare` is local-only. It runs the supported shadow strategies against
the same plan spec and optional resource profile rows, then reports makespan,
idle slot seconds, utilization, memory headroom, and memory-waste metrics. It is
an audit tool for choosing which strategy deserves a controlled run; it does
not submit jobs, reserve GPUs, fetch artifacts, or prove the selected strategy
is globally optimal.

GPU leases are currently exclusive. `allow_multiprocess_per_gpu=true` fails
closed because the runtime lease model has no shared-GPU mode, memory-fraction
lease, or per-process conflict rule yet. Do not use that policy to infer that
multiple training jobs can safely share one CUDA device.

`queue-status` is local-only. It reads queue artifacts already present on disk
and does not SSH, fetch, submit, cancel, or mutate remote state.
`queue-status-parity` compares that parser against a Python local artifact
oracle with the same no-remote boundary.

`mock-active-control` is local-only. It accepts specs with
`mode="local_mock_active_control"` and `allow_remote=false`, validates that the
output directory is under `/tmp/wca-exp-control-ts` or
`tools/wca-exp-control-ts/artifacts/control_shadow`, then writes deterministic
`manifest.json`, `remote_submission.json`, `status.jsonl`, and
`queue_summary.json` files under a local `queue/` directory. Failure fixtures
return structured `ok=false` reports and failed queue artifacts; unsafe output
specs fail before writing. Reports always include
`active_takeover_allowed=false`.

`audit-profiles` is local-only. It validates resource profile rows before they
drive planner estimates. Formal requirements should declare `recipe_id`,
matching fields, `min_non_oom_samples`, optional `max_oom_rate`, and optional
`max_stale_age_seconds`. Freshness-gated requirements fail if rows lack
`observed_at_epoch_seconds`, preventing old or anonymous constants from being
silently reused for formal scheduling.

`lease-store-validate` is local-only. It validates the file shape for a future
runtime lease store but does not create, refresh, release, or enforce leases.
The accepted store mode is `shadow_runtime_lease_store`; resource lease
enforcement must remain `shadow_only` until a separate remote runner gate
connects it to real GPU/process ownership.

`fetch-package-validate` is local-only. It validates an already fetched package
manifest, checks compact report paths, and for raw-complete packages compares a
local archive's byte size and streamed sha256 against the manifest's remote
claim. It does not SSH, rsync, retry transfers, extract archives, or decide that
TypeScript may fetch active queues.

`remote-status-snapshot-validate` is local-only. It validates the JSON shape
that a future live SSH read-only adapter must emit. It does not SSH, poll a VPS,
read remote files, fetch artifacts, or change queue state.

`kernel-equivalence-validate` is local-only. It validates an
`optimization_gate.json` produced by future Python/PyTorch equivalence scripts.
It does not import torch, compile extensions, run CUDA, select a backend, or
change model configs. Promotion is allowed only when the gate preserves
`FullRecursiveWorldStateNCA`, keeps the default backend unchanged, passes
required output/loss/gradient coverage, and satisfies low-precision quality
requirements when bf16/fp16 cases are present.

`migration-readiness` is also local-only. `ok=true` means the diagnostic report
was generated, not that TypeScript may take over active queues. The takeover
decision is `active_takeover_allowed`; for Slice I1 it remains false even when
local validation evidence passes. Optional `--lease-store` and
`--fetch-package` inputs, plus `--remote-status-snapshot`, only provide local
shadow validation evidence; passing those gates does not enable remote fetch,
remote submission, archive extraction, live status polling, or real lease
enforcement.
