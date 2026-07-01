# WCA Experiment Control TypeScript Shadow Plane

This package is a public, non-production TypeScript shadow of selected WCA
experiment-control concepts. It is included to document and exercise the typed
planner/control-plane migration path.

It does not submit jobs, fetch artifacts, mutate queues, SSH to remote machines,
or replace `scripts/wca_exp_control.py`.

## Public Smoke

```bash
cd tools/wca-exp-control-ts
npm install
npm run shadow:dry-run -- fixtures/sample-plan.json --pretty
npm run shadow:planner-compare -- fixtures/profile-backed-plan.json --profile-rows fixtures/profile-rows.json --pretty
```

The full internal TS parity tests depend on private run inventories and are not
part of the public source export. Public users should treat this package as a
shadow planner scaffold, not as the canonical experiment launcher.
