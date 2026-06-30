# AGENTS.md

## Mission

WCA is World Cellular Automata. The protected baseline is Full Dense RWS-NCA:
`H [B,N,D] -> local worlds L [B,N,N,D] -> pair dynamics -> recomposed H_next`.
Keep this file short. Long plans, historical results, runbooks, and loop details
belong in `docs/`.

## Agent Roles

- Main agent: program owner, architect, integrator, and executor of approved
  operations. It owns project progress, high-level design, experiment plans,
  conclusions, final integration, and final local/remote execution. It is not
  the default code writer for non-trivial work.
- Worker subagent: bounded implementation or concrete design owner. It receives
  a narrow task, allowed files, acceptance criteria, and verification command.
- Reviewer subagent: independent spec, code-quality, experiment-quality, and
  evidence-quality reviewer. It reviews worker output; high-risk review
  conclusions should get a second reviewer where practical.
- Explorer subagent: read-only investigator. It maps facts, files, artifacts,
  alternatives, and risks without editing.

Delegate non-trivial development, auxiliary design, and review. The main agent
arbitrates conflicts and makes the final integration decision.

## Loop Contract

Every loop must declare objective, acceptance criteria, verification, and stop
condition before implementation starts.

Flow:

1. Goal: name the user-facing outcome and non-goals.
2. Plan: choose the smallest loop that can close the goal.
3. Task slices: split into bounded worker, explorer, and reviewer prompts.
4. Subagent execution: workers produce patches/designs; explorers produce facts.
5. Review: reviewers check spec fit, scientific validity, quality, and tests.
6. Verification: run the narrowest command or artifact gate.
7. Evidence: record files, commands, artifacts, conclusions, and risks.
8. Stop or next: decide promote, repeat, prune, freeze, defer, or ask user.

No loop continues just because more improvement is possible. A loop stops when
its acceptance criteria pass, a blocker is proved, or the next decision needs
user approval.

## Baseline Protection

The Full Dense RWS-NCA reference must preserve:

- `H: [B,N,D]`
- `L: [B,N,N,D]`
- dense pair interaction semantics equivalent to `[B,N,N,N,D]`
- world -> local worlds -> recomposed world recursion

Chunking is allowed only when it preserves full dense semantics. Do not
silently replace the baseline with a GNN, Transformer, attention block,
`A @ H` diffusion, sparse edge-index model, or self-reference variant. Sparse,
hybrid, multiscale, egocentric, predictive, surprise, boredom, RL, or memory
mechanisms must be explicit variants or ablations under
`src/wca/models/variants/`.

## Data And Evidence Rules

- Model files must not import task generators, BFS oracle, WeatherBench/PDEBench
  loaders, or experiment scripts.
- Models may receive `H` and adjacency. They must not receive labels, raw BFS
  distances, oracle paths, greedy paths, future frames, or target fields as
  inputs.
- Labels and target fields are allowed only in losses, metrics, evaluation, and
  reports.
- Channel meanings must live in constants or schema files, not hidden-dimension
  inference.
- Maze batches require `H`, `adjacency`, `distance_field`, `distance_mask`,
  `start_idx`, `goal_idx`, `raw_distance`, and `maze_id`.
- Maze metrics must include `eval_mae`, `eval_mse`, `path_ok/path_success_rate`,
  `path_opt/path_optimal_rate`, `exact_start_distance_acc`, `goal_rank`,
  `spurious_local_minima_count`, `monotonic_descent_accuracy`, and
  `neighbor_order_accuracy`; `path_opt` is primary and MSE alone is insufficient.
- Weather/PDEBench formal comparisons require fixed eval plans/start indices,
  source and split manifests, per-sample rows, horizon rows, checkpoint policy,
  and report gates.

## Experiment And Infra Rules

- Pre-strict PDEBench queues are historical only. New formal evidence starts
  from V20d+ strict lines or later strict manifests.
- Before any remote/cloud submit, `scripts/wca_exp_control.py submit` must run
  the local pre-cloud gate automatically. This is a hard blocker: if no safe gate
  jobs are present, or any static/compile/preflight/contract-check job fails,
  remote submit must not be called. If the local venv is missing dependencies
  needed by the gate, install them locally and rerun before uploading. Do not use
  paid GPU queues as syntax, dependency, report-template, or contract-test
  probes.
- Python remains the active queue/fetch/report authority until TypeScript passes
  migration gates and the user accepts active takeover.
- TypeScript control-plane work is shadow-only until takeover is approved.
- `ok=true` is not executable. Execution requires the declared execution state
  plus manifest, resource, and evidence gates.
- Planner output must carry units for memory, time, utilization, idle time, and
  rejection/conflict counts. Do not claim global optimality without an exact
  solver gate.
- Runtime parallelism needs atomic GPU-group leases, single-writer artifact
  paths, shared-read/exclusive-write cache policy, TTL, and stale-lease recovery.
- Do not submit new GPU queues while an official strict queue is active unless
  the user explicitly asks.

## Engineering Rules

- Make surgical changes only. Do not refactor adjacent systems unless required.
- Prefer small modules, type hints, testable functions, and explicit schemas.
- Keep model code independent from task code.
- Write docs by type: C4/arc42 for architecture, Diataxis for docs taxonomy,
  ADR for decisions, strict experiment template for results.
- Use `rg` / `rg --files` first for search.
- Use `apply_patch` for manual edits.
- Run the narrowest useful validation: `git diff --check`, targeted tests, TS
  shadow tests, report/inventory gates, or remote status/fetch gates depending
  on touched files.
- Report in this hierarchy: architecture -> module -> file -> symbol ->
  concrete logic. Tie claims to paths, artifacts, symbols, and line references.

## Security

Never commit credentials, private URLs, secret chat logs, or unsanitized reviews.

## Reading Index

- Loop OS: `docs/80_loop_engineering_operating_system.md`
- Infra map/gates: `docs/79_experiment_infra_system_map_and_test_gates.md`
- Macro validity: `docs/77_wca_macro_architecture_and_algorithm_validity_criteria.md`
- Infra stop rules: `docs/76_infra_execution_completion_contract.md`
- TS migration: `docs/67_wca_ts_control_plane_cuda_migration_master_plan.md`
- Planner: `docs/75_profile_backed_dynamic_planner_contract.md`
- Leases: `docs/66_parallel_scheduler_locks_and_leases.md`
- CUDA gate: `docs/74_wca_cuda_kernel_equivalence_gate.md`
