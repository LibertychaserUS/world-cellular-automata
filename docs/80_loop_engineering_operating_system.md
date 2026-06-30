# WCA Loop Engineering Operating System

Date: 2026-06-23

Status: active collaboration contract. This document defines how WCA work is
framed, delegated, reviewed, verified, and stopped. It is not a feature plan.

## Purpose

WCA needs a stable development and collaboration process before more feature
work. The main agent should act as program owner, architect, integrator, and
executor of approved operations. Concrete implementation, auxiliary design,
read-only investigation, and review should be delegated to subagents when the
work is non-trivial.

Every loop must have:

- objective;
- acceptance criteria;
- verification;
- stop condition.

If any of those four are missing, the first task is to define them.

## Operating Principles

- Small closed loops beat broad open loops.
- The main agent owns the whole program state, not every code edit.
- Delegation must be bounded by files, outputs, and acceptance criteria.
- Reviews should be independent from implementation.
- Evidence is a deliverable, not a postscript.
- Stop when the loop is accepted, blocked, frozen, or needs a user decision.
- No model semantics, experiment evidence class, queue state, or remote runtime
  authority changes without an explicit gate.

## Documentation Standards

WCA docs must be written by document type, not as one mixed narrative. Use these
formats unless a better standard is explicitly chosen and recorded.

Standards referenced:

- C4 model: system context, container, component, code, dynamic, and deployment
  views for software architecture diagrams.
- arc42: architecture documentation structure covering goals, constraints,
  context, solution strategy, building blocks, runtime view, deployment view,
  risks, and quality requirements.
- Diataxis: documentation taxonomy separating tutorials, how-to guides,
  reference, and explanation.
- ADR / MADR: decision records with context, decision, alternatives,
  consequences, and status.

| Document Type | Standard | Required Structure | Typical Files |
|---|---|---|---|
| Architecture map | C4 model + arc42 | system context, containers/modules, components, runtime/deployment view, risks | `docs/79_*`, future architecture docs |
| Operating process | lifecycle + RACI + state machine | roles, lifecycle, RACI, states, failure handling, done criteria | this document |
| How-to / runbook | Diataxis how-to | goal, prerequisites, command sequence, expected output, rollback | VPS/bootstrap/runbook docs |
| Reference | Diataxis reference | stable facts, schemas, commands, fields, invariants | schema/control-plane references |
| Decision record | ADR / MADR style | context, decision, alternatives, consequences, status | major architecture choices |
| Experiment protocol | strict research protocol | hypothesis, data, split, model, baseline, metric, seed/eval plan, success gate | V21+ formal plans |
| Experiment result | strict evidence report | config, artifacts, raw rows, statistical result, failure analysis, next action | result analysis docs |

Formatting rules:

- Architecture docs must separate system context, module/container view,
  component view, runtime flow, and deployment/resource view. A single diagram
  is not enough for major infra or model claims.
- Diagrams are supporting evidence, not the claim. Each diagram must be paired
  with text naming responsibilities, inputs/outputs, dependencies, and control
  flow.
- ADRs are for decisions that constrain future work. Do not bury decisions only
  in chat summaries.
- Experiment reports must distinguish formal, provisional, capacity, and
  exploratory evidence.
- Every report should be navigable by:

```text
architecture -> module -> file/artifact -> symbol/field -> concrete logic
```

## Lifecycle

### 1. Intake

The main agent extracts:

- user-facing goal;
- repo/worktree scope;
- allowed files and forbidden files;
- whether this is process, design, code, experiment, infra, or review work;
- current risks, active queues, dirty files, and likely verification.

For WCA experiment or infra work, check whether an official strict queue is
active before planning new execution.

### 2. Goal Framing

The main agent writes or confirms a goal using the goal template below. The
goal must include non-goals so the loop cannot quietly become feature work.

### 3. Plan

The main agent chooses the smallest loop that can close the goal. The plan must
name:

- task slices;
- which slices are delegated;
- expected review;
- verification command or artifact gate;
- stop condition.

### 4. Task Slicing

Slice by ownership boundary, not by convenience. A good slice has one role, one
output, and one acceptance gate.

Examples:

- explorer: map where maze metrics infer goal from channels;
- worker: patch only the explicit `goal_idx` metric path;
- reviewer: check oracle separation and metric correctness;
- main: integrate, run targeted tests, update evidence.

### 5. Subagent Execution

Subagents receive a prompt contract. They should not infer hidden permission.
Workers may edit only declared files. Explorers are read-only. Reviewers do not
rewrite the patch unless asked for a concrete fix slice.

### 6. Integration

The main agent reads all outputs, resolves conflicts, rejects drift, and applies
the final project judgment. Integration includes checking that the result still
fits the high-level architecture, experiment plan, and current WCA evidence
rules.

### 7. Review

Review happens after implementation or design output and before final
acceptance. For high-risk work, use two review passes where practical:

- spec review: does the planned change satisfy the goal and avoid forbidden
  directions?
- output review: does the patch, design, or report satisfy the acceptance gate?

### 8. Verification

Run the narrowest useful verification:

- docs-only: `git diff --check`;
- Python logic: targeted pytest or script;
- schema/control-plane: manifest or parity tests;
- experiment evidence: inventory/report gates;
- remote work: status/fetch/report commands, not new submits unless approved.

Verification failure keeps the loop open only if the objective can still be
closed in the same bounded loop. Otherwise freeze or ask.

### 9. Evidence

The main agent records:

- changed files;
- decisions and rejected alternatives;
- commands run and results;
- artifacts or reports produced;
- unresolved risks;
- next decision state.

### 10. Stop Or Next

The loop stops in one of these states:

- accepted;
- repeat with a narrower goal;
- prune the direction;
- freeze as historical or invalid evidence;
- blocked on missing input/resource;
- needs user decision;
- delegated follow-up.

## RACI

| Work Item | Main Agent | Worker | Reviewer | Explorer | User |
|---|---|---|---|---|---|
| Goal and non-goals | Accountable | Consulted | Consulted | Consulted | Approves priority |
| Architecture direction | Accountable | Consulted | Reviews | Investigates | Decides major pivots |
| Project progress state | Accountable | Informs | Reviews evidence | Informs | Reads outcome |
| Concrete implementation | Integrates | Responsible | Reviews | None | Approves scope if needed |
| Auxiliary design | Accountable | Responsible | Reviews | Investigates | Approves major choices |
| Read-only investigation | Uses output | None | May review | Responsible | None |
| Code/spec review | Resolves | Responds | Responsible | May supply facts | None |
| Experiment plan | Accountable | Drafts slices | Reviews validity | Investigates prior evidence | Approves costly runs |
| Final execution operations | Responsible | None | May review gate | May inspect state | Approves risky ops |
| Evidence and conclusion | Accountable | Supplies details | Reviews quality | Supplies facts | Reads decision |

## Goal Template

Use this template for any non-trivial loop.

```text
Loop ID:
Objective:
Why now:
Non-goals:
Repo/worktree:
Allowed files:
Forbidden actions:
Acceptance criteria:
Verification:
Stop condition:
Subagent slices:
Review plan:
Evidence target:
Decision states available:
```

Acceptance criteria should be observable. "Improve infra" is not acceptable.
"`docs/79` names the changed layer, the manifest validator rejects missing
source manifests, and `git diff --check` passes" is acceptable.

## Subagent Prompt Contract

Every subagent prompt should include:

- role: worker, reviewer, or explorer;
- context: relevant files, docs, current decision state;
- objective: one bounded outcome;
- allowed files or read-only scope;
- forbidden actions;
- acceptance criteria;
- expected output format;
- verification to run or not run;
- time or depth limit when useful.

### Worker Prompt Requirements

Worker prompts must say:

- exact files or modules allowed;
- whether tests may be edited;
- expected patch shape;
- behavior that must not change;
- command the worker should run, if any;
- output summary required for reviewer.

Workers should not perform broad cleanup, rename modules, change experiment
evidence class, submit queues, or modify remote state unless explicitly asked.

### Reviewer Prompt Requirements

Reviewer prompts must say what to review against:

- user requirement;
- architecture or algorithm invariant;
- experiment/data validity rule;
- code-quality rule;
- evidence and test gate.

Reviewers should return findings first, ordered by severity, with file or
artifact references. They should distinguish "blocks acceptance" from "follow-up
cleanup".

### Explorer Prompt Requirements

Explorer prompts must be read-only and ask for facts:

- relevant files and symbols;
- current behavior;
- prior reports or artifacts;
- risks and unknowns;
- possible task slices.

Explorers should not propose large rewrites unless asked for alternatives.

## Review Protocol

Use this checklist for spec and output review:

- Does the work satisfy the stated objective and non-goals?
- Does it preserve Full Dense RWS-NCA semantics unless this is an approved
  variant?
- Does model code stay separated from data generators, oracles, loaders, and
  experiment scripts?
- Are channel meanings, start indices, goal indices, and dataset schemas
  explicit?
- For Weather/PDEBench, are split/source manifests, fixed eval plans, per-sample
  rows, horizon rows, checkpoint policy, and report gates preserved?
- Are active-control-plane and shadow-control-plane boundaries preserved?
- Are tests or artifact gates proportional to the risk?
- Are security rules satisfied?
- Is there a clear stop or next decision?

Severity:

- P0: invalidates scientific claim, corrupts data/model semantics, leaks
  secrets, or performs unauthorized execution.
- P1: blocks acceptance or creates likely regression.
- P2: should fix soon but does not block the loop.
- P3: optional improvement or style note.

The main agent must resolve all P0/P1 findings before acceptance.

## Experiment Loop

Use this for training/evaluation/research loops.

1. State research question and evidence class: formal, provisional, capacity,
   or exploratory.
2. Define candidate, baseline, dataset, split, metrics, seeds, eval plan, and
   primary decision metric.
3. Freeze or reference manifest and analysis plan.
4. Run provenance and leakage checks.
5. Run planner/resource checks with units.
6. Submit through `scripts/wca_exp_control.py submit`, which automatically runs
   the generated queue's local pre-cloud gate first. Missing safe gate jobs or
   failing static/compile/preflight/contract-check jobs are hard blockers.
   Install missing local `.venv` dependencies and rerun; do not spend remote GPU
   time discovering syntax, dependency, report-template, or contract-test
   failures.
7. Confirm queue state and approval before submit.
8. Execute only through the active authority.
9. Fetch or aggregate artifacts after producer stability is proved.
10. Run inventory, report, and statistical gates.
11. Decide promote, repeat, prune, freeze, or defer.

Experiment done means the conclusion is supported by artifacts, not by the fact
that training completed.

## Infra Loop

Use this for control plane, scheduler, planner, cache, CUDA, reporting, and
runtime work.

1. Name the layer changed: manifest, validator, planner, lease, queue, runtime,
   artifact, report, statistical decision, or migration.
2. State active vs shadow authority.
3. Define the interface or contract before implementation.
4. Add fixture or targeted test for the contract.
5. Implement the smallest slice.
6. Review for execution safety and backwards compatibility.
7. Verify with static checks, targeted tests, parity reports, or dry-run gates.
8. Decide whether the result is shadow-only, eligible for human review,
   executable, blocked, or complete.

Infra done means the gate is satisfied and the next operator decision is clear.
It does not mean every possible guardrail has been built.

## Decision States

| State | Meaning | Allowed Next Action |
|---|---|---|
| draft | Goal or gate is incomplete | clarify or run explorer |
| planned | Goal, slices, gate, and stop condition exist | delegate or implement trivial slice |
| delegated | Subagent work is in progress or returned | integrate or request revision |
| under_review | Output exists but has not passed review | resolve findings |
| verified | Acceptance gate passed | record evidence and decide |
| accepted | Loop goal is complete | stop or open new loop |
| repeat | Same question needs another bounded pass | create narrower loop |
| prune | Direction is not worth continuing | stop and record why |
| frozen | Evidence is historical, invalid, or superseded | do not reuse as formal proof |
| blocked | Required input/resource is missing | ask user or wait |
| needs_user | Multiple valid choices require priority decision | ask user |
| executable | Execution gates and approval exist | main agent may run operation |

## Failure Handling

- If the objective is unclear, stop and define it. Do not implement around it.
- If the acceptance gate is unclear, write the gate first.
- If a worker drifts beyond scope, reject the drift and keep any usable bounded
  piece only after review.
- If reviewer findings conflict, the main agent decides using the goal,
  architecture, and evidence rules.
- If verification fails because of the change, fix within the loop if bounded;
  otherwise freeze and create a new loop.
- If verification fails for unrelated pre-existing reasons, record the evidence
  and do not claim full pass.
- If remote artifacts are missing, unstable, or provenance-drifting, freeze the
  run as incomplete evidence.
- If an active strict queue exists, prefer status, fetch, inventory, and report
  work over replacement submission.
- If a task starts producing new goals, stop and split them.

## What Counts As Done

General loop done:

- objective satisfied or explicitly blocked;
- acceptance criteria evaluated;
- verification result recorded;
- changed files or artifacts listed;
- P0/P1 review findings resolved or recorded as blockers;
- next decision state named.

Docs/process done:

- `AGENTS.md` remains concise;
- detailed rules live in `docs/`;
- cross-links point to the authoritative detailed doc;
- `git diff --check` passes.

Code done:

- behavior change is scoped;
- schema and metric changes have targeted tests;
- baseline semantics are preserved;
- no unrelated refactor or formatting churn;
- narrow validation passes or failure is explained.

Experiment done:

- manifest and evidence class are explicit;
- source/split/eval plan are fixed;
- artifacts pass inventory/report gates;
- conclusion states promote, repeat, prune, freeze, or defer;
- no paper-facing claim relies on provisional or stale evidence.

Infra done:

- changed layer is named;
- active/shadow authority is clear;
- contract or interface is documented;
- targeted test, parity report, dry run, or gate result exists;
- final state is accepted, blocked, shadow-only, eligible for human review, or
  executable.
