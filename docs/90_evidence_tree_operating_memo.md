# WCA Evidence Tree Operating Memo

Date: 2026-06-25

Status: active project-control memo. Use this document to recover the current
WCA research loop after context compaction.

## Core Problem

The project is no longer short of ideas. The real risk is that model theory,
experimental science, and engineering systems keep triggering each other until
the project never reaches a stable conclusion.

The three work lines are:

```text
1. Model theory: does recursive world-state computation have value?
2. Experimental science: are bench, baseline, metric, split, and statistics strict?
3. Engineering systems: are infra, queue, cache, GPU use, and artifacts reliable?
```

New findings are not necessarily contradictions. Most are missing guardrails or
missing attribution tests. The project should absorb them through a stable
evidence tree instead of reopening every branch.

## Four-Bucket Routing

Every new idea must be routed before execution:

```text
A. Mainline evidence
B. Necessary guardrail
C. System debt
D. Future branch
```

### A. Mainline Evidence

Mainline evidence answers one question:

```text
Does the WCA recursive world-state core provide reproducible, attributable,
and scalable advantage?
```

Current mainline only includes:

- PDEBench reaction-diffusion;
- V25 PatchMean recursion ladder;
- V25b learnable interface;
- V25c guardrails;
- token-equivalent baselines;
- N=256 scaling;
- rollout.

Other ideas must not interrupt the current mainline until promoted.

### B. Necessary Guardrail

A guardrail enters the current stage only if it prevents a plausible false
positive.

Examples:

- no leakage;
- trajectory split;
- fixed eval start indices;
- final vs best checkpoint separation;
- tokenizer-only controls;
- outer-steps ablation;
- decoder ablation;
- numeric equivalence gate;
- dtype contract.

Guardrails are not new research directions. They protect mainline claims.

### C. System Debt

System debt is infrastructure work that exists only to make evidence
reproducible, auditable, and cost-controlled.

Examples:

- remote fetch and artifact inventory;
- manifest control;
- GPU planner;
- cache and prefetch;
- numeric equivalence;
- TS control plane;
- CUDA kernel.

System debt must have an explicit endpoint:

- integrity endpoint: automatically decide whether a run can enter formal tables;
- throughput endpoint: choose batch/job/slot from profile evidence;
- runtime endpoint: prove faster execution with numeric equivalence.

If an infra change does not serve the current evidence tree, defer it.

### D. Future Branch

Future branches can be important, but they do not enter the current queue until
the present stage passes its gates.

Examples:

- unified physical latent;
- contact-rich embodied WCA;
- SNN or path-strengthening variants;
- RL or intrinsic reward;
- native cell-state interfaces;
- multiscale;
- sparse/hybrid;
- WeatherBench SOTA track;
- world-model or JEPA comparison.

Future branches belong in roadmap documents, not in the active training queue.

## Stage Plan

All experiments must declare their stage.

```text
Stage 1: Establish
  Prove WCA has a real signal.
  Current status: V25 provides positive evidence.

Stage 2: Attribute
  Prove the signal comes from WCA core, not tokenizer, decoder, checkpoint,
  data, metric, or baseline artifacts.
  Current status: V25b/V25c.

Stage 3: Scale
  Prove the signal improves with N, D, recursion, data, and compute.
  Next candidates: N=256, D=192/256, multi-seed.

Stage 4: Generalize
  Prove the signal is not limited to one PDEBench subtask.
  Next candidates: more PDE families, rollout, weather/physics transfer.
```

Experiments without a stage should not run.

## Decision Rule

For every new idea, ask:

```text
1. Does it directly verify the WCA core?
2. Does it prevent a current-mainline false positive?
3. Does it fix current evidence execution or auditability?
4. Is it interesting but future-facing?
```

Map answers to:

```text
1 -> A. Mainline evidence
2 -> B. Necessary guardrail
3 -> C. System debt
4 -> D. Future branch
```

Only A/B/C can enter near-term execution. D is documented and revisited after
the current stage completes.

## Current Execution Order

Do not open new research directions until this sequence is resolved:

1. Let V25b finish.
2. Use V25c guardrails to test whether the learnable interface is explaining
   the result rather than the WCA core.
3. Run token-equivalent baselines.
4. If WCA core contribution survives attribution, run N=256 / D scaling.
5. If scaling holds, run rollout and more PDE families.
6. Only then promote unified physical latent, world-model, contact-rich, or
   JEPA-comparison branches.

## Main-Agent Rule

The main agent should maintain:

- evidence tree;
- stage ownership;
- acceptance criteria;
- formal/provisional conclusion boundaries;
- integration review;
- final user-facing claim.

Subagents should be used for:

- bounded engineering tasks;
- experiment implementation;
- artifact inspection;
- literature/design scouting;
- spec and code review.

The main agent must not delegate final evidence claims. Subagent output must be
reconciled against artifacts, tests, and the active acceptance criteria.
