# WCA Strict Experiment Route and Dense-to-Sparse Plan

Date: 2026-05-13

## Position

WCA is now treated as an early recursive world-field architecture, not a maze toy and not a production weather model.

The current evidence supports a narrow claim:

- WCA can form useful fields on small maze and field tasks.
- Residual prediction is necessary for field dynamics.
- Task-matched local support can outperform full support.
- Capacity and recursion depth have shown positive signals.

It does not yet support these claims:

- WCA is weather SOTA.
- WCA is a general world model.
- WCA is more efficient than tuned U-Net/FNO/Neural Operator baselines.

## Operating Rule

Every experiment must answer one question and must define:

- hypothesis;
- control group;
- success criteria;
- failure interpretation;
- next action.

No queue should exist only to "try more training".

## E0: Protocol Guardrails

Purpose: prevent false positives before spending H800 time.

Hard requirements:

- no target leakage into `H`;
- report persistence improvement on every field run;
- separate `best_mse` and `final_mse`;
- report `best_epoch` and final-minus-best behavior;
- include per-variable metrics for WeatherBench2;
- bundle `config.json`, `summary.json`, `train_log.csv`, `model.pt`, dataset audit, and queue summary.

## E1: V13 Horizon-Conditioned Weather Test

Question: did V12 mixed-horizon fail because horizon was not explicit?

Groups:

- V13 control mixed horizon;
- V13 fixed h4/h8 controls;
- V13 horizon-conditioned mixed;
- V13 conditioned + tendency mixed;
- V13 warm-start conditioned + tendency.

Success:

- conditioned final MSE beats control mixed final MSE by at least 5%;
- conditioned best MSE beats control mixed mean final MSE by at least 5%;
- conditioned/tendency runs remain positive against persistence by at least 0.20;
- final degradation from best is smaller than V12.

Action:

- if final improves: scale conditioned mixed;
- if only best improves: stabilize training and checkpoint selection;
- if fixed still wins: prefer fixed-horizon or curriculum;
- if all fail: stop mixed-horizon scaling.

Local tool:

```bash
python3 scripts/decide_weatherbench2_v13.py \
  artifacts/reports/weatherbench2_v13_gpu0_control_final/results.csv \
  artifacts/reports/weatherbench2_v13_gpu1_conditioned_final/results.csv \
  --output-dir artifacts/reports/weatherbench2_v13_decision
```

## E2: Matched Baseline Fairness Test

Question: is WCA actually better than strong matched baselines?

V14 queue:

```text
configs/queues/h800_weatherbench2_v14_matched_baseline_queue.json
```

Runs:

- ConvNet conditioned mixed;
- ConvNet conditioned + tendency mixed;
- FNO conditioned mixed;
- FNO conditioned + tendency mixed;
- U-Net conditioned mixed;
- U-Net conditioned + tendency mixed.

Fairness dimensions:

- same WeatherBench2 cache;
- same variables;
- same train/eval split;
- same mixed horizon choices;
- same residual/tendency interface where applicable.

Success:

- WCA conditioned/tendency beats at least one strong baseline;
- WCA is not more than 2x slower to best checkpoint;
- gains appear in more than one variable, not only one channel.

Failure:

- if U-Net/FNO wins cleanly, WCA remains an interesting research architecture but not a superior field predictor under this setup.

## E3: Autoregressive Rollout

Question: is WCA a stable world-field model or only a direct horizon predictor?

Train h1, evaluate:

- direct h1/h2/h4/h8;
- autoregressive h1 -> h2 -> h4 -> h8.

Metrics:

- rollout MSE over horizon;
- relative L2 over horizon;
- per-variable drift;
- energy/spectrum drift proxy;
- collapse/explosion checks.

Success:

- AR h8 MSE no worse than 1.5x direct h8;
- no variable explosion or collapse.

## E4: Maze Topology Stress

Question: is the learned field topologically safe?

Families:

- open field;
- snake;
- bottleneck;
- dead ends;
- near-goal blocked;
- near-start blocked;
- symmetric;
- hard detour.

Metrics:

- path success;
- path optimality;
- loop rate;
- goal rank;
- spurious local minima;
- monotonic descent accuracy;
- neighbor order accuracy.

8x8 success:

- heldout path optimality >= 0.80;
- loop rate <= 0.10;
- average goal rank <= 2;
- average spurious local minima <= 1.

## E5: Support / Sparsity / Dense Ablation

Question: is local support better only semantically, or also computationally?

Groups:

- full dense support;
- dense masked grid;
- true sparse grid;
- true sparse moore;
- hybrid local + global anchors;
- multiscale WCA.

Equivalence gate:

Before any sparse efficiency or scaling claim, `SupportSparseRWSNCA` must match the corresponding dense-masked local-world model under:

- same weights;
- same input `H`;
- same edge/support mask;
- same pair kernel;
- same outer/inner schedule;
- same readout.

Required checks:

- dense-masked grid vs true sparse grid output difference within declared floating-point tolerance;
- dense-masked moore vs true sparse moore output difference within declared floating-point tolerance;
- gradient equivalence or a documented reason gradient equivalence is not expected due to implementation order;
- no collapse into ordinary GNN over `H`.

If this gate fails, sparse WCA remains an implementation prototype and cannot support formal capability or efficiency claims.

Success for true sparse grid:

- metric no worse than dense masked grid by 5%;
- samples/sec improves by at least 2x;
- peak memory falls by at least 2x.

Important distinction:

`FullDenseRWSNCA` remains the small-scale reference model.
`SupportSparseRWSNCA` becomes the scale path only if it preserves local-world semantics.

Sparse WCA is not ordinary GNN message passing. The update still happens inside each local world:

```text
center c
receiver r
sender s in support(r)
L[c, r] receives from L[c, s]
```

The complexity target changes from:

```text
O(B * N^3 * D)
```

to:

```text
O(B * N^2 * K * D)
```

where `K` is the local support degree.

## E6: Scaling and Efficiency

Question: does WCA improve predictably with capacity and recursion?

Sweep:

- hidden_dim: 64, 96, 128, 192, 256;
- inner_steps: 1, 2, 3, 4;
- outer_steps: 4, 8, 12, 16;
- N: 64, 128, 256;
- support: grid, moore, hybrid, full reference.

Report:

- metric vs params;
- metric vs wall-clock;
- metric vs peak memory;
- time-to-best;
- final-minus-best.

## E7: Architecture Ablation

Question: which mechanism matters?

Groups:

- full WCA baseline;
- no local-world `L`;
- `L` without pair evolution;
- simple linear pair kernel;
- residual readout;
- horizon conditioning;
- tendency baseline;
- support-sparse kernel.

Each ablation must answer one mechanism question.

## Dense-to-Sparse Implementation Order

1. Preserve `FullDenseRWSNCA` as the reference baseline.
2. Implement `SupportSparseRWSNCA` with grid support and equivalence tests against dense masked grid.
3. Sweep speed/memory/quality at N=64, 128, 256.
4. Add hybrid global anchors only if sparse grid is too slow at long-range propagation.
5. Add multiscale WCA only after hybrid anchors show value.

Short-term systems work:

- bf16;
- activation checkpointing around inner local-world evolution;
- true support-sparse pair kernel;
- `torch.compile` where stable;
- only then FSDP/ZeRO.

Avoid rewriting the whole model in ordinary C++.
If kernel work becomes necessary, prefer a Triton/CUDA pair kernel after the pair interaction has been made more structured.

## Near-Term Queue Order

1. Finish V13 and run `scripts/decide_weatherbench2_v13.py`.
2. Run V14 matched conditioned baselines on GPU1.
3. Prepare V15 autoregressive rollout queue.
4. Prepare V16 dense-masked vs true-sparse equivalence and speed queue.

## Current Interpretation Rule

If V13 succeeds but V14 baselines beat WCA, the gain is likely from the task interface, not WCA.

If V13 succeeds and WCA beats V14 matched baselines, WCA becomes a stronger parameter-efficiency candidate.

If V13 fails, do not scale mixed-horizon weather. Return to fixed horizon, topology losses, rollout stability, and sparse kernel work.
