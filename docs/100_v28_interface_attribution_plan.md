# V28 Interface Attribution Plan

## Status

- Bucket: B. Necessary guardrail / A. Mainline attribution evidence
- Stage: Attribute
- Claim id: `V28-interface-attribution`
- Scope: PDEBench reaction-diffusion, N=256, patch 8, mixed horizons `h1,h2,h4,h8`

## Why This Exists

V27 closed the strict N=256 token-matched comparison, but the result is not a clean indictment of the WCA core.

The V27 WCA candidates used the `patch_mean` field-token interface:

```text
raw field [B,T,C,128,128]
  -> split into 8x8 patches
  -> average each patch
  -> node tokens / H channels
  -> dense WCA core
  -> node-wise residual prediction
```

This interface is deliberately simple and protected, but it is lossy. It removes patch-local gradients, edges, and high-frequency local structure before WCA sees the state. V27 showed:

- PatchMean WCA has positive long-horizon signal at `h4/h8`.
- PatchMean WCA is weak or negative at `h1/h2`.
- Token-equivalent Conv baseline currently beats PatchMean WCA across horizons.

Therefore the next question is:

```text
Did V27 lose because the WCA core is weak, or because PatchMean is a poor field interface?
```

## Architecture Under Test

### Current PatchMean Interface

File-level location:

- `src/wca/models/field_wca.py`
- `src/wca/models/field_tokenizers.py`
- `scripts/train_field.py`

Current strict V27 WCA path:

```text
PDEBench cache
  -> field batch
  -> patch mean tokenization
  -> H [B,N,D]
  -> FullRecursiveWorldStateNCA
  -> field token prediction
```

PatchMean is not a learnable encoder. It is a fixed reduction.

### Learnable Interface Path

`FieldTokenizerWCA` keeps the protected WCA core, but replaces the fixed patch mean with a learnable patch-local encoder:

```text
field_input [B,T,C,H,W]
  -> ConvStemTokenizer or MLPStemTokenizer
  -> encoded_tokens [B,N,token_dim]
  -> H [B,N,hidden_dim]
  -> FullRecursiveWorldStateNCA
  -> FieldPatchTokenDecoder
  -> residual prediction
```

This is still token-level field prediction, not a full-resolution image decoder.

## Experiment Matrix

All runs use:

- dataset: PDEBench reaction-diffusion cache `artifacts/field_datasets/pdebench/v20/cache.pt`
- grid: `128x128`
- patch: `8x8`
- N: `256`
- horizons: `1,2,4,8`
- split unit: trajectory
- eval split: test
- eval seed: `2026062201`
- eval samples: `64`
- checkpoint policy: final primary, best diagnostic

### Protected References

1. V26c PatchMean WCA N256 candidates
2. V27 token MLP N256 baselines
3. V27 token Conv N256 baselines

### New V28 Main Candidates

1. MLP-stem WCA, `D=128`, `outer_steps=10`, `inner_steps=2`, seeds `16301,16302`
2. Conv-stem WCA, `D=128`, `outer_steps=10`, `inner_steps=2`, seeds `16301,16302`

### New V28 Negative Controls

1. MLP-stem tokenizer-only, no WCA core, seed `16301`
2. Conv-stem tokenizer-only, no WCA core, seed `16301`
3. MLP-stem outer0, WCA wrapper but no recursion, seed `16301`
4. Conv-stem outer0, WCA wrapper but no recursion, seed `16301`

## Success Criteria

V28 is not a SOTA claim. It is an attribution gate.

Promotion requires all of:

1. Sentinel hard failures are zero.
2. Full-core learnable-interface WCA beats its matching tokenizer-only and outer0 controls.
3. Full-core learnable-interface WCA closes a material fraction of the V27 token_conv gap, especially at `h1/h2`.
4. If Conv-stem WCA beats MLP-stem WCA, report that the local learnable interface matters.
5. If token_conv still beats learnable-interface WCA, report that the WCA core has not yet beaten the strongest token-equivalent baseline under this contract.

## Failure Criteria

Freeze or repeat instead of scale if:

1. tokenizer-only or outer0 matches full-core WCA;
2. sentinels fail;
3. improvements appear only in best checkpoint and disappear in final;
4. decoder or interface capacity explains the result better than WCA recursion;
5. V28 produces non-comparable eval indices, split, or horizon plan.

## Formal Interpretation

Possible outcomes:

```text
learnable WCA > token_conv and > controls
  -> WCA core likely has attributable value after interface bottleneck is fixed.

learnable WCA > controls but < token_conv
  -> WCA core contributes, but current WCA stack is not stronger than token Conv baseline.

learnable WCA ~= controls
  -> V25b/V28 gains are likely interface/decoder effects, not WCA recursion.

sentinel failure
  -> no formal model-quality claim.
```

## Next Action

Generate and submit:

```text
configs/control/v28/westb_pdebench_v28_interface_attribution_manifest.json
artifacts/control/westb_pdebench_v28_interface_attribution/generated_queue.json
```

The queue must pass local pre-cloud gate before remote submit.

## V28 Failure And V28b Repair

The first V28 submission failed before training produced any model-quality evidence.

Observed failure:

```text
job: train_wca_mlpstem_n256_fullcore_d128_o10_i2_seed16301
failure: CUDA out of memory during train_field.py smoke_test_shapes
GPU: RTX PRO 6000 class, 96GB
attempted model: N=256, D=128, outer_steps=10, inner_steps=2, full dense core
```

Interpretation:

This is not a negative WCA result. It is a capacity-boundary failure. The original V28 design copied the V25 deep-recursion setting into N=256 full dense WCA. That oversteps the current full dense implementation's memory envelope.

V28b repair:

```text
experiment_id: westb_pdebench_v28b_interface_attribution_capacity_safe
scope: same N=256 / patch=8 / split / horizon / eval plan
change: use capacity-matched WCA points already known to fit N=256 PatchMean
```

V28b full-core candidates:

```text
MLP-stem WCA D64 outer2 inner2 seeds 16401,16402
MLP-stem WCA D96 outer2 inner1 seeds 16401,16402
Conv-stem WCA D64 outer2 inner2 seeds 16401,16402
Conv-stem WCA D96 outer2 inner1 seeds 16401,16402
```

V28b controls:

```text
MLP-stem outer0 D96 seed 16401
MLP-stem tokenizer-only D96 seed 16401
Conv-stem outer0 D96 seed 16401
Conv-stem tokenizer-only D96 seed 16401
```

V28b tests the same attribution question under a feasible full dense capacity boundary. If V28b still fails on memory, the next step is not deeper full dense; it is either smaller D/N or the planned sparse/hybrid implementation track.
