# WCA Architecture

## Baseline: Full Dense RWS-NCA

The baseline model is a recursive world-state neural cellular automaton.

## State Tensors

```text
H: [B, N, D]
L: [B, N, N, D]
dense pair interaction: semantically [B, N, N, N, D]
```

## Recursion

```text
H_t
  -> project_full_world(H_t)
  -> L_t
  -> evolve_local_worlds(L_t, adjacency)
  -> compose_world(L_t)
  -> H_{t+1}
```

Each node constructs a full local world. The local world is not merely the node's neighborhood. It is a complete internal representation of the world from that center's perspective.

## What WCA Is Not

The WCA baseline is not:

- a standard GNN;
- a transformer;
- an attention layer;
- simple graph diffusion;
- A @ H message passing.

## Variants

Variants may add:

- egocentric sensing;
- predictive feedback;
- surprise signals;
- memory;
- alternative readouts.

Variants must not overwrite the baseline.
