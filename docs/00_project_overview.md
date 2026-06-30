# Project Overview

WCA means World Cellular Automata.

The current research baseline is Full Dense RWS-NCA, a recursive world-state neural cellular automaton. It constructs one full local world per node, evolves each local world through dense pair interactions, and recomposes the global state from local centers.

The first benchmark is grid-maze shortest-distance field learning. The model learns a scalar field over open cells. Functional performance is measured by greedy navigation over that predicted field, not by MSE alone.

Current priorities:

1. Preserve the v0.2-heavy baseline behavior.
2. Separate model, oracle, data, training, reporting, and visualization.
3. Make start and goal explicit batch fields.
4. Evaluate seen and held-out maze pools.
5. Keep DeepSeek v0.3 mechanisms as variants.
