from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from torch import Tensor


class MazeBatch(TypedDict, total=False):
    H: Tensor
    adjacency: Tensor
    distance_field: Tensor
    distance_mask: Tensor
    start_idx: Tensor
    goal_idx: Tensor
    target_idx: Tensor
    label: Tensor
    baseline_label: Tensor
    raw_distance: Tensor
    open_mask: Tensor
    source_sign: Tensor
    distractor_sign: Tensor
    maze_id: Optional[List[str]]


TensorBatch = Dict[str, Tensor]
