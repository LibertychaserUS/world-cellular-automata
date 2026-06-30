from __future__ import annotations

from typing import List


EGOCENTRIC_NORTH_CH = 0
EGOCENTRIC_SOUTH_CH = 1
EGOCENTRIC_WEST_CH = 2
EGOCENTRIC_EAST_CH = 3
EGOCENTRIC_TRAVERSABLE_CH = 4
EGOCENTRIC_START_CH = 5
EGOCENTRIC_GOAL_CH = 6
EGOCENTRIC_X_CH = 7
EGOCENTRIC_Y_CH = 8


def compute_egocentric_sensing(grid_size: int, walls_list: List[int], node_idx: int) -> List[float]:
    """Return normalized ray distances in 4 cardinal directions.

    This helper mirrors the low-reference v0.3 prototype idea. It is not used
    by the baseline data path.
    """
    if node_idx in walls_list:
        return [-1.0, -1.0, -1.0, -1.0]

    r, c = divmod(node_idx, grid_size)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    sensing = []
    max_sense = grid_size
    for dr, dc in directions:
        dist = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < grid_size and 0 <= nc < grid_size:
            if (nr * grid_size + nc) in walls_list:
                dist += 1
                break
            dist += 1
            nr += dr
            nc += dc
        sensing.append(1.0 - dist / max_sense if dist > 0 else 0.0)
    return sensing
