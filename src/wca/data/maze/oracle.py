from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

GRID_NEIGHBORS_CACHE: Dict[int, List[List[int]]] = {}


def get_grid_neighbors(grid_size: int) -> List[List[int]]:
    if grid_size in GRID_NEIGHBORS_CACHE:
        return GRID_NEIGHBORS_CACHE[grid_size]
    n_nodes = grid_size * grid_size
    neighbors: List[List[int]] = [[] for _ in range(n_nodes)]
    for r in range(grid_size):
        for c in range(grid_size):
            i = r * grid_size + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    neighbors[i].append(nr * grid_size + nc)
    GRID_NEIGHBORS_CACHE[grid_size] = neighbors
    return neighbors


def bfs_shortest_distance(grid_size: int, walls: List[int], start: int, goal: int) -> Optional[int]:
    wall_set = set(walls)
    if start in wall_set or goal in wall_set:
        return None

    neighbors = get_grid_neighbors(grid_size)
    queue: deque[Tuple[int, int]] = deque([(start, 0)])
    seen = {start}
    while queue:
        node, dist = queue.popleft()
        if node == goal:
            return dist
        for nxt in neighbors[node]:
            if nxt in wall_set or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def bfs_distance_field(grid_size: int, walls: List[int], goal: int) -> List[float]:
    n_nodes = grid_size * grid_size
    wall_set = set(walls)
    distances = [-1.0 for _ in range(n_nodes)]
    if goal in wall_set:
        return distances

    neighbors = get_grid_neighbors(grid_size)
    queue: deque[Tuple[int, int]] = deque([(goal, 0)])
    seen = {goal}
    distances[goal] = 0.0

    while queue:
        node, dist = queue.popleft()
        for nxt in neighbors[node]:
            if nxt in wall_set or nxt in seen:
                continue
            seen.add(nxt)
            distances[nxt] = float(dist + 1)
            queue.append((nxt, dist + 1))
    return distances


def greedy_oracle_path(grid_size: int, walls: List[int], start: int, goal: int) -> List[int]:
    field = bfs_distance_field(grid_size, walls, goal)
    wall_set = set(walls)
    path = [start]
    current = start
    visited = {start}
    for _ in range(grid_size * grid_size * 2):
        if current == goal:
            break
        candidates = [node for node in get_grid_neighbors(grid_size)[current] if node not in wall_set and field[node] >= 0.0]
        if not candidates:
            break
        current = min(candidates, key=lambda node: field[node])
        path.append(current)
        if current in visited and current != goal:
            break
        visited.add(current)
    return path


def manhattan_distance(grid_size: int, start: int, goal: int) -> int:
    sr, sc = divmod(start, grid_size)
    gr, gc = divmod(goal, grid_size)
    return abs(sr - gr) + abs(sc - gc)
