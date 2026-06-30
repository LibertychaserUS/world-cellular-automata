from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from wca.config import Config
from wca.data.maze.oracle import bfs_shortest_distance, manhattan_distance

MazeSpec = Tuple[List[int], int, int, int]

FIXED_MAZE_CACHE: Dict[Tuple[int, float, int], MazeSpec] = {}
FIXED_MAZE_SET_CACHE: Dict[Tuple[int, float, int, int, int, int], List[MazeSpec]] = {}


def maze_detour_gap(grid_size: int, start: int, goal: int, bfs_distance: int) -> int:
    return int(bfs_distance) - manhattan_distance(grid_size, start, goal)


def maze_passes_filters(cfg: Config, start: int, goal: int, dist: int) -> bool:
    min_dist = cfg.min_bfs_distance
    min_gap = cfg.min_detour_gap
    if min_dist <= 0:
        min_dist = max(1, 2 * (cfg.grid_size - 1))
    if dist < min_dist:
        return False
    if maze_detour_gap(cfg.grid_size, start, goal, dist) < min_gap:
        return False
    return True


def sample_start_goal_for_maze(cfg: Config, rng: random.Random) -> Tuple[int, int]:
    n_nodes = cfg.grid_size * cfg.grid_size
    min_manhattan = max(1, cfg.min_bfs_distance // 2)
    for _ in range(32):
        start = rng.randrange(n_nodes)
        goal = rng.randrange(n_nodes)
        if goal == start:
            continue
        if manhattan_distance(cfg.grid_size, start, goal) >= min_manhattan:
            return start, goal
    start = rng.randrange(n_nodes)
    goal = rng.randrange(n_nodes)
    while goal == start:
        goal = rng.randrange(n_nodes)
    return start, goal


def generate_candidate_maze(cfg: Config, rng: random.Random) -> Optional[MazeSpec]:
    n_nodes = cfg.grid_size * cfg.grid_size
    start, goal = sample_start_goal_for_maze(cfg, rng)
    walls: List[int] = []
    for node in range(n_nodes):
        if node in {start, goal}:
            continue
        if rng.random() < cfg.wall_prob:
            walls.append(node)
    dist = bfs_shortest_distance(cfg.grid_size, walls, start, goal)
    if dist is None or dist <= 0:
        return None
    return walls, start, goal, dist


def maze_score(cfg: Config, walls: List[int], start: int, goal: int, dist: int) -> float:
    n_nodes = cfg.grid_size * cfg.grid_size
    gap = maze_detour_gap(cfg.grid_size, start, goal, dist)
    wall_ratio = len(walls) / max(1, n_nodes)
    return float(dist) + 0.75 * float(gap) + 0.25 * wall_ratio


def generate_single_maze(grid_size: int, wall_prob: float) -> MazeSpec:
    n_nodes = grid_size * grid_size
    local_cfg = Config(task="maze", grid_size=grid_size, n_nodes=n_nodes, wall_prob=wall_prob)
    rng = random
    for _ in range(300):
        candidate = generate_candidate_maze(local_cfg, rng)  # type: ignore[arg-type]
        if candidate is not None:
            return candidate

    start = 0
    goal = n_nodes - 1
    dist = bfs_shortest_distance(grid_size, [], start, goal)
    if dist is None:
        raise RuntimeError("Empty grid should be reachable.")
    return [], start, goal, dist


def generate_structured_maze(cfg: Config, rng: Optional[random.Random] = None) -> MazeSpec:
    rng = rng or random
    best: Optional[MazeSpec] = None
    best_score = -1.0

    for _ in range(cfg.max_generation_attempts):
        candidate = generate_candidate_maze(cfg, rng)  # type: ignore[arg-type]
        if candidate is None:
            continue
        walls, start, goal, dist = candidate
        score = maze_score(cfg, walls, start, goal, dist)
        if score > best_score:
            best = candidate
            best_score = score
        if maze_passes_filters(cfg, start, goal, dist):
            return candidate

    if best is not None:
        return best
    return generate_single_maze(cfg.grid_size, cfg.wall_prob)


def generate_fixed_hard_maze(cfg: Config) -> MazeSpec:
    cache_key = (cfg.grid_size, float(cfg.wall_prob), int(cfg.seed))
    if cache_key in FIXED_MAZE_CACHE:
        return FIXED_MAZE_CACHE[cache_key]

    rng = random.Random(cfg.seed)
    n_nodes = cfg.grid_size * cfg.grid_size
    best: Optional[MazeSpec] = None
    best_score = -1.0

    for _ in range(2500):
        start = rng.randrange(n_nodes)
        goal = rng.randrange(n_nodes)
        while goal == start:
            goal = rng.randrange(n_nodes)

        walls: List[int] = []
        for node in range(n_nodes):
            if node in {start, goal}:
                continue
            if rng.random() < cfg.wall_prob:
                walls.append(node)

        dist = bfs_shortest_distance(cfg.grid_size, walls, start, goal)
        if dist is None or dist <= 0:
            continue

        wall_bonus = len(walls) / max(1, n_nodes)
        score = float(dist) + 0.25 * wall_bonus
        if score > best_score:
            best = (walls, start, goal, dist)
            best_score = score

    if best is None:
        best = generate_single_maze(cfg.grid_size, cfg.wall_prob)

    FIXED_MAZE_CACHE[cache_key] = best
    return best


def generate_fixed_maze_set(cfg: Config) -> List[MazeSpec]:
    cache_key = (
        cfg.grid_size,
        float(cfg.wall_prob),
        int(cfg.seed),
        int(cfg.fixed_set_size),
        int(cfg.min_bfs_distance),
        int(cfg.min_detour_gap),
    )
    if cache_key in FIXED_MAZE_SET_CACHE:
        return FIXED_MAZE_SET_CACHE[cache_key]

    rng = random.Random(cfg.seed + 999)
    maze_set: List[MazeSpec] = []
    seen_signatures = set()
    best_candidates: List[Tuple[float, MazeSpec]] = []
    max_attempts = max(cfg.max_generation_attempts, cfg.fixed_set_size * 500)

    for _ in range(1, max_attempts + 1):
        candidate = generate_candidate_maze(cfg, rng)
        if candidate is None:
            continue
        walls, start, goal, dist = candidate
        signature = (tuple(sorted(walls)), start, goal)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        score = maze_score(cfg, walls, start, goal, dist)
        best_candidates.append((score, candidate))

        if maze_passes_filters(cfg, start, goal, dist):
            maze_set.append(candidate)
            if len(maze_set) >= cfg.fixed_set_size:
                break

    if len(maze_set) < cfg.fixed_set_size and best_candidates:
        best_candidates.sort(key=lambda item: item[0], reverse=True)
        current_signatures = {(tuple(sorted(item[0])), item[1], item[2]) for item in maze_set}
        for _, candidate in best_candidates:
            if len(maze_set) >= cfg.fixed_set_size:
                break
            walls, start, goal, _ = candidate
            signature = (tuple(sorted(walls)), start, goal)
            if signature in current_signatures:
                continue
            current_signatures.add(signature)
            maze_set.append(candidate)

    if not maze_set:
        maze_set.append(generate_fixed_hard_maze(cfg))

    FIXED_MAZE_SET_CACHE[cache_key] = maze_set
    return maze_set
