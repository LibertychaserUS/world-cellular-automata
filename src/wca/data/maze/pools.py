from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from wca.config import Config
from wca.data.maze.generator import MazeSpec, generate_structured_maze, maze_detour_gap

STRUCTURED_MAZE_POOL_CACHE: Dict[Tuple[int, float, int, int, int, int, int], List[MazeSpec]] = {}

MazeSignature = Tuple[Tuple[int, ...], int, int, int]


def structured_maze_worker(args: Tuple[Config, int]) -> MazeSpec:
    cfg, seed = args
    rng = random.Random(seed)
    return generate_structured_maze(cfg, rng)


def generate_structured_maze_pool(cfg: Config) -> List[MazeSpec]:
    if cfg.maze_pool_size <= 0:
        return []

    cache_key = (
        cfg.grid_size,
        float(cfg.wall_prob),
        int(cfg.seed),
        int(cfg.maze_pool_size),
        int(cfg.min_bfs_distance),
        int(cfg.min_detour_gap),
        int(cfg.max_generation_attempts),
    )
    if cache_key in STRUCTURED_MAZE_POOL_CACHE:
        return STRUCTURED_MAZE_POOL_CACHE[cache_key]

    start_time = time.perf_counter()
    mazes: List[MazeSpec] = []

    if cfg.maze_pool_workers > 1:
        worker_args = [(cfg, cfg.seed + 10000 + i) for i in range(cfg.maze_pool_size)]
        with ProcessPoolExecutor(max_workers=cfg.maze_pool_workers) as executor:
            futures = [executor.submit(structured_maze_worker, arg) for arg in worker_args]
            for future in as_completed(futures):
                mazes.append(future.result())
    else:
        rng = random.Random(cfg.seed + 10000)
        for _ in range(1, cfg.maze_pool_size + 1):
            mazes.append(generate_structured_maze(cfg, rng))

    STRUCTURED_MAZE_POOL_CACHE[cache_key] = mazes
    elapsed = time.perf_counter() - start_time
    distances = [item[3] for item in mazes]
    gaps = [maze_detour_gap(cfg.grid_size, item[1], item[2], item[3]) for item in mazes]
    wall_counts = [len(item[0]) for item in mazes]
    if mazes:
        print(
            f"Structured maze pool ready: count={len(mazes)} elapsed={elapsed:.2f}s "
            f"dist[min/max/avg]={min(distances)}/{max(distances)}/{sum(distances) / len(distances):.2f} "
            f"gap[min/max/avg]={min(gaps)}/{max(gaps)}/{sum(gaps) / len(gaps):.2f} "
            f"walls[avg]={sum(wall_counts) / len(wall_counts):.2f}"
        )
    return mazes


def maze_signature(maze: MazeSpec) -> MazeSignature:
    walls, start, goal, dist = maze
    return (tuple(sorted(int(wall) for wall in walls)), int(start), int(goal), int(dist))


def maze_signatures(mazes: Iterable[MazeSpec]) -> Set[MazeSignature]:
    return {maze_signature(maze) for maze in mazes}


def assert_disjoint_maze_pools(train_pool: List[MazeSpec], heldout_pool: List[MazeSpec]) -> None:
    overlap = maze_signatures(train_pool).intersection(maze_signatures(heldout_pool))
    if overlap:
        sample = sorted(overlap)[0]
        raise ValueError(
            "Held-out maze pool overlaps the train pool by maze tuple: "
            f"walls={list(sample[0])} start={sample[1]} goal={sample[2]} distance={sample[3]}"
        )


def _heldout_generation_attempt_limit(cfg: Config) -> int:
    return max(int(cfg.max_generation_attempts), max(1, int(cfg.heldout_pool_size)) * 50)


def generate_heldout_maze_pool(cfg: Config, train_pool: List[MazeSpec]) -> List[MazeSpec]:
    if cfg.heldout_pool_size <= 0:
        return []

    blocked = maze_signatures(train_pool)
    heldout: List[MazeSpec] = []
    heldout_signatures: Set[MazeSignature] = set()
    rng = random.Random(int(cfg.seed) + int(cfg.heldout_seed_offset) + 10000)

    for _ in range(_heldout_generation_attempt_limit(cfg)):
        candidate = generate_structured_maze(cfg, rng)
        signature = maze_signature(candidate)
        if signature in blocked or signature in heldout_signatures:
            continue
        heldout.append(candidate)
        heldout_signatures.add(signature)
        if len(heldout) >= cfg.heldout_pool_size:
            return heldout

    raise ValueError(
        "Could not generate a disjoint held-out maze pool: "
        f"requested={cfg.heldout_pool_size} generated={len(heldout)} "
        f"train_size={len(train_pool)} seed={cfg.seed} heldout_seed_offset={cfg.heldout_seed_offset}"
    )


def save_maze_pool(mazes: List[MazeSpec], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "walls": walls,
            "start": start,
            "goal": goal,
            "distance": dist,
            "maze_id": f"maze_{index:06d}",
        }
        for index, (walls, start, goal, dist) in enumerate(mazes)
    ]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_maze_pool(path: str | Path) -> List[MazeSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        (list(item["walls"]), int(item["start"]), int(item["goal"]), int(item["distance"]))
        for item in payload
    ]


def ensure_train_heldout_pools(cfg: Config) -> tuple[List[MazeSpec], List[MazeSpec]]:
    train_pool = ensure_maze_pool(cfg)
    heldout_pool = ensure_heldout_maze_pool(cfg, train_pool)
    return train_pool, heldout_pool


def split_train_eval_pool(pool: List[MazeSpec], eval_fraction: float = 0.2) -> tuple[List[MazeSpec], List[MazeSpec]]:
    if not pool:
        return [], []
    split = max(1, int(len(pool) * (1.0 - eval_fraction)))
    return pool[:split], pool[split:]


def ensure_maze_pool(cfg: Config) -> List[MazeSpec]:
    if cfg.maze_pool_size <= 0:
        return []
    if cfg.pool_path and Path(cfg.pool_path).exists():
        return load_maze_pool(cfg.pool_path)
    mazes = generate_structured_maze_pool(cfg)
    if cfg.pool_path:
        save_maze_pool(mazes, cfg.pool_path)
    return mazes


def ensure_heldout_maze_pool(cfg: Config, train_pool: List[MazeSpec]) -> List[MazeSpec]:
    if not cfg.evaluate_heldout:
        return []
    if cfg.heldout_pool_size <= 0 and not cfg.heldout_pool_path:
        return []
    if cfg.heldout_pool_path and Path(cfg.heldout_pool_path).exists():
        heldout_pool = load_maze_pool(cfg.heldout_pool_path)
        assert_disjoint_maze_pools(train_pool, heldout_pool)
        return heldout_pool

    heldout_pool = generate_heldout_maze_pool(cfg, train_pool)
    assert_disjoint_maze_pools(train_pool, heldout_pool)
    if cfg.heldout_pool_path:
        save_maze_pool(heldout_pool, cfg.heldout_pool_path)
    return heldout_pool
