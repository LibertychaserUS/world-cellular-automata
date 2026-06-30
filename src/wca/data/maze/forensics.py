from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from torch import Tensor

from wca.data.maze.metrics import _open_mask_from_batch, classify_path
from wca.data.maze.oracle import get_grid_neighbors
from wca.schemas import TensorBatch


def _as_float(value: Tensor | float) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _sample_open_nodes(batch: TensorBatch, sample_idx: int) -> List[int]:
    open_mask = _open_mask_from_batch(batch).detach().cpu()
    return [idx for idx, is_open in enumerate(open_mask[sample_idx].tolist()) if bool(is_open)]


def _global_min_nodes(prediction_field: Tensor, batch: TensorBatch, sample_idx: int) -> List[int]:
    open_nodes = _sample_open_nodes(batch, sample_idx)
    if not open_nodes:
        return []
    values = prediction_field[sample_idx].detach().cpu()
    min_value = min(float(values[node].item()) for node in open_nodes)
    return [node for node in open_nodes if abs(float(values[node].item()) - min_value) < 1e-8]


def _local_minima_nodes(prediction_field: Tensor, batch: TensorBatch, grid_size: int, sample_idx: int) -> List[int]:
    open_mask = _open_mask_from_batch(batch).detach().cpu()
    values = prediction_field[sample_idx].detach().cpu()
    goal = int(batch["goal_idx"][sample_idx].detach().cpu().item())
    minima: List[int] = []
    for node in _sample_open_nodes(batch, sample_idx):
        if node == goal:
            continue
        neighbors = [nxt for nxt in get_grid_neighbors(grid_size)[node] if bool(open_mask[sample_idx, nxt].item())]
        if neighbors and all(float(values[node].item()) <= float(values[nxt].item()) for nxt in neighbors):
            minima.append(node)
    return minima


def trace_greedy_path(prediction_field: Tensor, batch: TensorBatch, grid_size: int, sample_idx: int) -> Dict[str, Any]:
    """Trace one greedy descent path and classify why it succeeds or fails."""
    if prediction_field.ndim != 2:
        raise ValueError(f"Expected prediction_field [B,N], got {tuple(prediction_field.shape)}")

    values = prediction_field.detach().cpu()
    true_field = batch["distance_field"].detach().cpu()
    open_mask = _open_mask_from_batch(batch).detach().cpu()
    start = int(batch["start_idx"][sample_idx].detach().cpu().item())
    goal = int(batch["goal_idx"][sample_idx].detach().cpu().item())
    raw_distance = _as_float(batch["raw_distance"][sample_idx])

    current = start
    path = [current]
    visited = {current}
    steps: List[Dict[str, Any]] = []
    reason = "max_steps"
    max_steps = values.shape[1] * 2

    for _ in range(max_steps):
        if current == goal:
            reason = "reached_goal"
            break

        pred_value = float(values[sample_idx, current].item())
        true_value = float(true_field[sample_idx, current].item())
        candidates = [nxt for nxt in get_grid_neighbors(grid_size)[current] if bool(open_mask[sample_idx, nxt].item())]
        if not candidates:
            reason = "dead_end"
            steps.append(
                {
                    "node": current,
                    "pred": pred_value,
                    "true": true_value,
                    "neighbors": [],
                    "chosen": None,
                    "oracle_neighbors": [],
                }
            )
            break

        oracle_neighbors = [
            nxt for nxt in candidates if float(true_field[sample_idx, nxt].item()) < true_value
        ]
        chosen = min(candidates, key=lambda idx: float(values[sample_idx, idx].item()))
        chosen_pred = float(values[sample_idx, chosen].item())
        chosen_true = float(true_field[sample_idx, chosen].item())
        neighbor_rows = [
            {
                "node": nxt,
                "pred": float(values[sample_idx, nxt].item()),
                "true": float(true_field[sample_idx, nxt].item()),
                "is_oracle_descent": nxt in oracle_neighbors,
                "is_chosen": nxt == chosen,
            }
            for nxt in candidates
        ]

        steps.append(
            {
                "node": current,
                "pred": pred_value,
                "true": true_value,
                "neighbors": sorted(neighbor_rows, key=lambda item: item["pred"]),
                "chosen": {
                    "node": chosen,
                    "pred": chosen_pred,
                    "true": chosen_true,
                    "is_oracle_descent": chosen in oracle_neighbors,
                },
                "oracle_neighbors": oracle_neighbors,
            }
        )

        current = chosen
        path.append(current)
        if current in visited and current != goal:
            reason = "loop"
            break
        visited.add(current)

    if path[-1] == goal:
        reason = "reached_goal"

    reached, optimal, looped = classify_path(path, goal, raw_distance)
    global_min = _global_min_nodes(prediction_field, batch, sample_idx)
    local_minima = _local_minima_nodes(prediction_field, batch, grid_size, sample_idx)
    goal_value = float(values[sample_idx, goal].item())
    start_value = float(values[sample_idx, start].item())
    min_value = min(float(values[sample_idx, node].item()) for node in _sample_open_nodes(batch, sample_idx))

    if not reached and reason == "loop":
        failure_class = "loop"
    elif not reached and path[-1] in local_minima:
        failure_class = "local_minimum"
    elif not reached and goal not in global_min:
        failure_class = "wrong_global_minimum"
    elif reached and not optimal:
        failure_class = "suboptimal_success"
    elif reached:
        failure_class = "optimal_success" if optimal else "success"
    else:
        failure_class = reason

    first_wrong_step = None
    for index, step in enumerate(steps):
        chosen = step.get("chosen")
        if chosen is not None and not bool(chosen["is_oracle_descent"]):
            first_wrong_step = index
            break

    return {
        "sample_idx": sample_idx,
        "maze_id": batch.get("maze_id", [""])[sample_idx] if isinstance(batch.get("maze_id"), list) else "",
        "start_idx": start,
        "goal_idx": goal,
        "raw_distance": raw_distance,
        "path": path,
        "path_length": max(0, len(path) - 1),
        "reached": reached,
        "optimal": optimal,
        "looped": looped,
        "stop_reason": reason,
        "failure_class": failure_class,
        "first_wrong_step": first_wrong_step,
        "start_pred": start_value,
        "goal_pred": goal_value,
        "global_min_pred": min_value,
        "global_min_nodes": global_min,
        "goal_is_global_min": goal in global_min,
        "local_minima_count": len(local_minima),
        "local_minima_nodes": local_minima,
        "steps": steps,
    }


def summarize_traces(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(traces)
    if total == 0:
        return {"sample_count": 0}
    classes = Counter(str(trace["failure_class"]) for trace in traces)
    reached = sum(1 for trace in traces if bool(trace["reached"]))
    optimal = sum(1 for trace in traces if bool(trace["optimal"]))
    looped = sum(1 for trace in traces if bool(trace["looped"]))
    wrong_global_min = sum(1 for trace in traces if not bool(trace["goal_is_global_min"]))
    local_minima_total = sum(int(trace["local_minima_count"]) for trace in traces)
    return {
        "sample_count": total,
        "reached": reached,
        "optimal": optimal,
        "looped": looped,
        "path_success_rate": reached / total,
        "path_optimal_rate": optimal / total,
        "path_loop_rate": looped / total,
        "goal_not_global_min_rate": wrong_global_min / total,
        "avg_local_minima_count": local_minima_total / total,
        "failure_class_counts": dict(classes),
    }


def render_trace_markdown(summary: Dict[str, Any], traces: List[Dict[str, Any]], max_examples: int = 12) -> str:
    lines = [
        "# Maze Path Forensics",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if key == "failure_class_counts":
            continue
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Failure Classes", "", "| class | count |", "|---|---:|"])
    for key, value in sorted(summary.get("failure_class_counts", {}).items()):
        lines.append(f"| `{key}` | {value} |")

    selected = sorted(
        traces,
        key=lambda item: (
            bool(item["reached"]),
            bool(item["optimal"]),
            -int(item["local_minima_count"]),
            int(item["sample_idx"]),
        ),
    )[:max_examples]
    lines.extend(["", "## Example Traces", ""])
    for trace in selected:
        trace_label = trace.get("trace_idx", trace["sample_idx"])
        lines.extend(
            [
                f"### Trace {trace_label} - {trace['failure_class']}",
                "",
                f"- sample_idx: `{trace['sample_idx']}` pool_index: `{trace.get('pool_index', '')}`",
                f"- maze_id: `{trace.get('maze_id', '')}`",
                f"- start: `{trace['start_idx']}` goal: `{trace['goal_idx']}` raw_distance: `{trace['raw_distance']}`",
                f"- reached: `{trace['reached']}` optimal: `{trace['optimal']}` looped: `{trace['looped']}`",
                f"- path_length: `{trace['path_length']}` path: `{trace['path']}`",
                f"- start_pred: `{trace['start_pred']:.6f}` goal_pred: `{trace['goal_pred']:.6f}` global_min_pred: `{trace['global_min_pred']:.6f}`",
                f"- global_min_nodes: `{trace['global_min_nodes']}` goal_is_global_min: `{trace['goal_is_global_min']}`",
                f"- local_minima_count: `{trace['local_minima_count']}` local_minima_nodes: `{trace['local_minima_nodes']}`",
                f"- first_wrong_step: `{trace['first_wrong_step']}`",
                "",
            ]
        )
        for idx, step in enumerate(trace["steps"][:8]):
            chosen = step.get("chosen")
            lines.append(
                f"  - step {idx}: node `{step['node']}` pred `{step['pred']:.6f}` true `{step['true']:.6f}` "
                f"chosen `{chosen}` oracle_neighbors `{step['oracle_neighbors']}`"
            )
        lines.append("")
    return "\n".join(lines)
