from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from wca.config import Config


def make_run_dir(base: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def finite_float(value: float) -> Optional[float]:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def clean_metrics_dict(metrics: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, value in metrics.items():
        value_float = finite_float(float(value))
        if value_float is not None:
            cleaned[key] = value_float
    return cleaned


def update_best_metrics(best: Dict[str, float], first_epoch: Dict[str, int], row: Dict[str, float], epoch: int) -> None:
    eval_mae = finite_float(row.get("eval_mae", float("nan")))
    eval_mse = finite_float(row.get("eval_mse", float("nan")))
    path_ok = finite_float(row.get("eval_path_success_rate", float("nan")))
    path_opt = finite_float(row.get("eval_path_optimal_rate", float("nan")))
    exact = finite_float(row.get("eval_start_exact_acc", float("nan")))
    field_horizon_score = finite_float(row.get("eval_field_horizon_stratified_score", float("nan")))
    loss = finite_float(row.get("loss", float("nan")))

    if eval_mae is not None and eval_mae < best.get("best_eval_mae", float("inf")):
        best["best_eval_mae"] = eval_mae
        best["best_eval_mae_epoch"] = float(epoch)
    if eval_mse is not None and eval_mse < best.get("best_eval_mse", float("inf")):
        best["best_eval_mse"] = eval_mse
        best["best_eval_mse_epoch"] = float(epoch)
    if field_horizon_score is not None and field_horizon_score < best.get(
        "best_eval_field_horizon_stratified_score", float("inf")
    ):
        best["best_eval_field_horizon_stratified_score"] = field_horizon_score
        best["best_eval_field_horizon_stratified_score_epoch"] = float(epoch)
    if loss is not None and loss < best.get("best_loss", float("inf")):
        best["best_loss"] = loss
        best["best_loss_epoch"] = float(epoch)
    if path_ok is not None and path_ok > best.get("best_path_ok", float("-inf")):
        best["best_path_ok"] = path_ok
        best["best_path_ok_epoch"] = float(epoch)
    if path_opt is not None and path_opt > best.get("best_path_opt", float("-inf")):
        best["best_path_opt"] = path_opt
        best["best_path_opt_epoch"] = float(epoch)
    if exact is not None and exact > best.get("best_exact", float("-inf")):
        best["best_exact"] = exact
        best["best_exact_epoch"] = float(epoch)

    thresholds = [0.25, 0.50, 0.75, 1.00]
    for threshold in thresholds:
        if path_ok is not None and path_ok >= threshold:
            first_epoch.setdefault(f"first_path_ok_ge_{threshold:.2f}", epoch)
        if path_opt is not None and path_opt >= threshold:
            first_epoch.setdefault(f"first_path_opt_ge_{threshold:.2f}", epoch)
        if exact is not None and exact >= threshold:
            first_epoch.setdefault(f"first_exact_ge_{threshold:.2f}", epoch)


def write_summary(
    cfg: Config,
    run_dir: Path,
    final_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    first_epoch: Dict[str, int],
    model_name: str = "FullRecursiveWorldStateNCA-heavy-dense",
    model_details: Optional[Dict[str, Any]] = None,
) -> None:
    structural_invariant = (
        "local_worlds tensor has shape [B, N, N, D]; dense receiver/sender pairs are fully computed, "
        "optionally in sender chunks to reduce peak memory"
    )
    core_executed_by_config = bool(model_details.get("wca_core_executed_by_config", True)) if model_details else True
    if model_details and not core_executed_by_config:
        structural_invariant = (
            "field-interface guardrail; WCA core is not executed, no local_worlds tensor is produced, "
            "and this run is not evidence for recursive WCA dynamics"
        )
    summary = {
        "config": asdict(cfg),
        "final_metrics": clean_metrics_dict(final_metrics),
        "best_metrics": clean_metrics_dict(best_metrics),
        "first_threshold_epochs": first_epoch,
        "model": model_name,
        "structural_invariant": structural_invariant,
        "run_dir": str(run_dir),
    }
    if model_details:
        summary["model_details"] = model_details
    if cfg.checkpoint:
        summary["checkpoint"] = cfg.checkpoint
    if cfg.parent_checkpoint:
        summary["parent_checkpoint"] = cfg.parent_checkpoint
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_final_report(cfg: Config, run_dir: Path, final_metrics: Dict[str, float], best_metrics: Dict[str, float]) -> None:
    final_clean = clean_metrics_dict(final_metrics)
    best_clean = clean_metrics_dict(best_metrics)
    print("\n" + "=" * 88)
    print("FINAL WCA REPORT")
    print("=" * 88)
    print(
        f"mode={cfg.maze_mode} | task={cfg.task} | grid={cfg.grid_size}x{cfg.grid_size} | "
        f"epochs={cfg.epochs} | batch_size={cfg.batch_size}"
    )
    print(
        f"loss={final_clean.get('loss', float('nan')):.6f} | "
        f"eval_mae={final_clean.get('eval_mae', float('nan')):.6f} | "
        f"eval_mse={final_clean.get('eval_mse', float('nan')):.6f} | "
        f"path_ok={final_clean.get('eval_path_success_rate', float('nan')):.3f} | "
        f"path_opt={final_clean.get('eval_path_optimal_rate', float('nan')):.3f}"
    )
    print(
        f"best_eval_mae={best_clean.get('best_eval_mae', float('nan')):.6f} | "
        f"best_path_ok={best_clean.get('best_path_ok', float('nan')):.3f} | "
        f"best_path_opt={best_clean.get('best_path_opt', float('nan')):.3f}"
    )
    print(f"run_dir={run_dir}")
    print("=" * 88)
