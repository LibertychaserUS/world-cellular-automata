#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def load_metric_values(path: Path, metric: str) -> list[float]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or metric not in reader.fieldnames:
            raise ValueError(f"Metric {metric!r} not found in {path}")
        values: list[float] = []
        for row in reader:
            value = _to_float(row.get(metric))
            if value is not None:
                values.append(value)
    return values


def evaluate_gate(values: list[float], *, min_value: float, min_rows: int, mode: str) -> dict[str, Any]:
    if len(values) < min_rows:
        return {
            "passed": False,
            "reason": f"Only {len(values)} numeric metric rows; required at least {min_rows}.",
            "values": values,
        }
    if mode == "all":
        passed = all(value >= min_value for value in values)
    elif mode == "any":
        passed = any(value >= min_value for value in values)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return {
        "passed": passed,
        "mode": mode,
        "min_value": min_value,
        "min_rows": min_rows,
        "values": values,
        "min_observed": min(values) if values else None,
        "max_observed": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail a queue when collected experiment metrics miss a gate.")
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--metric", default="final_eval_field_mse_improvement_vs_persistence")
    parser.add_argument("--min-value", type=float, default=0.0)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--mode", choices=["all", "any"], default="all")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    try:
        values = load_metric_values(args.results_csv, args.metric)
        payload = evaluate_gate(values, min_value=args.min_value, min_rows=args.min_rows, mode=args.mode)
    except (OSError, ValueError) as exc:
        payload = {"passed": False, "reason": str(exc), "values": []}

    payload.update({"results_csv": args.results_csv.as_posix(), "metric": args.metric})
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    if not payload.get("passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
