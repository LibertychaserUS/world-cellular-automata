from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_summary(run_dir: str | Path) -> Dict[str, Any]:
    return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))


def compact_run_report(run_dir: str | Path) -> str:
    summary = load_summary(run_dir)
    final = summary.get("final_metrics", {})
    best = summary.get("best_metrics", {})
    return "\n".join(
        [
            f"run_dir={summary.get('run_dir', run_dir)}",
            f"model={summary.get('model', '')}",
            f"final_eval_mae={final.get('eval_mae', 'NA')}",
            f"final_path_ok={final.get('eval_path_success_rate', 'NA')}",
            f"final_path_opt={final.get('eval_path_optimal_rate', 'NA')}",
            f"best_path_ok={best.get('best_path_ok', 'NA')}",
            f"best_path_opt={best.get('best_path_opt', 'NA')}",
        ]
    )
