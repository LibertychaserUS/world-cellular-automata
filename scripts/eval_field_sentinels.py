#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.eval_weatherbench_horizons import (  # noqa: E402
    checkpoint_paths as wca_checkpoint_paths,
    discover_run_dirs,
    horizon_max_from_config,
    model_name_for_config,
)
from scripts.field_eval_plan import (  # noqa: E402
    DEFAULT_EVAL_SEED,
    attach_fixed_eval_plan,
    eval_plan_hash,
    fixed_eval_plan_for_horizons,
    horizon_eval_seed,
    start_indices_hash,
    write_fixed_eval_plan,
)
from scripts.train_field_baseline import (  # noqa: E402
    build_model as build_raw_field_baseline,
    field_prediction_baseline,
    model_input_from_batch,
)
from scripts.train_field_token_baseline import (  # noqa: E402
    build_model as build_token_baseline,
    input_dim_for_config,
    predict_tokens as predict_token_baseline,
)
from wca.config import Config  # noqa: E402
from wca.data.field.real_cache import SUPPORTED_REAL_FIELD_DATASETS  # noqa: E402
from wca.data.field.synthetic import configure_field_nodes, field_patch_shape, field_token_shape, parse_field_target_steps_choices, patchify_field  # noqa: E402
from wca.data.field.synthetic import field_horizon_features  # noqa: E402
from wca.models.field_wca import FieldTokenizerWCA  # noqa: E402
from wca.models.rws_nca import FullRecursiveWorldStateNCA  # noqa: E402
from wca.training.checkpointing import load_checkpoint  # noqa: E402
from wca.training.evaluator import make_batch  # noqa: E402
from wca.training.prediction import predict_for_task  # noqa: E402
from wca.utils.device import resolve_device  # noqa: E402
from wca.utils.precision import autocast_context  # noqa: E402
from wca.utils.seed import set_seed  # noqa: E402


CSV_FIELDS = [
    "source_run_dir",
    "checkpoint_kind",
    "checkpoint_path",
    "horizon",
    "model",
    "seed",
    "eval_plan_seed",
    "eval_horizon_seed",
    "eval_plan_hash",
    "eval_start_indices_hash",
    "eval_sample_count",
    "field_horizon_conditioning",
    "field_tendency_baseline",
    "baseline_mse",
    "label_leakage_max_abs_diff",
    "label_leakage_pass",
    "horizon_probe_horizon",
    "horizon_shuffle_max_abs_diff",
    "horizon_shuffle_required",
    "horizon_shuffle_pass",
    "input_shuffle_mse",
    "input_shuffle_mse_ratio",
    "input_shuffle_degraded",
]
SENTINEL_EVAL_PLAN_FILENAME = "sentinel_eval_plan.json"


ForwardFn = Callable[[nn.Module, Config, dict[str, Any]], Tensor]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_run_dir(run_dir: Path) -> Config:
    payload = _read_json(run_dir / "config.json")
    allowed = {field.name for field in fields(Config)}
    cfg = Config(**{key: value for key, value in payload.items() if key in allowed})
    if cfg.task != "field":
        raise ValueError(f"Expected field run config in {run_dir}, got task={cfg.task!r}")
    configure_field_nodes(cfg)
    return cfg


def clone_config(cfg: Config) -> Config:
    allowed = {field.name for field in fields(Config)}
    return Config(**{key: value for key, value in cfg.to_dict().items() if key in allowed})


def prepare_horizon_config(base_cfg: Config, horizon: int, *, eval_batches: int, device: str) -> Config:
    cfg = clone_config(base_cfg)
    cfg.field_horizon_max_steps = horizon_max_from_config(base_cfg, [horizon])
    cfg.field_target_steps_choices = ""
    cfg.field_target_steps = int(horizon)
    cfg.eval_batches = int(eval_batches)
    cfg.device = device
    configure_field_nodes(cfg)
    return cfg


def checkpoint_paths(run_dir: Path, cfg: Config, kinds: list[str]) -> list[tuple[str, Path]]:
    if str(getattr(cfg, "baseline_model", "")) in {"convnet", "fno", "unet", "token_mlp", "token_conv"}:
        mapping = {
            "final": run_dir / "final_model.pt",
            "best": run_dir / "best_model.pt",
            "model": run_dir / "model.pt",
        }
        return [(kind, mapping[kind]) for kind in kinds if kind in mapping and mapping[kind].exists()]
    return wca_checkpoint_paths(run_dir, [kind for kind in kinds if kind in {"final", "best"}])


def build_model_and_forward(cfg: Config, device: torch.device) -> tuple[nn.Module, ForwardFn, str]:
    baseline_model = str(getattr(cfg, "baseline_model", ""))
    if baseline_model in {"convnet", "fno", "unet"}:
        model = build_raw_field_baseline(
            baseline_model,
            int(getattr(cfg, "field_output_dim", 1)),
            int(getattr(cfg, "baseline_width", 0) or 64),
            int(getattr(cfg, "baseline_depth", 0) or 6),
            int(getattr(cfg, "fno_modes", 0) or 12),
            condition_channels=4 if bool(getattr(cfg, "field_horizon_conditioning", False)) else 0,
        ).to(device)
        return model, forward_raw_field_baseline, f"{baseline_model}-field-baseline"
    if baseline_model in {"token_mlp", "token_conv"}:
        model = build_token_baseline(
            baseline_model,
            input_dim_for_config(cfg),
            int(getattr(cfg, "field_output_dim", 1)),
            field_token_shape(cfg),
            int(getattr(cfg, "baseline_width", 0) or (128 if baseline_model == "token_mlp" else 64)),
            int(getattr(cfg, "baseline_depth", 0) or (3 if baseline_model == "token_mlp" else 4)),
        ).to(device)
        return model, forward_token_baseline, f"{baseline_model}-field-token-baseline"

    if str(getattr(cfg, "field_tokenizer", "patch_mean")) != "patch_mean":
        return FieldTokenizerWCA(cfg).to(device), forward_wca, model_name_for_config(cfg)

    model = FullRecursiveWorldStateNCA(
        n_nodes=cfg.n_nodes,
        hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim,
        inner_steps=cfg.inner_steps,
        pair_chunk_size=cfg.pair_chunk_size,
        output_dim=int(getattr(cfg, "field_output_dim", 1)),
        activation_checkpoint_inner=cfg.activation_checkpoint_inner,
    ).to(device)
    return model, forward_wca, model_name_for_config(cfg)


def forward_wca(model: nn.Module, cfg: Config, batch: dict[str, Any]) -> Tensor:
    module = model.module if hasattr(model, "module") else model
    if isinstance(module, FieldTokenizerWCA):
        prediction, _diagnostics = model(batch, cfg.outer_steps)
        return prediction
    input_visibility = batch.get("input_visibility")
    if input_visibility is None:
        H_final, _diagnostics = model(batch["H"], batch["adjacency"], cfg.outer_steps)
    else:
        H_final, _diagnostics = model(
            batch["H"],
            batch["adjacency"],
            cfg.outer_steps,
            input_visibility=input_visibility,
            input_visibility_channels=batch.get("input_visibility_channels"),
        )
    return predict_for_task(module, cfg, H_final, batch)


def forward_raw_field_baseline(model: nn.Module, cfg: Config, batch: dict[str, Any]) -> Tensor:
    raw_field = model(model_input_from_batch(batch, cfg))
    if bool(getattr(cfg, "field_residual_readout", False)):
        prediction_field = field_prediction_baseline(batch, cfg, raw_field) + float(
            getattr(cfg, "field_residual_scale", 1.0)
        ) * raw_field
    else:
        prediction_field = raw_field
    prediction = patchify_field(prediction_field, field_patch_shape(cfg))
    if prediction.shape[-1] == 1:
        prediction = prediction.squeeze(-1)
    return prediction


def forward_token_baseline(model: nn.Module, cfg: Config, batch: dict[str, Any]) -> Tensor:
    prediction, _target = predict_token_baseline(model, batch, cfg)
    return prediction


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if isinstance(value, Tensor) else value for key, value in batch.items()}


def replace_targets(batch: dict[str, Any]) -> dict[str, Any]:
    altered = _clone_batch(batch)
    if isinstance(altered.get("label"), Tensor):
        altered["label"] = altered["label"].flip(0) + 1234.5
    if isinstance(altered.get("field_target"), Tensor):
        altered["field_target"] = altered["field_target"].flip(0) - 987.25
    return altered


def replace_horizon(batch: dict[str, Any], horizon: int) -> dict[str, Any]:
    altered = _clone_batch(batch)
    current = altered.get("field_target_steps_actual")
    if isinstance(current, Tensor):
        altered["field_target_steps_actual"] = torch.full_like(current, int(horizon))
    return altered


def materialized_horizon_channel_start(cfg: Config, batch: dict[str, Any]) -> int:
    """Return the first H channel containing horizon features for patch-mean WCA.

    Patch-mean WCA receives horizon conditioning through materialized H channels
    created by the batch maker. Mutating only field_target_steps_actual after
    make_batch does not touch the actual model input for this model family.
    """
    field_input = batch.get("field_input")
    if not isinstance(field_input, Tensor):
        raise ValueError("materialized H horizon sentinel requires batch['field_input']")
    current_tokens = patchify_field(field_input[:, -1], field_patch_shape(cfg))
    token_channels = int(current_tokens.shape[-1])
    return token_channels * 2 if int(cfg.hidden_dim) >= token_channels * 2 else token_channels


def replace_materialized_horizon_features(batch: dict[str, Any], cfg: Config, horizon: int) -> dict[str, Any]:
    altered = replace_horizon(batch, horizon)
    if not bool(getattr(cfg, "field_horizon_conditioning", False)):
        return altered
    if str(getattr(cfg, "field_tokenizer", "patch_mean")) != "patch_mean":
        return altered
    if str(getattr(cfg, "baseline_model", "")) in {"convnet", "fno", "unet", "token_mlp", "token_conv"}:
        return altered
    H = altered.get("H")
    if not isinstance(H, Tensor):
        return altered
    start = materialized_horizon_channel_start(cfg, altered)
    if H.shape[-1] < start + 4:
        raise ValueError(
            "materialized H horizon sentinel requires four horizon channels. "
            f"hidden_dim={H.shape[-1]}, required={start + 4}."
        )
    features = field_horizon_features(cfg, int(horizon), H.device, H.dtype)
    altered["H"] = H.clone()
    altered["H"][:, :, start : start + 4] = features.view(1, 1, 4)
    return altered


def shuffle_field_input(batch: dict[str, Any]) -> dict[str, Any]:
    altered = _clone_batch(batch)
    field_input = altered.get("field_input")
    if not isinstance(field_input, Tensor):
        raise ValueError("input shuffle sentinel requires batch['field_input']")
    if field_input.shape[0] > 1:
        altered["field_input"] = field_input.roll(shifts=1, dims=0)
    else:
        altered["field_input"] = field_input.flip(-1)
    return altered


def mse(prediction: Tensor, batch: dict[str, Any]) -> float:
    label = batch["label"].to(device=prediction.device, dtype=prediction.dtype)
    return float((prediction - label).pow(2).mean().detach().cpu().item())


def max_abs_diff(left: Tensor, right: Tensor) -> float:
    return float((left - right).abs().max().detach().cpu().item())


def alternate_horizon(current: int, horizons: list[int]) -> int:
    for horizon in horizons:
        if int(horizon) != int(current):
            return int(horizon)
    return int(current) + 1


def effective_eval_samples(cfg: Config, requested_eval_samples: int, requested_eval_batch_size: int) -> int:
    if requested_eval_samples > 0:
        return int(requested_eval_samples)
    if cfg.field_dataset not in SUPPORTED_REAL_FIELD_DATASETS:
        return 0
    return int(requested_eval_batch_size or cfg.batch_size)


@torch.no_grad()
def run_sentinels_on_batch(
    model: nn.Module,
    cfg: Config,
    batch: dict[str, Any],
    *,
    forward_fn: ForwardFn,
    horizon_probe: int,
) -> dict[str, Any]:
    model.eval()
    prediction = forward_fn(model, cfg, batch)
    leaked_prediction = forward_fn(model, cfg, replace_targets(batch))
    horizon_cfg = clone_config(cfg)
    horizon_cfg.field_target_steps = int(horizon_probe)
    horizon_batch = replace_materialized_horizon_features(batch, horizon_cfg, horizon_probe)
    horizon_prediction = forward_fn(model, horizon_cfg, horizon_batch)
    shuffled_batch = shuffle_field_input(batch)
    shuffled_prediction = forward_fn(model, cfg, shuffled_batch)

    baseline_mse = mse(prediction, batch)
    shuffled_mse = mse(shuffled_prediction, batch)
    ratio = shuffled_mse / max(baseline_mse, 1e-12)
    label_diff = max_abs_diff(prediction, leaked_prediction)
    horizon_diff = max_abs_diff(prediction, horizon_prediction)
    horizon_required = horizon_sentinel_required(cfg)
    return {
        "baseline_mse": baseline_mse,
        "label_leakage_max_abs_diff": label_diff,
        "label_leakage_pass": label_diff <= 1e-8,
        "horizon_probe_horizon": int(horizon_probe),
        "horizon_shuffle_max_abs_diff": horizon_diff,
        "horizon_shuffle_required": horizon_required,
        "horizon_shuffle_pass": (not horizon_required) or horizon_diff > 0.0,
        "input_shuffle_mse": shuffled_mse,
        "input_shuffle_mse_ratio": ratio,
        "input_shuffle_degraded": ratio > 1.05,
    }


def horizon_sentinel_required(cfg: Config) -> bool:
    """Return whether horizon perturbation should change predictions.

    Mechanism-negative controls intentionally bypass the recursive WCA core or
    the learned dynamics. They remain useful in sentinel tables, but treating
    their horizon insensitivity as a hard failure confuses expected negative
    control behavior with leakage/eval failure.
    """
    if not bool(getattr(cfg, "field_horizon_conditioning", False)):
        return False
    if bool(getattr(cfg, "field_tokenizer_only", False)):
        return False
    if int(getattr(cfg, "outer_steps", 0) or 0) == 0 and str(getattr(cfg, "baseline_model", "")) not in {
        "convnet",
        "fno",
        "unet",
        "token_mlp",
        "token_conv",
    }:
        return False
    return True


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.9g}"
    return str(value)


def evaluate_run_checkpoint(
    run_dir: Path,
    checkpoint_kind: str,
    checkpoint_path: Path,
    horizons: list[int],
    *,
    eval_batches: int,
    device_name: str,
    eval_seed: int,
    eval_samples: int,
    eval_batch_size: int,
    field_split: str,
) -> list[dict[str, Any]]:
    base_cfg = config_from_run_dir(run_dir)
    device = resolve_device(device_name)
    set_seed(base_cfg.seed)
    model, forward_fn, model_name = build_model_and_forward(base_cfg, device)
    if str(getattr(base_cfg, "baseline_model", "")) in {"convnet", "fno", "unet", "token_mlp", "token_conv"}:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        load_checkpoint(checkpoint_path, model, map_location=device)

    planned_eval_samples = effective_eval_samples(base_cfg, eval_samples, eval_batch_size)
    starts_by_horizon = (
        fixed_eval_plan_for_horizons(base_cfg, horizons, eval_seed=eval_seed, eval_samples=planned_eval_samples, field_split=field_split)
        if planned_eval_samples > 0
        else {}
    )
    plan_hash = eval_plan_hash(
        horizons,
        eval_batches,
        eval_seed,
        eval_samples=planned_eval_samples,
        eval_batch_size=eval_batch_size,
        start_indices_by_horizon=starts_by_horizon,
        field_split=field_split,
    )
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        cfg = prepare_horizon_config(base_cfg, horizon, eval_batches=eval_batches, device=str(device))
        cfg._field_eval_split_override = field_split
        sample_count = 0
        start_hash = ""
        if planned_eval_samples > 0:
            starts = starts_by_horizon[int(horizon)]
            attach_fixed_eval_plan(cfg, starts, eval_batch_size=eval_batch_size or cfg.batch_size)
            sample_count = int(starts.numel())
            start_hash = start_indices_hash(starts)
        probe_horizon = alternate_horizon(horizon, horizons)
        with torch.random.fork_rng(devices=_fork_rng_devices(device)):
            torch.manual_seed(horizon_eval_seed(eval_seed, horizon))
            batch = make_batch(cfg, device, field_split=field_split)
        with torch.random.fork_rng(devices=_fork_rng_devices(device)):
            torch.manual_seed(horizon_eval_seed(eval_seed, horizon))
            with autocast_context(device, cfg.precision):
                sentinel = run_sentinels_on_batch(
                    model,
                    cfg,
                    batch,
                    forward_fn=forward_fn,
                    horizon_probe=probe_horizon,
                )
        rows.append(
            {
                "source_run_dir": run_dir.as_posix(),
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_path": checkpoint_path.as_posix(),
                "horizon": int(horizon),
                "model": model_name,
                "seed": cfg.seed,
                "eval_plan_seed": eval_seed,
                "eval_horizon_seed": horizon_eval_seed(eval_seed, horizon),
                "eval_plan_hash": plan_hash,
                "eval_start_indices_hash": start_hash,
                "eval_sample_count": sample_count,
                "field_horizon_conditioning": bool(getattr(cfg, "field_horizon_conditioning", False)),
                "field_tendency_baseline": bool(getattr(cfg, "field_tendency_baseline", False)),
                **sentinel,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key)) for key in CSV_FIELDS})


def write_sentinel_eval_plan(
    output_dir: Path,
    starts_by_horizon: dict[int, torch.Tensor],
    *,
    eval_seed: int,
    field_split: str,
) -> None:
    write_fixed_eval_plan(
        output_dir / SENTINEL_EVAL_PLAN_FILENAME,
        starts_by_horizon,
        eval_seed=eval_seed,
        field_split=field_split,
    )


def summarize(rows: list[dict[str, Any]], *, horizons: list[int], checkpoint_kinds: list[str], eval_seed: int) -> dict[str, Any]:
    failures = [
        {
            "source_run_dir": row["source_run_dir"],
            "checkpoint_kind": row["checkpoint_kind"],
            "horizon": row["horizon"],
            "label_leakage_pass": row["label_leakage_pass"],
            "horizon_shuffle_pass": row["horizon_shuffle_pass"],
        }
        for row in rows
        if not bool(row.get("label_leakage_pass")) or not bool(row.get("horizon_shuffle_pass"))
    ]
    return {
        "schema_version": 1,
        "claim_id": "V25e-eval-sentinel",
        "evidence_status": "provisional_guardrail",
        "rows": len(rows),
        "horizons": [int(horizon) for horizon in horizons],
        "checkpoint_kinds": checkpoint_kinds,
        "eval_seed": int(eval_seed),
        "hard_failures": failures,
        "hard_failure_count": len(failures),
        "notes": [
            "label and field_target are mutated only for leakage sentinels and are not model inputs",
            "horizon shuffle is a hard failure only when field_horizon_conditioning is true and prediction diff is zero",
            "input shuffle reports degradation ratio and flag without a formal hard threshold",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V25e field eval sentinels for leakage, horizon, and input behavior.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--horizons", default="1,2,4,8")
    parser.add_argument("--checkpoint-kinds", default="final,best")
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--field-split", default="eval", choices=["eval", "val", "test"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/v25e_eval_sentinels"))
    args = parser.parse_args()

    horizons = parse_field_target_steps_choices(args.horizons)
    kinds = [item.strip() for item in args.checkpoint_kinds.split(",") if item.strip()]
    if any(kind not in {"final", "best", "model"} for kind in kinds):
        raise SystemExit("--checkpoint-kinds may only contain final,best,model")
    run_dirs = discover_run_dirs(args.run_dirs)
    if not run_dirs:
        raise SystemExit("No run directories with config.json and model.pt found.")

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        cfg = config_from_run_dir(run_dir)
        for checkpoint_kind, checkpoint_path in checkpoint_paths(run_dir, cfg, kinds):
            rows.extend(
                evaluate_run_checkpoint(
                    run_dir,
                    checkpoint_kind,
                    checkpoint_path,
                    horizons,
                    eval_batches=args.eval_batches,
                    device_name=args.device,
                    eval_seed=args.eval_seed,
                    eval_samples=args.eval_samples,
                    eval_batch_size=args.eval_batch_size,
                    field_split=args.field_split,
                )
            )
    if not rows:
        raise SystemExit("No checkpoints were evaluated.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_cfg = config_from_run_dir(run_dirs[0])
    planned_eval_samples = effective_eval_samples(first_cfg, args.eval_samples, args.eval_batch_size)
    if planned_eval_samples > 0:
        starts_by_horizon = fixed_eval_plan_for_horizons(
            first_cfg,
            horizons,
            eval_seed=args.eval_seed,
            eval_samples=planned_eval_samples,
            field_split=args.field_split,
        )
        write_sentinel_eval_plan(
            args.output_dir,
            starts_by_horizon,
            eval_seed=args.eval_seed,
            field_split=args.field_split,
        )
    write_csv(args.output_dir / "sentinel_results.csv", rows)
    summary = summarize(rows, horizons=horizons, checkpoint_kinds=kinds, eval_seed=args.eval_seed)
    (args.output_dir / "sentinel_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "hard_failure_count": summary["hard_failure_count"], "output_dir": args.output_dir.as_posix()}, sort_keys=True))
    if summary["hard_failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
