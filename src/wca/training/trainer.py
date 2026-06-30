from __future__ import annotations

import csv
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.nn.parallel import DistributedDataParallel

from wca.config import Config
from wca.data.field.synthetic import configure_field_nodes
from wca.data.maze.generator import MazeSpec
from wca.data.maze.pools import ensure_heldout_maze_pool, ensure_maze_pool
from wca.data.maze.metrics import compute_metrics
from wca.models.diagnostics import diagnostics_metrics
from wca.models.field_wca import FieldTokenizerWCA
from wca.models.rws_nca import FullRecursiveWorldStateNCA
from wca.training.checkpointing import (
    best_checkpoint_improved,
    best_checkpoint_row_from_metrics,
    load_checkpoint,
    save_raw_model_state,
    save_training_state,
)
from wca.training.evaluator import evaluate, evaluate_field_horizon_stratified, evaluate_pool, make_batch
from wca.training.losses import compute_task_loss
from wca.training.prediction import predict_for_task
from wca.training.reports import (
    clean_metrics_dict,
    make_run_dir,
    print_final_report,
    update_best_metrics,
    write_summary,
)
from wca.schemas import TensorBatch
from wca.utils.device import resolve_device, sync_device
from wca.utils.distributed import (
    DistributedContext,
    barrier,
    broadcast_object,
    cleanup_distributed,
    init_distributed_from_env,
    reduce_metric_dict,
    reduce_scalar,
)
from wca.utils.precision import autocast_context, cuda_memory_metrics, make_grad_scaler
from wca.utils.seed import set_seed


FIELDNAMES = [
    "epoch",
    "loss",
    "mse",
    "mae",
    "sign_acc",
    "source_alignment",
    "distractor_alignment",
    "center_diversity",
    "collapse_score",
    "state_energy",
    "diag_energy",
    "mean_label",
    "mean_pred",
    "mean_raw_distance",
    "start_mse",
    "start_mae",
    "distance_mae_steps",
    "start_exact_acc",
    "path_success_rate",
    "path_optimal_rate",
    "path_length_ratio",
    "path_loop_rate",
    "goal_rank",
    "spurious_local_minima_count",
    "monotonic_descent_accuracy",
    "neighbor_order_accuracy",
    "field_energy_error",
    "field_relative_l2",
    "field_patch_count",
    "field_adjacency_density",
    "field_adjacency_degree",
    "field_input_visibility_density",
    "field_input_visibility_degree",
    "field_persistence_mse",
    "field_persistence_mae",
    "field_persistence_relative_l2",
    "field_delta_mse",
    "field_delta_mae",
    "field_mse_improvement_vs_persistence",
    "cuda_memory_allocated_mb",
    "cuda_memory_reserved_mb",
    "cuda_peak_memory_allocated_mb",
    "cuda_peak_memory_reserved_mb",
    "step_seconds",
    "eval_mse",
    "eval_mae",
    "eval_sign_acc",
    "eval_source_alignment",
    "eval_distractor_alignment",
    "eval_mean_label",
    "eval_mean_pred",
    "eval_mean_raw_distance",
    "eval_start_mse",
    "eval_start_mae",
    "eval_distance_mae_steps",
    "eval_start_exact_acc",
    "eval_path_success_rate",
    "eval_path_optimal_rate",
    "eval_path_length_ratio",
    "eval_path_loop_rate",
    "eval_goal_rank",
    "eval_spurious_local_minima_count",
    "eval_monotonic_descent_accuracy",
    "eval_neighbor_order_accuracy",
    "eval_field_energy_error",
    "eval_field_relative_l2",
    "eval_field_patch_count",
    "eval_field_adjacency_density",
    "eval_field_adjacency_degree",
    "eval_field_input_visibility_density",
    "eval_field_input_visibility_degree",
    "eval_field_persistence_mse",
    "eval_field_persistence_mae",
    "eval_field_persistence_relative_l2",
    "eval_field_delta_mse",
    "eval_field_delta_mae",
    "eval_field_mse_improvement_vs_persistence",
    "eval_center_diversity",
    "eval_collapse_score",
    "eval_state_energy",
    "eval_diag_energy",
    "field_core_executed",
    "eval_field_core_executed",
]

_WEATHERBENCH2_VARIABLE_NAMES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
_PER_VARIABLE_FIELDNAMES = []
for variable_name in _WEATHERBENCH2_VARIABLE_NAMES:
    _PER_VARIABLE_FIELDNAMES.extend(
        [
            f"field_mse_{variable_name}",
            f"field_mae_{variable_name}",
            f"field_persistence_mse_{variable_name}",
            f"field_persistence_mae_{variable_name}",
            f"field_mse_improvement_vs_persistence_{variable_name}",
        ]
    )
FIELDNAMES.extend(_PER_VARIABLE_FIELDNAMES)
FIELDNAMES.extend([f"eval_{name}" for name in _PER_VARIABLE_FIELDNAMES])
FIELDNAMES.extend(
    [
        "eval_field_horizon_stratified_score",
        "eval_field_horizon_stratified_horizon_count",
        "eval_field_horizon_stratified_valid_count",
    ]
)
for horizon in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48):
    FIELDNAMES.extend(
        [
            f"eval_h{horizon}_mse",
            f"eval_h{horizon}_field_persistence_mse",
            f"eval_h{horizon}_field_relative_mse",
            f"eval_h{horizon}_field_mse_improvement_vs_persistence",
            f"eval_h{horizon}_field_relative_l2",
        ]
    )

_EVAL_FIELDNAMES = [name.removeprefix("eval_") for name in FIELDNAMES if name.startswith("eval_")]
FIELDNAMES.extend(
    [f"{prefix}_{name}" for prefix in ("seen", "heldout") for name in _EVAL_FIELDNAMES]
)


def _model_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _uses_learnable_field_tokenizer(cfg: Config) -> bool:
    return cfg.task == "field" and str(getattr(cfg, "field_tokenizer", "patch_mean")) != "patch_mean"


def _forward_task_prediction(
    model: torch.nn.Module,
    module: torch.nn.Module,
    cfg: Config,
    batch: TensorBatch,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(module, FieldTokenizerWCA):
        return model(batch, cfg.outer_steps)
    H_final, diagnostics = model(
        batch["H"],
        batch["adjacency"],
        cfg.outer_steps,
        input_visibility=batch.get("input_visibility"),
        input_visibility_channels=batch.get("input_visibility_channels"),
    )
    prediction = predict_for_task(module, cfg, H_final, batch)
    return prediction, diagnostics


def _model_report_name_and_details(module: torch.nn.Module) -> tuple[str, Dict[str, object]]:
    if isinstance(module, FieldTokenizerWCA):
        core_executed_by_config = (not module.tokenizer_only) and int(getattr(module.cfg, "outer_steps", 0)) > 0
        if module.tokenizer_only:
            suffix = "tokenizer-only"
        elif not core_executed_by_config:
            suffix = "tokenizer-bypass-o0"
        else:
            suffix = "WCA"
        details = module.parameter_breakdown()
        details["field_tokenizer_only"] = module.tokenizer_only
        details["wca_core_executed_by_config"] = core_executed_by_config
        return f"{module.tokenizer_name}-{suffix}", details
    return "FullRecursiveWorldStateNCA-heavy-dense", {}


def _default_pool_path(cfg: Config) -> str:
    return f"artifacts/maze_pools/{cfg.maze_mode}_{cfg.grid_size}x{cfg.grid_size}_seed{cfg.seed}_n{cfg.maze_pool_size}.json"


def _default_heldout_pool_path(cfg: Config) -> str:
    return (
        f"artifacts/maze_pools/{cfg.maze_mode}_{cfg.grid_size}x{cfg.grid_size}_"
        f"seed{cfg.seed}_heldout{cfg.heldout_seed_offset}_n{cfg.heldout_pool_size}.json"
    )


def prepare_maze_pool(cfg: Config, ctx: DistributedContext) -> List[MazeSpec]:
    if cfg.task != "maze" or cfg.maze_mode != "structured-random" or cfg.maze_pool_size <= 0:
        return []
    if not cfg.pool_path:
        cfg.pool_path = _default_pool_path(cfg)
    if ctx.is_rank0:
        ensure_maze_pool(cfg)
    barrier(ctx)
    return ensure_maze_pool(cfg)


def prepare_heldout_maze_pool(cfg: Config, ctx: DistributedContext, train_pool: List[MazeSpec]) -> List[MazeSpec]:
    if (
        cfg.task != "maze"
        or cfg.maze_mode != "structured-random"
        or not cfg.evaluate_heldout
        or (cfg.heldout_pool_size <= 0 and not cfg.heldout_pool_path)
    ):
        return []
    if not cfg.heldout_pool_path:
        cfg.heldout_pool_path = _default_heldout_pool_path(cfg)
    if ctx.is_rank0:
        ensure_heldout_maze_pool(cfg, train_pool)
    barrier(ctx)
    return ensure_heldout_maze_pool(cfg, train_pool)


def _build_model(cfg: Config, device: torch.device, ctx: DistributedContext) -> torch.nn.Module:
    if _uses_learnable_field_tokenizer(cfg):
        model = FieldTokenizerWCA(cfg).to(device)
    else:
        model = FullRecursiveWorldStateNCA(
            n_nodes=cfg.n_nodes,
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.edge_dim,
            inner_steps=cfg.inner_steps,
            pair_chunk_size=cfg.pair_chunk_size,
            output_dim=int(getattr(cfg, "field_output_dim", 1)) if cfg.task == "field" else 1,
            activation_checkpoint_inner=cfg.activation_checkpoint_inner,
        ).to(device)
    if ctx.enabled:
        if device.type == "cuda":
            return DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
        return DistributedDataParallel(model)
    return model


def _trace(ctx: DistributedContext, message: str) -> None:
    if os.environ.get("WCA_TRACE_RANKS") == "1":
        print(f"[rank{ctx.rank}] {message}", flush=True)


def smoke_test_shapes(cfg: Config) -> None:
    ctx = init_distributed_from_env()
    try:
        device = resolve_device(cfg.device, local_rank=ctx.local_rank, distributed=ctx.enabled)
        if cfg.task == "maze":
            cfg.n_nodes = cfg.grid_size * cfg.grid_size
        if cfg.task == "field":
            configure_field_nodes(cfg)
        set_seed(cfg.seed + ctx.rank)
        maze_pool = prepare_maze_pool(cfg, ctx)
        model = _build_model(cfg, device, ctx)
        module = _model_module(model)
        batch = make_batch(cfg, device, maze_pool=maze_pool)
        H = batch["H"]
        adjacency = batch["adjacency"]
        expected = (cfg.batch_size, cfg.n_nodes, cfg.n_nodes, cfg.hidden_dim)
        if isinstance(module, FieldTokenizerWCA):
            prediction, diagnostics = _forward_task_prediction(model, module, cfg, batch)
            H_for_shape = diagnostics["field_H"]
            L = diagnostics.get("last_local_worlds")
        else:
            L = module.project_full_world(
                H,
                input_visibility=batch.get("input_visibility"),
                input_visibility_channels=batch.get("input_visibility_channels"),
            )
            H_final, diagnostics = model(
                H,
                adjacency,
                cfg.outer_steps,
                input_visibility=batch.get("input_visibility"),
                input_visibility_channels=batch.get("input_visibility_channels"),
            )
            H_for_shape = H_final
            prediction = predict_for_task(module, cfg, H_final, batch)
        if L is not None and tuple(L.shape) != expected:
            raise AssertionError(f"Expected local worlds shape {expected}, got {tuple(L.shape)}")
        if tuple(H_for_shape.shape) != (cfg.batch_size, cfg.n_nodes, cfg.hidden_dim):
            raise AssertionError(f"Unexpected WCA H shape: {tuple(H_for_shape.shape)}")
        if "last_local_worlds" not in diagnostics and not (
            isinstance(module, FieldTokenizerWCA)
            and (module.tokenizer_only or cfg.outer_steps <= 0)
        ):
            raise AssertionError("Missing last_local_worlds diagnostics.")
        expected_prediction = tuple(batch["distance_field"].shape if cfg.task == "maze" else batch["label"].shape)
        if tuple(prediction.shape) != expected_prediction:
            raise AssertionError(f"Expected prediction shape {expected_prediction}, got {tuple(prediction.shape)}")
        loss = compute_task_loss(cfg.task, prediction, batch, cfg)
        if not torch.isfinite(loss):
            raise AssertionError(f"Smoke loss is not finite: {float(loss.detach().cpu().item())}")
        if ctx.is_rank0:
            print(
                "Shape smoke test passed: "
                f"H={tuple(H_for_shape.shape)}, L={tuple(L.shape) if L is not None else 'bypass'}, "
                f"prediction={tuple(prediction.shape)}, loss={float(loss.detach().cpu().item()):.6g}"
            )
    finally:
        cleanup_distributed(ctx)


def train(cfg: Config) -> Optional[Path]:
    ctx = init_distributed_from_env()
    run_dir: Optional[Path] = None
    try:
        device = resolve_device(cfg.device, local_rank=ctx.local_rank, distributed=ctx.enabled)
        cfg.device = str(device)
        if cfg.task == "maze":
            cfg.n_nodes = cfg.grid_size * cfg.grid_size
        if cfg.task == "field":
            configure_field_nodes(cfg)
        set_seed(cfg.seed + ctx.rank)
        if ctx.is_rank0:
            print(f"Using device: {device} | distributed={ctx.enabled} | world_size={ctx.world_size}")

        maze_pool = prepare_maze_pool(cfg, ctx)
        heldout_pool = prepare_heldout_maze_pool(cfg, ctx, maze_pool)
        if ctx.is_rank0:
            run_dir = make_run_dir(cfg.run_dir)
            (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        run_dir_str = broadcast_object(str(run_dir) if run_dir is not None else "", ctx)
        run_dir = Path(run_dir_str)

        model = _build_model(cfg, device, ctx)
        module = _model_module(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        scaler = make_grad_scaler(device, cfg.precision)
        grad_accum_steps = max(1, int(cfg.grad_accum_steps))
        checkpoint_state = None
        start_epoch = 1
        if cfg.checkpoint:
            checkpoint_state = load_checkpoint(
                cfg.checkpoint,
                module,
                optimizer=optimizer,
                reset_optimizer=cfg.reset_optimizer,
                map_location=device,
            )
            start_epoch = checkpoint_state.epoch + 1 if checkpoint_state.epoch > 0 else 1
            if not cfg.parent_checkpoint:
                cfg.parent_checkpoint = cfg.checkpoint
            if ctx.is_rank0:
                optimizer_status = (
                    "reset"
                    if cfg.reset_optimizer
                    else "restored"
                    if checkpoint_state.has_optimizer_state
                    else "not present"
                )
                print(
                    f"Loaded checkpoint: {cfg.checkpoint} | "
                    f"epoch={checkpoint_state.epoch} | optimizer={optimizer_status}"
                )
                if start_epoch > cfg.epochs:
                    print(
                        f"Checkpoint epoch {checkpoint_state.epoch} is already past cfg.epochs={cfg.epochs}; "
                        "no training epochs will run. cfg.epochs is the absolute final epoch, not extra epochs."
                    )

        csv_file = None
        writer = None
        if ctx.is_rank0:
            csv_file = (run_dir / "train_log.csv").open("w", newline="")
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()

        last_row: Dict[str, float] = {}
        best_metrics: Dict[str, float] = dict(checkpoint_state.best_metrics) if checkpoint_state else {}
        first_epoch: Dict[str, int] = dict(checkpoint_state.first_threshold_epochs) if checkpoint_state else {}
        best_checkpoint_row: Optional[Dict[str, float]] = (
            dict(checkpoint_state.best_checkpoint_row)
            if checkpoint_state and checkpoint_state.best_checkpoint_row is not None
            else best_checkpoint_row_from_metrics(best_metrics)
        )

        for epoch in range(start_epoch, cfg.epochs + 1):
            _trace(ctx, f"epoch {epoch} start")
            model.train()
            epoch_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            metric_sums: Dict[str, float] = {}
            diagnostics: Dict[str, torch.Tensor] = {}
            loss_total = 0.0
            for accum_idx in range(grad_accum_steps):
                _trace(ctx, "make_batch start")
                batch = make_batch(cfg, device, maze_pool=maze_pool)
                sync_context = (
                    model.no_sync()
                    if ctx.enabled and hasattr(model, "no_sync") and accum_idx < grad_accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    with autocast_context(device, cfg.precision):
                        _trace(ctx, "forward start")
                        prediction, diagnostics = _forward_task_prediction(model, module, cfg, batch)
                        loss = compute_task_loss(cfg.task, prediction, batch, cfg)
                        scaled_loss = loss / float(grad_accum_steps)
                    _trace(ctx, "backward start")
                    scaler.scale(scaled_loss).backward()
                loss_total += float(loss.detach().item())

                _trace(ctx, "metrics start")
                micro_metrics = compute_metrics(cfg.task, prediction.detach(), batch, cfg.grid_size)
                for key, value in micro_metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.max_grad_norm))
            _trace(ctx, "optimizer step start")
            scaler.step(optimizer)
            scaler.update()
            sync_device(device)
            step_seconds = time.perf_counter() - epoch_start

            metric_row = {key: value / float(grad_accum_steps) for key, value in metric_sums.items()}
            metric_row.update(diagnostics_metrics(diagnostics))
            metric_row.update(cuda_memory_metrics(device))
            _trace(ctx, "metric reduce start")
            metric_row = reduce_metric_dict(metric_row, device, ctx)
            _trace(ctx, "loss reduce start")
            loss_value = reduce_scalar(loss_total / float(grad_accum_steps), device, ctx, average=True)
            step_seconds = reduce_scalar(step_seconds, device, ctx, average=True)

            eval_row: Dict[str, float] = {}
            if epoch == 1 or epoch % cfg.log_every == 0 or epoch == cfg.epochs:
                _trace(ctx, "eval start")
                eval_row = evaluate(model, cfg, device, ctx=ctx, maze_pool=maze_pool)
                if cfg.task == "field" and cfg.checkpoint_score == "field_horizon_stratified":
                    eval_row.update(
                        evaluate_field_horizon_stratified(model, cfg, device, ctx=ctx, maze_pool=maze_pool)
                    )
                if heldout_pool:
                    eval_row.update(evaluate_pool(model, cfg, device, prefix="seen", ctx=ctx, maze_pool=maze_pool))
                    eval_row.update(
                        evaluate_pool(model, cfg, device, prefix="heldout", ctx=ctx, maze_pool=heldout_pool)
                    )
                _trace(ctx, "eval done")

            row = {
                "epoch": float(epoch),
                "loss": loss_value,
                "step_seconds": step_seconds,
                **metric_row,
                **eval_row,
            }
            for name in FIELDNAMES:
                row.setdefault(name, float("nan"))
            last_row = row

            if ctx.is_rank0:
                assert writer is not None
                writer.writerow(row)
                update_best_metrics(best_metrics, first_epoch, row, epoch)
                if best_checkpoint_improved(best_checkpoint_row, row):
                    best_checkpoint_row = dict(row)
                    save_raw_model_state(run_dir / "best_model.pt", module)
                    save_training_state(
                        run_dir / "best_training_state.pt",
                        module,
                        optimizer,
                        epoch=epoch,
                        best_metrics=best_metrics,
                        first_threshold_epochs=first_epoch,
                        best_checkpoint_row=best_checkpoint_row,
                    )
                if epoch == 1 or epoch % cfg.log_every == 0:
                    print(
                        f"epoch={epoch:04d} loss={row['loss']:.4f} mae={row['mae']:.4f} "
                        f"eval_mae={row.get('eval_mae', float('nan')):.4f} "
                        f"path_ok={row.get('eval_path_success_rate', float('nan')):.3f} "
                        f"path_opt={row.get('eval_path_optimal_rate', float('nan')):.3f} "
                        f"goal_rank={row.get('eval_goal_rank', float('nan')):.3f} "
                        f"step={row['step_seconds']:.3f}s"
                    )

        if ctx.is_rank0:
            assert run_dir is not None
            final_epoch = int(last_row.get("epoch", max(start_epoch - 1, 0)))
            save_raw_model_state(run_dir / "model.pt", module)
            save_training_state(
                run_dir / "training_state.pt",
                module,
                optimizer,
                epoch=final_epoch,
                best_metrics=best_metrics,
                first_threshold_epochs=first_epoch,
                best_checkpoint_row=best_checkpoint_row,
            )
            final_metrics = clean_metrics_dict(last_row)
            best_metrics_clean = clean_metrics_dict(best_metrics)
            model_name, model_details = _model_report_name_and_details(module)
            write_summary(cfg, run_dir, final_metrics, best_metrics_clean, first_epoch, model_name, model_details)
            print(f"\nSaved run to: {run_dir}")
            print_final_report(cfg, run_dir, final_metrics, best_metrics_clean)
            if csv_file is not None:
                csv_file.close()
        return run_dir if ctx.is_rank0 else None
    finally:
        cleanup_distributed(ctx)
