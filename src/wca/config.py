from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable


@dataclass
class Config:
    seed: int = 42
    task: str = "source"
    n_nodes: int = 16
    grid_size: int = 5
    wall_prob: float = 0.22
    field_dataset: str = "synthetic_heat"
    field_grid_size: int = 32
    field_grid_height: int = 0
    field_grid_width: int = 0
    field_patch_size: int = 4
    field_patch_height: int = 0
    field_patch_width: int = 0
    field_input_steps: int = 2
    field_target_steps: int = 1
    field_target_steps_choices: str = ""
    field_horizon_max_steps: int = 0
    field_horizon_conditioning: bool = False
    field_tendency_baseline: bool = False
    field_tendency_scale: float = 1.0
    field_diffusion_rate: float = 0.18
    field_decay: float = 0.995
    field_residual_readout: bool = False
    field_residual_scale: float = 1.0
    field_adjacency_mode: str = "grid"
    field_input_scope: str = "global"
    field_data_path: str = ""
    field_output_dim: int = 1
    field_train_start: int = 0
    field_train_size: int = 0
    field_eval_start: int = 0
    field_eval_size: int = 0
    field_val_start: int = 0
    field_val_size: int = 0
    field_test_start: int = 0
    field_test_size: int = 0
    field_stride: int = 1
    field_variable: str = ""
    field_variables: str = ""
    field_tokenizer: str = "patch_mean"
    field_token_dim: int = 0
    field_tokenizer_width: int = 0
    field_decoder_width: int = 0
    field_tokenizer_only: bool = False
    field_baseline_scope: str = ""
    baseline_model: str = ""
    baseline_width: int = 0
    baseline_depth: int = 0
    fno_modes: int = 0
    maze_mode: str = "random"
    fixed_set_size: int = 8
    min_bfs_distance: int = 0
    min_detour_gap: int = 0
    max_generation_attempts: int = 3000
    maze_pool_size: int = 0
    maze_pool_workers: int = 0
    heldout_pool_path: str = ""
    heldout_pool_size: int = 0
    heldout_seed_offset: int = 500000
    evaluate_heldout: bool = False
    hidden_dim: int = 48
    edge_dim: int = 12
    batch_size: int = 16
    epochs: int = 500
    lr: float = 3e-4
    outer_steps: int = 8
    inner_steps: int = 3
    pair_chunk_size: int = 0
    activation_checkpoint_inner: bool = False
    precision: str = "fp32"
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    device: str = "auto"
    log_every: int = 10
    eval_batches: int = 8
    checkpoint_score: str = "auto"
    field_checkpoint_horizons: str = ""
    field_checkpoint_eval_batches: int = 0
    field_checkpoint_score_weights: str = ""
    field_checkpoint_score_metric: str = "relative_mse"
    field_checkpoint_seed_offset: int = 910000
    field_loss_weight: float = 1.0
    neighbor_rank_loss_weight: float = 0.0
    descent_margin_loss_weight: float = 0.0
    bellman_loss_weight: float = 0.0
    loss_margin: float = 0.01
    bellman_step_scale: float = 1.0
    visualize: bool = False
    visual_every: int = 5
    smoke_test_only: bool = False
    run_dir: str = "runs/rws_nca_full_heavy"
    pool_path: str = ""
    checkpoint: str = ""
    reset_optimizer: bool = False
    parent_checkpoint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: str | Path) -> Dict[str, Any]:
    """Load the flat YAML subset used by this project without adding PyYAML."""
    data: Dict[str, Any] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data


def load_config(path: str | Path) -> Config:
    data = load_simple_yaml(path)
    allowed = {field.name for field in fields(Config)}
    filtered = {key: value for key, value in data.items() if key in allowed}
    cfg = Config(**filtered)
    if cfg.task == "maze":
        cfg.n_nodes = cfg.grid_size * cfg.grid_size
    if cfg.task == "field":
        _configure_field_shape(cfg)
    return cfg


def apply_overrides(cfg: Config, overrides: argparse.Namespace, names: Iterable[str]) -> Config:
    for name in names:
        value = getattr(overrides, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if cfg.task == "maze":
        cfg.n_nodes = cfg.grid_size * cfg.grid_size
    if cfg.task == "field":
        _configure_field_shape(cfg)
    return cfg


def _configure_field_shape(cfg: Config) -> None:
    height = int(cfg.field_grid_height or cfg.field_grid_size)
    width = int(cfg.field_grid_width or cfg.field_grid_size)
    patch_height = int(cfg.field_patch_height or cfg.field_patch_size)
    patch_width = int(cfg.field_patch_width or cfg.field_patch_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"field grid height/width must be positive, got {height}x{width}")
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"field patch height/width must be positive, got {patch_height}x{patch_width}")
    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(
            f"field grid {height}x{width} must be divisible by patch {patch_height}x{patch_width}"
        )
    cfg.grid_size = height // patch_height
    cfg.n_nodes = (height // patch_height) * (width // patch_width)


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--task", type=str, default=None, choices=["source", "maze", "field"])
    parser.add_argument("--n-nodes", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--wall-prob", type=float, default=None)
    parser.add_argument("--field-dataset", type=str, default=None)
    parser.add_argument("--field-grid-size", type=int, default=None)
    parser.add_argument("--field-grid-height", type=int, default=None)
    parser.add_argument("--field-grid-width", type=int, default=None)
    parser.add_argument("--field-patch-size", type=int, default=None)
    parser.add_argument("--field-patch-height", type=int, default=None)
    parser.add_argument("--field-patch-width", type=int, default=None)
    parser.add_argument("--field-input-steps", type=int, default=None)
    parser.add_argument("--field-target-steps", type=int, default=None)
    parser.add_argument("--field-target-steps-choices", type=str, default=None)
    parser.add_argument("--field-horizon-max-steps", type=int, default=None)
    parser.add_argument("--field-horizon-conditioning", action="store_true", default=None)
    parser.add_argument("--field-tendency-baseline", action="store_true", default=None)
    parser.add_argument("--field-tendency-scale", type=float, default=None)
    parser.add_argument("--field-diffusion-rate", type=float, default=None)
    parser.add_argument("--field-decay", type=float, default=None)
    parser.add_argument("--field-residual-readout", action="store_true", default=None)
    parser.add_argument("--field-residual-scale", type=float, default=None)
    parser.add_argument(
        "--field-adjacency-mode",
        type=str,
        default=None,
        choices=["grid", "moore", "line", "torus", "full"],
    )
    parser.add_argument(
        "--field-input-scope",
        type=str,
        default=None,
        choices=["global", "local", "radius1", "radius2", "radius4"],
    )
    parser.add_argument("--field-data-path", type=str, default=None)
    parser.add_argument("--field-output-dim", type=int, default=None)
    parser.add_argument("--field-train-start", type=int, default=None)
    parser.add_argument("--field-train-size", type=int, default=None)
    parser.add_argument("--field-eval-start", type=int, default=None)
    parser.add_argument("--field-eval-size", type=int, default=None)
    parser.add_argument("--field-stride", type=int, default=None)
    parser.add_argument("--field-variable", type=str, default=None)
    parser.add_argument("--field-variables", type=str, default=None)
    parser.add_argument(
        "--field-tokenizer",
        type=str,
        default=None,
        choices=["patch_mean", "conv_stem", "mlp_stem", "native_cell_state"],
    )
    parser.add_argument("--field-token-dim", type=int, default=None)
    parser.add_argument("--field-tokenizer-width", type=int, default=None)
    parser.add_argument("--field-decoder-width", type=int, default=None)
    parser.add_argument("--field-tokenizer-only", action="store_true", default=None)
    parser.add_argument("--field-baseline-scope", type=str, default=None, choices=["", "token_equivalent", "raw_field_anchor"])
    parser.add_argument("--baseline-model", type=str, default=None, choices=["", "convnet", "fno", "unet"])
    parser.add_argument("--baseline-width", type=int, default=None)
    parser.add_argument("--baseline-depth", type=int, default=None)
    parser.add_argument("--fno-modes", type=int, default=None)
    parser.add_argument("--maze-mode", type=str, default=None, choices=["random", "fixed", "fixed-set", "structured-random"])
    parser.add_argument("--fixed-set-size", type=int, default=None)
    parser.add_argument("--min-bfs-distance", type=int, default=None)
    parser.add_argument("--min-detour-gap", type=int, default=None)
    parser.add_argument("--max-generation-attempts", type=int, default=None)
    parser.add_argument("--maze-pool-size", type=int, default=None)
    parser.add_argument("--maze-pool-workers", type=int, default=None)
    parser.add_argument("--heldout-pool-path", type=str, default=None)
    parser.add_argument("--heldout-pool-size", type=int, default=None)
    parser.add_argument("--heldout-seed-offset", type=int, default=None)
    parser.add_argument("--evaluate-heldout", action="store_true", default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--edge-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--outer-steps", type=int, default=None)
    parser.add_argument("--inner-steps", type=int, default=None)
    parser.add_argument("--pair-chunk-size", type=int, default=None)
    parser.add_argument("--activation-checkpoint-inner", action="store_true", default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument(
        "--checkpoint-score",
        type=str,
        default=None,
        choices=["auto", "eval_mse", "field_horizon_stratified"],
    )
    parser.add_argument("--field-checkpoint-horizons", type=str, default=None)
    parser.add_argument("--field-checkpoint-eval-batches", type=int, default=None)
    parser.add_argument("--field-checkpoint-score-weights", type=str, default=None)
    parser.add_argument(
        "--field-checkpoint-score-metric",
        type=str,
        default=None,
        choices=["relative_mse", "mse"],
    )
    parser.add_argument("--field-checkpoint-seed-offset", type=int, default=None)
    parser.add_argument("--field-loss-weight", type=float, default=None)
    parser.add_argument("--neighbor-rank-loss-weight", type=float, default=None)
    parser.add_argument("--descent-margin-loss-weight", type=float, default=None)
    parser.add_argument("--bellman-loss-weight", type=float, default=None)
    parser.add_argument("--loss-margin", type=float, default=None)
    parser.add_argument("--bellman-step-scale", type=float, default=None)
    parser.add_argument("--visualize", action="store_true", default=None)
    parser.add_argument("--visual-every", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--pool-path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--reset-optimizer", action="store_true", default=None)
    parser.add_argument("--parent-checkpoint", type=str, default=None)
    parser.add_argument("--smoke-test-only", action="store_true", default=None)


COMMON_OVERRIDE_NAMES = [
    "seed",
    "task",
    "n_nodes",
    "grid_size",
    "wall_prob",
    "field_dataset",
    "field_grid_size",
    "field_grid_height",
    "field_grid_width",
    "field_patch_size",
    "field_patch_height",
    "field_patch_width",
    "field_input_steps",
    "field_target_steps",
    "field_target_steps_choices",
    "field_horizon_max_steps",
    "field_horizon_conditioning",
    "field_tendency_baseline",
    "field_tendency_scale",
    "field_diffusion_rate",
    "field_decay",
    "field_residual_readout",
    "field_residual_scale",
    "field_adjacency_mode",
    "field_input_scope",
    "field_data_path",
    "field_output_dim",
    "field_train_start",
    "field_train_size",
    "field_eval_start",
    "field_eval_size",
    "field_stride",
        "field_variable",
        "field_variables",
        "field_tokenizer",
        "field_token_dim",
        "field_tokenizer_width",
        "field_decoder_width",
        "field_tokenizer_only",
        "field_baseline_scope",
        "baseline_model",
    "baseline_width",
    "baseline_depth",
    "fno_modes",
    "maze_mode",
    "fixed_set_size",
    "min_bfs_distance",
    "min_detour_gap",
    "max_generation_attempts",
    "maze_pool_size",
    "maze_pool_workers",
    "heldout_pool_path",
    "heldout_pool_size",
    "heldout_seed_offset",
    "evaluate_heldout",
    "hidden_dim",
    "edge_dim",
    "batch_size",
    "epochs",
    "lr",
    "outer_steps",
    "inner_steps",
    "pair_chunk_size",
    "activation_checkpoint_inner",
    "precision",
    "grad_accum_steps",
    "max_grad_norm",
    "device",
    "log_every",
    "eval_batches",
    "checkpoint_score",
    "field_checkpoint_horizons",
    "field_checkpoint_eval_batches",
    "field_checkpoint_score_weights",
    "field_checkpoint_score_metric",
    "field_checkpoint_seed_offset",
    "field_loss_weight",
    "neighbor_rank_loss_weight",
    "descent_margin_loss_weight",
    "bellman_loss_weight",
    "loss_margin",
    "bellman_step_scale",
    "visualize",
    "visual_every",
    "run_dir",
    "pool_path",
    "checkpoint",
    "reset_optimizer",
    "parent_checkpoint",
    "smoke_test_only",
]


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config) if args.config else Config()
    return apply_overrides(cfg, args, COMMON_OVERRIDE_NAMES)
