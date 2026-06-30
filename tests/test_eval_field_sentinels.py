from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from scripts.eval_field_sentinels import (
    SENTINEL_EVAL_PLAN_FILENAME,
    build_model_and_forward,
    checkpoint_paths,
    effective_eval_samples,
    horizon_sentinel_required,
    run_sentinels_on_batch,
    summarize,
    write_csv,
    write_sentinel_eval_plan,
)
from wca.config import Config
from wca.data.field.synthetic import field_horizon_features


class FakeFieldModel(nn.Module):
    pass


def _batch() -> dict[str, torch.Tensor]:
    field_input = torch.zeros(2, 2, 1, 1, 2)
    field_input[0, -1, 0, 0] = torch.tensor([1.0, 1.0])
    field_input[1, -1, 0, 0] = torch.tensor([-1.0, -1.0])
    label = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    return {
        "field_input": field_input,
        "label": label,
        "field_target": label.reshape(2, 1, 1, 2).clone(),
        "field_target_steps_actual": torch.tensor([1, 1], dtype=torch.long),
    }


def _fake_forward(_model: nn.Module, cfg: Config, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    prediction = batch["field_input"][:, -1, 0].reshape(2, 2).clone()
    if cfg.field_horizon_conditioning:
        horizon = batch["field_target_steps_actual"].to(dtype=prediction.dtype).view(-1, 1)
        prediction = prediction + 0.1 * horizon
    return prediction


def _horizon_blind_forward(_model: nn.Module, _cfg: Config, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["field_input"][:, -1, 0].reshape(2, 2).clone()


def _materialized_horizon_forward(_model: nn.Module, _cfg: Config, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    horizon_feature = batch["H"][:, :, 2].mean(dim=1, keepdim=True)
    return horizon_feature.expand(2, 2).clone()


def test_sentinels_ignore_label_and_target_but_detect_horizon_and_input_changes() -> None:
    cfg = Config(task="field", field_horizon_conditioning=True)

    result = run_sentinels_on_batch(
        FakeFieldModel(),
        cfg,
        _batch(),
        forward_fn=_fake_forward,
        horizon_probe=4,
    )

    assert result["label_leakage_max_abs_diff"] == 0.0
    assert result["label_leakage_pass"] is True
    assert result["horizon_shuffle_max_abs_diff"] > 0.0
    assert result["horizon_shuffle_pass"] is True
    assert result["input_shuffle_mse_ratio"] > 1.05
    assert result["input_shuffle_degraded"] is True


def test_horizon_conditioned_model_with_zero_horizon_diff_is_failure() -> None:
    cfg = Config(task="field", field_horizon_conditioning=True)

    result = run_sentinels_on_batch(
        FakeFieldModel(),
        cfg,
        _batch(),
        forward_fn=_horizon_blind_forward,
        horizon_probe=4,
    )

    assert result["horizon_shuffle_max_abs_diff"] == 0.0
    assert result["horizon_shuffle_required"] is True
    assert result["horizon_shuffle_pass"] is False


def test_patchmean_wca_horizon_sentinel_updates_materialized_h_channels() -> None:
    cfg = Config(
        task="field",
        field_horizon_conditioning=True,
        field_grid_height=1,
        field_grid_width=2,
        field_patch_height=1,
        field_patch_width=1,
        field_patch_size=1,
        field_horizon_max_steps=8,
        hidden_dim=8,
    )
    batch = _batch()
    H = torch.zeros(2, 2, 8)
    H[:, :, 2:6] = field_horizon_features(cfg, 1, H.device, H.dtype).view(1, 1, 4)
    batch["H"] = H

    result = run_sentinels_on_batch(
        FakeFieldModel(),
        cfg,
        batch,
        forward_fn=_materialized_horizon_forward,
        horizon_probe=4,
    )

    assert result["horizon_shuffle_max_abs_diff"] > 0.0
    assert result["horizon_shuffle_required"] is True
    assert result["horizon_shuffle_pass"] is True


def test_mechanism_negative_controls_do_not_require_horizon_sensitivity() -> None:
    tokenizer_only = Config(task="field", field_horizon_conditioning=True, field_tokenizer_only=True, outer_steps=10)
    outer0 = Config(task="field", field_horizon_conditioning=True, field_tokenizer_only=False, outer_steps=0)
    token_baseline = Config(task="field", baseline_model="token_mlp", field_horizon_conditioning=True, outer_steps=0)

    assert horizon_sentinel_required(tokenizer_only) is False
    assert horizon_sentinel_required(outer0) is False
    assert horizon_sentinel_required(token_baseline) is True

    result = run_sentinels_on_batch(
        FakeFieldModel(),
        tokenizer_only,
        _batch(),
        forward_fn=_horizon_blind_forward,
        horizon_probe=4,
    )

    assert result["horizon_shuffle_max_abs_diff"] == 0.0
    assert result["horizon_shuffle_required"] is False
    assert result["horizon_shuffle_pass"] is True


def test_summary_and_csv_record_hard_failures(tmp_path: Path) -> None:
    rows = [
        {
            "source_run_dir": "runs/example",
            "checkpoint_kind": "final",
            "checkpoint_path": "runs/example/model.pt",
            "horizon": 1,
            "model": "fake",
            "seed": 42,
            "eval_plan_seed": 7,
            "eval_horizon_seed": 1016,
            "eval_plan_hash": "abc",
            "eval_start_indices_hash": "starts",
            "eval_sample_count": 2,
            "field_horizon_conditioning": True,
            "field_tendency_baseline": False,
            "baseline_mse": 0.0,
            "label_leakage_max_abs_diff": 0.0,
            "label_leakage_pass": True,
            "horizon_probe_horizon": 4,
            "horizon_shuffle_max_abs_diff": 0.0,
            "horizon_shuffle_required": True,
            "horizon_shuffle_pass": False,
            "input_shuffle_mse": 4.0,
            "input_shuffle_mse_ratio": 4000000000000.0,
            "input_shuffle_degraded": True,
        }
    ]

    summary = summarize(rows, horizons=[1], checkpoint_kinds=["final"], eval_seed=7)
    write_csv(tmp_path / "sentinel_results.csv", rows)

    assert summary["hard_failure_count"] == 1
    csv_text = (tmp_path / "sentinel_results.csv").read_text(encoding="utf-8")
    assert "label_leakage_max_abs_diff" in csv_text
    assert "false" in csv_text


def test_sentinel_eval_plan_does_not_overwrite_formal_eval_plan(tmp_path: Path) -> None:
    write_sentinel_eval_plan(
        tmp_path,
        {1: torch.tensor([10, 20], dtype=torch.long)},
        eval_seed=7,
        field_split="test",
    )

    assert SENTINEL_EVAL_PLAN_FILENAME != "eval_plan.json"
    assert (tmp_path / SENTINEL_EVAL_PLAN_FILENAME).exists()
    assert not (tmp_path / "eval_plan.json").exists()


def test_effective_eval_samples_defaults_to_one_batch_for_real_cache_only() -> None:
    real_cfg = Config(task="field", field_dataset="weatherbench2_era5_cache", batch_size=7)
    synthetic_cfg = Config(task="field", field_dataset="synthetic_heat", batch_size=7)

    assert effective_eval_samples(real_cfg, requested_eval_samples=0, requested_eval_batch_size=0) == 7
    assert effective_eval_samples(real_cfg, requested_eval_samples=0, requested_eval_batch_size=3) == 3
    assert effective_eval_samples(real_cfg, requested_eval_samples=5, requested_eval_batch_size=3) == 5
    assert effective_eval_samples(synthetic_cfg, requested_eval_samples=0, requested_eval_batch_size=0) == 0


def test_token_baseline_checkpoint_paths_use_state_dict_files(tmp_path: Path) -> None:
    cfg = Config(task="field", baseline_model="token_mlp")
    (tmp_path / "final_model.pt").write_bytes(b"final")
    (tmp_path / "best_model.pt").write_bytes(b"best")

    assert checkpoint_paths(tmp_path, cfg, ["final", "best", "model"]) == [
        ("final", tmp_path / "final_model.pt"),
        ("best", tmp_path / "best_model.pt"),
    ]


def test_sentinel_builds_token_baseline_forward_without_wca_checkpoint_loader() -> None:
    cfg = Config(
        task="field",
        baseline_model="token_mlp",
        field_grid_size=8,
        field_patch_size=4,
        field_output_dim=1,
        baseline_width=4,
        baseline_depth=2,
    )
    model, forward_fn, model_name = build_model_and_forward(cfg, torch.device("cpu"))
    batch = {
        "field_prediction_baseline": torch.zeros(2, 4),
        "field_previous_tokens": torch.zeros(2, 4),
        "label": torch.zeros(2, 4),
    }

    prediction = forward_fn(model, cfg, batch)

    assert model_name == "token_mlp-field-token-baseline"
    assert prediction.shape == (2, 4)
