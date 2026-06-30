from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from scripts.eval_field_horizon_stratified import (
    _discover_field_run_dirs,
    _evaluate_token_baseline_run,
    _is_baseline_run,
    _is_token_baseline_run,
    _validate_strict_matched_run_scope,
    _validate_strict_run_contracts,
    _validate_matched_eval_plan,
    _validate_matched_persistence,
)
from scripts.field_eval_plan import (
    attach_fixed_eval_plan,
    eval_plan_hash,
    horizon_eval_seed,
    make_fixed_eval_start_indices,
    start_indices_hash,
)
from scripts.train_field_baseline import build_model
from scripts.train_field_token_baseline import build_model as build_token_model
from scripts.train_field_token_baseline import input_dim_for_config
from wca.config import Config


def test_field_horizon_eval_discovers_timestamped_runs(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "experiment" / "20260622_000000"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(Config(task="field").to_dict()), encoding="utf-8")
    torch.save({}, run / "model.pt")

    assert _discover_field_run_dirs([tmp_path / "runs" / "experiment"]) == [run]


def test_field_horizon_eval_identifies_baseline_run_from_summary(tmp_path: Path) -> None:
    model = build_model("convnet", output_dim=1, width=4, depth=3, modes=4, condition_channels=4)
    (tmp_path / "config.json").write_text(json.dumps(Config(task="field").to_dict()), encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps({"model": "convnet-field-baseline"}), encoding="utf-8")
    torch.save(model.state_dict(), tmp_path / "model.pt")

    assert _is_baseline_run(tmp_path) is True


def test_field_horizon_eval_identifies_baseline_run_from_config_metadata(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(Config(task="field", baseline_model="fno", baseline_width=8, baseline_depth=2, fno_modes=4).to_dict()),
        encoding="utf-8",
    )

    assert _is_baseline_run(tmp_path) is True


def test_field_horizon_eval_identifies_token_baseline_without_raw_baseline_fallback(tmp_path: Path) -> None:
    cfg = Config(
        task="field",
        baseline_model="token_conv",
        field_baseline_scope="token_equivalent",
        field_grid_size=8,
        field_patch_size=4,
        field_output_dim=1,
        baseline_width=4,
        baseline_depth=2,
    )
    model = build_token_model(
        "token_conv",
        input_dim_for_config(cfg),
        output_dim=1,
        token_shape=(2, 2),
        width=4,
        depth=2,
    )
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    torch.save(model.state_dict(), tmp_path / "model.pt")

    assert _is_token_baseline_run(tmp_path) is True
    assert _is_baseline_run(tmp_path) is False


def test_field_horizon_eval_does_not_treat_token_equivalent_wca_as_token_baseline(tmp_path: Path) -> None:
    cfg = Config(
        task="field",
        baseline_model="",
        field_tokenizer="mlp_stem",
        field_baseline_scope="token_equivalent",
    )
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    (tmp_path / "summary.json").write_text(
        json.dumps({"model": "FieldTokenizerWCA", "field_baseline_scope": "token_equivalent"}),
        encoding="utf-8",
    )
    torch.save({}, tmp_path / "model.pt")

    assert _is_token_baseline_run(tmp_path) is False


def test_strict_scope_preflight_rejects_raw_field_external_anchor_in_token_table(tmp_path: Path) -> None:
    token_run = tmp_path / "token"
    token_run.mkdir()
    token_cfg = Config(
        task="field",
        baseline_model="token_mlp",
        field_baseline_scope="token_equivalent",
        field_grid_height=128,
        field_grid_width=128,
        field_patch_height=16,
        field_patch_width=16,
    )
    (token_run / "config.json").write_text(json.dumps(token_cfg.to_dict()), encoding="utf-8")

    raw_anchor = tmp_path / "raw_anchor"
    raw_anchor.mkdir()
    raw_cfg = Config(
        task="field",
        baseline_model="fno",
        field_grid_height=128,
        field_grid_width=128,
        field_patch_height=16,
        field_patch_width=16,
    )
    (raw_anchor / "config.json").write_text(json.dumps(raw_cfg.to_dict()), encoding="utf-8")

    try:
        _validate_strict_matched_run_scope([token_run, raw_anchor])
    except SystemExit as exc:
        assert "raw-field external anchors" in str(exc)
        assert "raw_field_external_anchor_not_token_equivalent" in str(exc)
    else:
        raise AssertionError("Expected strict scope preflight to reject raw-field external anchors.")


def test_field_horizon_eval_token_baseline_uses_token_evaluator(monkeypatch, tmp_path: Path) -> None:
    cfg = Config(
        task="field",
        baseline_model="token_mlp",
        field_baseline_scope="token_equivalent",
        field_grid_size=8,
        field_patch_size=4,
        field_output_dim=1,
        baseline_width=4,
        baseline_depth=2,
        seed=123,
    )
    model = build_token_model(
        "token_mlp",
        input_dim_for_config(cfg),
        output_dim=1,
        token_shape=(2, 2),
        width=4,
        depth=2,
    )
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    torch.save(model.state_dict(), tmp_path / "model.pt")

    calls = {"token": 0}

    def fake_evaluate(_model, returned_cfg, _device):
        calls["token"] += 1
        assert returned_cfg.field_baseline_scope == "token_equivalent"
        return {
            "eval_mse": 1.0,
            "eval_mae": 0.5,
            "eval_field_relative_l2": 0.25,
            "eval_field_persistence_mse": 2.0,
            "eval_field_persistence_mae": 1.0,
            "eval_field_mse_improvement_vs_persistence": 0.5,
        }

    monkeypatch.setattr("scripts.eval_field_horizon_stratified.evaluate_token_model", fake_evaluate)

    rows = _evaluate_token_baseline_run(
        tmp_path,
        [4],
        ["best"],
        eval_batches=1,
        device_name="cpu",
        eval_seed=2026062201,
        eval_samples=0,
        eval_batch_size=0,
        per_sample_rows=None,
        field_split="test",
    )

    assert calls["token"] == 1
    assert rows[0]["model"] == "token_mlp-field-token-baseline"
    assert rows[0]["field_baseline_scope"] == "token_equivalent"
    assert rows[0]["baseline_model"] == "token_mlp"
    assert rows[0]["eval_mse"] == 1.0


def test_eval_plan_hash_changes_when_seed_or_batch_count_changes() -> None:
    base = eval_plan_hash([1, 2, 4, 8], 64, 2026062201)

    assert horizon_eval_seed(2026062201, 8) == 2026062201 + 8 * 1009
    assert eval_plan_hash([1, 2, 4, 8], 64, 2026062201) == base
    assert eval_plan_hash([1, 2, 4, 8], 32, 2026062201) != base
    assert eval_plan_hash([1, 2, 4, 8], 64, 2026062202) != base
    assert eval_plan_hash([1, 2, 4, 8], 64, 2026062201, eval_samples=64, eval_batch_size=1) != base


def test_fixed_eval_start_indices_are_deterministic_for_real_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.pt"
    torch.save({"data": torch.zeros(32, 1, 8, 8), "variables": ["x"]}, cache_path)
    cfg = Config(
        task="field",
        field_dataset="pdebench_cache",
        field_data_path=str(cache_path),
        field_grid_size=8,
        field_patch_size=2,
        field_input_steps=2,
        field_target_steps=4,
        field_eval_start=8,
        field_eval_size=16,
        batch_size=4,
    )

    first = make_fixed_eval_start_indices(cfg, horizon=4, eval_seed=123, eval_samples=10)
    second = make_fixed_eval_start_indices(cfg, horizon=4, eval_seed=123, eval_samples=10)
    different = make_fixed_eval_start_indices(cfg, horizon=4, eval_seed=124, eval_samples=10)

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert len(set(first.tolist())) == 10
    assert int(first.min().item()) >= 8
    assert int(first.max().item()) < 8 + 16 - (2 + 4 - 1)
    assert eval_plan_hash(
        [4],
        10,
        123,
        eval_samples=10,
        eval_batch_size=1,
        start_indices_by_horizon={4: first},
    ) != eval_plan_hash(
        [4],
        10,
        123,
        eval_samples=10,
        eval_batch_size=1,
        start_indices_by_horizon={4: different},
    )
    assert start_indices_hash(first) == start_indices_hash(second)


def test_attach_fixed_eval_plan_overrides_batching() -> None:
    cfg = Config(task="field", batch_size=8, eval_batches=99)
    attach_fixed_eval_plan(cfg, torch.arange(6), eval_batch_size=2)

    assert cfg.batch_size == 2
    assert cfg.eval_batches == 3
    assert torch.equal(cfg._field_fixed_start_indices, torch.arange(6))
    assert cfg._field_fixed_start_cursor == 0


def test_matched_persistence_validation_rejects_different_eval_samples() -> None:
    rows = [
        {"source_run_dir": "run/a", "checkpoint_kind": "final", "horizon": 8, "eval_field_persistence_mse": 0.1},
        {"source_run_dir": "run/b", "checkpoint_kind": "final", "horizon": 8, "eval_field_persistence_mse": 0.2},
    ]

    try:
        _validate_matched_persistence(rows)
    except SystemExit as exc:
        assert "persistence baselines differ" in str(exc)
    else:
        raise AssertionError("Expected mismatched persistence validation to fail.")


def test_matched_persistence_validation_accepts_same_eval_samples() -> None:
    rows = [
        {"source_run_dir": "run/a", "checkpoint_kind": "final", "horizon": 8, "eval_field_persistence_mse": 0.1},
        {"source_run_dir": "run/b", "checkpoint_kind": "best", "horizon": 8, "eval_field_persistence_mse": 0.10000000001},
        {"source_run_dir": "run/c", "checkpoint_kind": "final", "horizon": 4, "eval_field_persistence_mse": 0.01},
    ]

    _validate_matched_persistence(rows)


def test_matched_eval_plan_validation_rejects_different_start_hashes() -> None:
    rows = [
        {
            "source_run_dir": "run/a",
            "horizon": 4,
            "eval_plan_hash": "plan-a",
            "eval_start_indices_hash": "starts-a",
            "eval_sample_count": 64,
        },
        {
            "source_run_dir": "run/b",
            "horizon": 4,
            "eval_plan_hash": "plan-b",
            "eval_start_indices_hash": "starts-b",
            "eval_sample_count": 64,
        },
    ]

    try:
        _validate_matched_eval_plan(rows)
    except SystemExit as exc:
        assert "eval plan hashes differ" in str(exc)
    else:
        raise AssertionError("Expected mismatched eval plan validation to fail.")


def _write_field_run_config(
    run_dir: Path,
    *,
    field_grid_height: int = 128,
    field_grid_width: int = 128,
    field_patch_height: int = 8,
    field_patch_width: int = 8,
    baseline_model: str = "",
) -> None:
    run_dir.mkdir(parents=True)
    cfg = Config(
        task="field",
        field_dataset="pdebench_cache",
        field_data_path="artifacts/field_datasets/pdebench/v20/cache.pt",
        field_grid_height=field_grid_height,
        field_grid_width=field_grid_width,
        field_patch_height=field_patch_height,
        field_patch_width=field_patch_width,
        field_output_dim=2,
        field_input_steps=2,
        field_stride=1,
        field_train_start=0,
        field_train_size=800,
        field_eval_start=800,
        field_eval_size=100,
        field_val_start=800,
        field_val_size=100,
        field_test_start=900,
        field_test_size=100,
        baseline_model=baseline_model,
    )
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    torch.save({}, run_dir / "model.pt")


def test_strict_run_contract_rejects_mixed_token_geometry_before_expensive_eval(tmp_path: Path) -> None:
    n256 = tmp_path / "runs" / "wca_n256"
    n64 = tmp_path / "runs" / "fno_n64"
    _write_field_run_config(n256, field_patch_height=8, field_patch_width=8)
    _write_field_run_config(n64, field_patch_height=16, field_patch_width=16, baseline_model="fno")

    try:
        _validate_strict_run_contracts([n256, n64])
    except SystemExit as exc:
        message = str(exc)
        assert "shared label/eval-token contract" in message
        assert "field_patch_height" in message
        assert "n_nodes" in message
    else:
        raise AssertionError("Expected mixed token geometry to fail strict run-contract validation.")


def test_strict_run_contract_accepts_same_token_geometry_across_model_families(tmp_path: Path) -> None:
    wca = tmp_path / "runs" / "wca"
    fno = tmp_path / "runs" / "fno"
    _write_field_run_config(wca, field_patch_height=8, field_patch_width=8)
    _write_field_run_config(fno, field_patch_height=8, field_patch_width=8, baseline_model="fno")

    _validate_strict_run_contracts([wca, fno])


def test_contract_check_only_exits_before_checkpoint_loading(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts import eval_field_horizon_stratified

    wca = tmp_path / "runs" / "wca"
    token = tmp_path / "runs" / "token"
    _write_field_run_config(wca, field_patch_height=8, field_patch_width=8)
    _write_field_run_config(token, field_patch_height=8, field_patch_width=8, baseline_model="token_mlp")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("contract-check-only must not evaluate checkpoints")

    monkeypatch.setattr(eval_field_horizon_stratified, "_evaluate_wca_run", fail_if_called)
    monkeypatch.setattr(eval_field_horizon_stratified, "_evaluate_token_baseline_run", fail_if_called)
    exit_code = eval_field_horizon_stratified.main(
        [
            wca.as_posix(),
            token.as_posix(),
            "--eval-batches",
            "2",
            "--eval-samples",
            "2",
            "--eval-batch-size",
            "1",
            "--contract-check-only",
        ]
    )

    assert exit_code is None
    assert json.loads(capsys.readouterr().out)["strict_contract_ok"] is True
