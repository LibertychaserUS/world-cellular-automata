import torch

from wca.config import Config, apply_overrides
from argparse import Namespace
from wca.data.field.synthetic import make_field_batch
from wca.models.field_tokenizers import ConvStemTokenizer, MLPStemTokenizer, extract_field_patches
from wca.models.field_wca import FieldTokenizerWCA
from wca.training.losses import compute_task_loss
from wca.training.trainer import _model_report_name_and_details


def test_extract_field_patches_preserves_patch_pixels() -> None:
    field = torch.arange(1 * 2 * 4 * 6, dtype=torch.float32).view(1, 2, 4, 6)

    patches = extract_field_patches(field, patch_size=(2, 3))

    assert patches.shape == (1, 4, 2, 2, 3)
    assert torch.equal(patches[0, 0], field[0, :, :2, :3])
    assert torch.equal(patches[0, 3], field[0, :, 2:4, 3:6])


def test_conv_and_mlp_tokenizers_emit_node_tokens() -> None:
    field_input = torch.randn(2, 2, 3, 8, 8)

    conv = ConvStemTokenizer(input_steps=2, channels=3, token_dim=7, width=5)
    mlp = MLPStemTokenizer(input_steps=2, channels=3, patch_size=4, token_dim=7, width=11)

    assert conv(field_input, patch_size=4).shape == (2, 4, 7)
    assert mlp(field_input, patch_size=4).shape == (2, 4, 7)


def test_field_tokenizer_config_overrides_are_explicit() -> None:
    cfg = Config(task="field")
    overrides = Namespace(
        field_tokenizer="conv_stem",
        field_token_dim=32,
        field_tokenizer_width=48,
        field_decoder_width=24,
        field_tokenizer_only=True,
        field_baseline_scope="token_equivalent",
    )

    updated = apply_overrides(
        cfg,
        overrides,
        [
            "field_tokenizer",
            "field_token_dim",
            "field_tokenizer_width",
            "field_decoder_width",
            "field_tokenizer_only",
            "field_baseline_scope",
        ],
    )

    assert updated.field_tokenizer == "conv_stem"
    assert updated.field_token_dim == 32
    assert updated.field_tokenizer_width == 48
    assert updated.field_decoder_width == 24
    assert updated.field_tokenizer_only is True
    assert updated.field_baseline_scope == "token_equivalent"


def test_field_tokenizer_wca_smoke_and_gradient_flow() -> None:
    cfg = Config(
        task="field",
        field_tokenizer="conv_stem",
        field_token_dim=8,
        field_grid_size=8,
        field_patch_size=4,
        field_input_steps=2,
        field_residual_readout=True,
        field_residual_scale=0.01,
        hidden_dim=20,
        edge_dim=4,
        batch_size=2,
        inner_steps=1,
        outer_steps=1,
    )
    batch = make_field_batch(cfg, torch.device("cpu"))
    model = FieldTokenizerWCA(cfg)

    prediction, diagnostics = model(batch, cfg.outer_steps)
    loss = compute_task_loss(cfg.task, prediction, batch, cfg)
    loss.backward()

    grad_norm = sum(
        parameter.grad.detach().abs().sum().item()
        for parameter in model.tokenizer.parameters()
        if parameter.grad is not None
    )
    assert prediction.shape == batch["label"].shape
    assert diagnostics["field_H"].shape == (2, 4, 20)
    assert diagnostics["last_local_worlds"].shape == (2, 4, 4, 20)
    assert grad_norm > 0.0


def test_field_tokenizer_wca_prediction_does_not_depend_on_target_label() -> None:
    torch.manual_seed(123)
    cfg = Config(
        task="field",
        field_tokenizer="mlp_stem",
        field_token_dim=8,
        field_grid_size=8,
        field_patch_size=4,
        field_input_steps=2,
        hidden_dim=20,
        edge_dim=4,
        batch_size=1,
        inner_steps=1,
        outer_steps=1,
    )
    batch = make_field_batch(cfg, torch.device("cpu"))
    model = FieldTokenizerWCA(cfg)

    first, _ = model(batch, cfg.outer_steps)
    altered = dict(batch)
    altered["field_target"] = batch["field_target"] + 1000.0
    altered["label"] = batch["label"] + 1000.0
    second, _ = model(altered, cfg.outer_steps)

    assert torch.allclose(first, second)


def test_field_tokenizer_only_bypasses_wca_core_and_reports_params() -> None:
    cfg = Config(
        task="field",
        field_tokenizer="conv_stem",
        field_tokenizer_only=True,
        field_token_dim=8,
        field_grid_size=8,
        field_patch_size=4,
        field_input_steps=2,
        field_residual_readout=True,
        field_residual_scale=0.01,
        hidden_dim=20,
        edge_dim=4,
        batch_size=2,
        inner_steps=1,
        outer_steps=10,
    )
    batch = make_field_batch(cfg, torch.device("cpu"))
    model = FieldTokenizerWCA(cfg)

    prediction, diagnostics = model(batch, cfg.outer_steps)
    loss = compute_task_loss(cfg.task, prediction, batch, cfg)
    loss.backward()
    breakdown = model.parameter_breakdown()

    assert prediction.shape == batch["label"].shape
    assert "last_local_worlds" not in diagnostics
    assert float(diagnostics["field_core_executed"].item()) == 0.0
    assert breakdown["wca_core_params"] == 0
    assert breakdown["field_tokenizer_params"] > 0
    assert breakdown["field_decoder_params"] > 0
    assert any(parameter.grad is not None for parameter in model.tokenizer.parameters())


def test_field_tokenizer_outer_zero_bypasses_existing_wca_core() -> None:
    cfg = Config(
        task="field",
        field_tokenizer="mlp_stem",
        field_token_dim=8,
        field_grid_size=8,
        field_patch_size=4,
        field_input_steps=2,
        hidden_dim=20,
        edge_dim=4,
        batch_size=1,
        inner_steps=1,
        outer_steps=0,
    )
    batch = make_field_batch(cfg, torch.device("cpu"))
    model = FieldTokenizerWCA(cfg)

    prediction, diagnostics = model(batch, cfg.outer_steps)
    breakdown = model.parameter_breakdown()

    assert prediction.shape == batch["label"].shape
    assert "last_local_worlds" not in diagnostics
    assert float(diagnostics["field_core_executed"].item()) == 0.0
    assert breakdown["wca_core_params"] > 0
    model_name, details = _model_report_name_and_details(model)
    assert model_name == "mlp_stem-tokenizer-bypass-o0"
    assert details["wca_core_executed_by_config"] is False
