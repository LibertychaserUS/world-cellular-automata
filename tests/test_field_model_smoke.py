import torch

from wca.config import Config
from wca.data.field.real_cache import make_real_field_batch
from wca.data.field.synthetic import make_field_batch
from wca.models.rws_nca import FullRecursiveWorldStateNCA
from wca.training.losses import compute_task_loss
from wca.training.prediction import predict_for_task


def test_field_batch_runs_through_baseline_wca() -> None:
    cfg = Config(
        task="field",
        field_grid_size=16,
        field_patch_size=4,
        field_input_steps=2,
        hidden_dim=16,
        edge_dim=4,
        batch_size=2,
        inner_steps=1,
        outer_steps=1,
    )
    batch = make_field_batch(cfg, torch.device("cpu"))
    model = FullRecursiveWorldStateNCA(cfg.n_nodes, cfg.hidden_dim, cfg.edge_dim, cfg.inner_steps)

    H_final, diagnostics = model(batch["H"], batch["adjacency"], cfg.outer_steps)
    prediction = model.predict_all_nodes(H_final)

    assert H_final.shape == (2, 16, 16)
    assert diagnostics["last_local_worlds"].shape == (2, 16, 16, 16)
    assert prediction.shape == batch["label"].shape


def test_field_readout_can_predict_vector_nodes() -> None:
    model = FullRecursiveWorldStateNCA(n_nodes=4, hidden_dim=8, edge_dim=4, inner_steps=1, output_dim=3)
    H = torch.randn(2, 4, 8)
    adjacency = torch.eye(4).unsqueeze(0).expand(2, 4, 4)

    H_final, _diagnostics = model(H, adjacency, outer_steps=1)
    prediction = model.predict_all_nodes(H_final)

    assert prediction.shape == (2, 4, 3)


def test_real_vector_field_batch_runs_prediction_and_loss(tmp_path) -> None:
    cache_path = tmp_path / "weather_vector_cache.pt"
    base = torch.arange(12, dtype=torch.float32).view(12, 1, 1, 1).expand(12, 1, 8, 8)
    data = torch.cat([base, base + 10.0, base - 10.0], dim=1).contiguous()
    torch.save({"schema_version": 1, "variables": ["t2m", "u10", "v10"], "data": data}, cache_path)
    cfg = Config(
        task="field",
        field_dataset="weatherbench_cache",
        field_data_path=str(cache_path),
        field_output_dim=3,
        field_grid_size=8,
        field_patch_size=2,
        field_input_steps=2,
        field_target_steps=1,
        field_residual_readout=True,
        field_residual_scale=0.02,
        hidden_dim=12,
        edge_dim=4,
        batch_size=2,
        inner_steps=1,
        outer_steps=1,
    )
    batch = make_real_field_batch(cfg, torch.device("cpu"))
    model = FullRecursiveWorldStateNCA(
        n_nodes=cfg.n_nodes,
        hidden_dim=cfg.hidden_dim,
        edge_dim=cfg.edge_dim,
        inner_steps=cfg.inner_steps,
        output_dim=cfg.field_output_dim,
    )

    H_final, _diagnostics = model(
        batch["H"],
        batch["adjacency"],
        cfg.outer_steps,
        input_visibility=batch["input_visibility"],
        input_visibility_channels=batch["input_visibility_channels"],
    )
    prediction = predict_for_task(model, cfg, H_final, batch)
    loss = compute_task_loss(cfg.task, prediction, batch, cfg)

    assert prediction.shape == batch["label"].shape == (2, 16, 3)
    assert torch.isfinite(loss)
