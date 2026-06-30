import torch

from wca.config import Config
from wca.data.field.synthetic import make_field_batch
from wca.data.maze.metrics import compute_metrics
from wca.training.losses import compute_task_loss


def test_field_loss_is_patch_mse() -> None:
    cfg = Config(task="field", field_grid_size=8, field_patch_size=2, hidden_dim=8, batch_size=2)
    batch = make_field_batch(cfg, torch.device("cpu"))
    prediction = batch["label"] + 0.5

    loss = compute_task_loss("field", prediction, batch, cfg)

    assert torch.isclose(loss, torch.tensor(0.25))


def test_field_metrics_report_mse_mae_and_rollout_stability() -> None:
    cfg = Config(task="field", field_grid_size=8, field_patch_size=2, hidden_dim=8, batch_size=2)
    batch = make_field_batch(cfg, torch.device("cpu"))
    prediction = batch["label"].clone()

    metrics = compute_metrics("field", prediction, batch, cfg.grid_size)

    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert "field_energy_error" in metrics
    assert "field_patch_count" in metrics
    assert metrics["field_relative_l2"] == 0.0
    assert metrics["field_adjacency_density"] > 0.0
    assert metrics["field_adjacency_degree"] > 0.0
    assert metrics["field_input_visibility_density"] > 0.0
    assert metrics["field_input_visibility_degree"] > 0.0
    assert metrics["field_persistence_mse"] >= 0.0
    assert metrics["field_persistence_mae"] >= 0.0
    assert metrics["field_persistence_relative_l2"] >= 0.0
    assert metrics["field_delta_mse"] == 0.0
    assert metrics["field_delta_mae"] == 0.0
    assert metrics["field_mse_improvement_vs_persistence"] >= 0.0


def test_field_metrics_report_per_variable_errors() -> None:
    label = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    prediction = torch.tensor([[[2.0, 2.0], [5.0, 3.0]]])
    persistence = torch.tensor([[[0.0, 1.0], [2.0, 6.0]]])
    batch = {
        "label": label,
        "field_prediction_baseline": persistence,
        "field_variable": "2m_temperature,10m_u_component_of_wind",
    }

    metrics = compute_metrics("field", prediction, batch, grid_size=1)

    assert torch.isclose(torch.tensor(metrics["field_mse_2m_temperature"]), torch.tensor(2.5))
    assert torch.isclose(torch.tensor(metrics["field_mae_2m_temperature"]), torch.tensor(1.5))
    assert torch.isclose(torch.tensor(metrics["field_mse_10m_u_component_of_wind"]), torch.tensor(0.5))
    assert torch.isclose(torch.tensor(metrics["field_mae_10m_u_component_of_wind"]), torch.tensor(0.5))
    assert "field_mse_improvement_vs_persistence_2m_temperature" in metrics
    assert "field_mse_improvement_vs_persistence_10m_u_component_of_wind" in metrics
