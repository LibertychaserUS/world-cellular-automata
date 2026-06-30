import torch

from wca.config import Config
from wca.data.maze.batch import make_maze_batch
from wca.models.rws_nca import FullRecursiveWorldStateNCA


def test_baseline_model_shape_invariants() -> None:
    cfg = Config(task="maze", grid_size=3, n_nodes=9, hidden_dim=12, edge_dim=4, batch_size=2, inner_steps=1, outer_steps=1)
    batch = make_maze_batch(cfg, torch.device("cpu"))
    model = FullRecursiveWorldStateNCA(cfg.n_nodes, cfg.hidden_dim, cfg.edge_dim, cfg.inner_steps)

    L = model.project_full_world(batch["H"])
    H_final, diagnostics = model(batch["H"], batch["adjacency"], cfg.outer_steps)

    assert batch["H"].shape == (2, 9, 12)
    assert L.shape == (2, 9, 9, 12)
    assert H_final.shape == (2, 9, 12)
    assert diagnostics["last_local_worlds"].shape == (2, 9, 9, 12)


def test_input_visibility_masks_only_selected_channels() -> None:
    model = FullRecursiveWorldStateNCA(n_nodes=4, hidden_dim=4, edge_dim=2, inner_steps=1)
    H = torch.arange(1 * 4 * 4, dtype=torch.float32).view(1, 4, 4)
    visibility = torch.eye(4).unsqueeze(0)
    channel_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

    local_worlds = model.project_full_world(H, input_visibility=visibility, input_visibility_channels=channel_mask)
    base = model.project_full_world(H)

    assert torch.allclose(local_worlds[0, 0, 1, :2], base[0, 0, 1, :2] - H[0, 1, :2], atol=1e-6)
    assert torch.allclose(local_worlds[0, 0, 1, 2:], base[0, 0, 1, 2:])
