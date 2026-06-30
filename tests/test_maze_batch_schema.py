import torch

from wca.config import Config
from wca.constants import GOAL_CH, START_CH
from wca.data.maze.batch import make_maze_batch


def test_maze_batch_has_explicit_start_and_goal() -> None:
    cfg = Config(task="maze", grid_size=4, n_nodes=16, hidden_dim=12, batch_size=3)
    batch = make_maze_batch(cfg, torch.device("cpu"))

    assert batch["H"].shape == (3, 16, 12)
    assert batch["adjacency"].shape == (3, 16, 16)
    assert batch["distance_field"].shape == (3, 16)
    assert batch["distance_mask"].shape == (3, 16)
    assert batch["start_idx"].shape == (3,)
    assert batch["goal_idx"].shape == (3,)
    assert torch.equal(batch["target_idx"], batch["start_idx"])


def test_start_goal_channels_match_explicit_indices() -> None:
    cfg = Config(task="maze", grid_size=4, n_nodes=16, hidden_dim=12, batch_size=2)
    batch = make_maze_batch(cfg, torch.device("cpu"))
    H = batch["H"]
    for b in range(cfg.batch_size):
        assert H[b, int(batch["start_idx"][b]), START_CH].item() == 1.0
        assert H[b, int(batch["goal_idx"][b]), GOAL_CH].item() == 1.0
