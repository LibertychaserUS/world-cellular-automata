import torch

from wca.config import Config
from wca.data.maze.batch import make_maze_batch


def test_distance_field_is_not_copied_into_input_channels() -> None:
    cfg = Config(task="maze", grid_size=4, n_nodes=16, hidden_dim=12, batch_size=2)
    batch = make_maze_batch(cfg, torch.device("cpu"))
    H = batch["H"]
    distance_field = batch["distance_field"]

    for channel in range(H.shape[-1]):
        assert not torch.allclose(H[:, :, channel], distance_field)
