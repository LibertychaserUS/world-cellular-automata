from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from wca.config import Config
from wca.schemas import TensorBatch


def field_grid_shape(cfg: Config) -> tuple[int, int]:
    height = int(getattr(cfg, "field_grid_height", 0) or cfg.field_grid_size)
    width = int(getattr(cfg, "field_grid_width", 0) or cfg.field_grid_size)
    if height <= 0 or width <= 0:
        raise ValueError(f"field grid height/width must be positive, got {height}x{width}")
    return height, width


def field_patch_shape(cfg: Config) -> tuple[int, int]:
    patch_height = int(getattr(cfg, "field_patch_height", 0) or cfg.field_patch_size)
    patch_width = int(getattr(cfg, "field_patch_width", 0) or cfg.field_patch_size)
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"field patch height/width must be positive, got {patch_height}x{patch_width}")
    return patch_height, patch_width


def field_token_shape(cfg: Config) -> tuple[int, int]:
    height, width = field_grid_shape(cfg)
    patch_height, patch_width = field_patch_shape(cfg)
    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(f"field grid {height}x{width} must be divisible by patch {patch_height}x{patch_width}")
    return height // patch_height, width // patch_width


def patch_grid_size(cfg: Config) -> int:
    token_height, token_width = field_token_shape(cfg)
    if token_height != token_width:
        raise ValueError(
            "patch_grid_size is only defined for square field token grids. "
            f"Use field_token_shape for rectangular grids, got {token_height}x{token_width}."
        )
    return token_height


def configure_field_nodes(cfg: Config) -> None:
    token_height, token_width = field_token_shape(cfg)
    cfg.grid_size = token_height
    cfg.n_nodes = token_height * token_width


def parse_field_target_steps_choices(raw: str) -> list[int]:
    choices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if any(choice <= 0 for choice in choices):
        raise ValueError(f"field_target_steps_choices must contain positive integers, got {raw!r}")
    return choices


def choose_field_target_steps(cfg: Config, device: torch.device) -> int:
    raw_choices = getattr(cfg, "field_target_steps_choices", "")
    choices = parse_field_target_steps_choices(raw_choices) if raw_choices else []
    if not choices:
        return int(cfg.field_target_steps)
    index = int(torch.randint(0, len(choices), (1,), device=device).item())
    return choices[index]


def max_field_target_steps(cfg: Config) -> int:
    explicit_max = int(getattr(cfg, "field_horizon_max_steps", 0) or 0)
    if explicit_max > 0:
        return explicit_max
    raw_choices = getattr(cfg, "field_target_steps_choices", "")
    choices = parse_field_target_steps_choices(raw_choices) if raw_choices else []
    return max(choices) if choices else int(cfg.field_target_steps)


def field_horizon_features(cfg: Config, target_steps: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    max_steps = max(1, max_field_target_steps(cfg))
    horizon = torch.tensor(float(target_steps), device=device, dtype=dtype)
    normalized = horizon / float(max_steps)
    log_scaled = torch.log1p(horizon) / torch.log1p(torch.tensor(float(max_steps), device=device, dtype=dtype))
    phase = normalized * torch.pi
    return torch.stack([normalized, torch.sin(phase), torch.cos(phase), log_scaled])


def inject_field_horizon_conditioning(H: Tensor, cfg: Config, target_steps: int, start_channel: int) -> int:
    if not bool(getattr(cfg, "field_horizon_conditioning", False)):
        return start_channel
    feature_count = 4
    if cfg.hidden_dim < start_channel + feature_count:
        raise ValueError(
            "field_horizon_conditioning requires enough hidden channels for four horizon features. "
            f"hidden_dim={cfg.hidden_dim}, required={start_channel + feature_count}."
        )
    features = field_horizon_features(cfg, target_steps, H.device, H.dtype)
    H[:, :, start_channel : start_channel + feature_count] = features.view(1, 1, feature_count)
    return start_channel + feature_count


def diffuse_step(field: Tensor, diffusion_rate: float, decay: float) -> Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=field.dtype,
        device=field.device,
    ).view(1, 1, 3, 3)
    laplacian = F.conv2d(field, kernel, padding=1)
    return (field + float(diffusion_rate) * laplacian) * float(decay)


def make_initial_field(batch_size: int, grid_size: int | tuple[int, int], device: torch.device) -> Tensor:
    if isinstance(grid_size, tuple):
        height, width = int(grid_size[0]), int(grid_size[1])
    else:
        height = width = int(grid_size)
    y_coords = torch.linspace(-1.0, 1.0, height, device=device)
    x_coords = torch.linspace(-1.0, 1.0, width, device=device)
    y, x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    field = torch.zeros(batch_size, 1, height, width, device=device)
    centers = torch.rand(batch_size, 3, 2, device=device) * 1.6 - 0.8
    widths = torch.rand(batch_size, 3, 1, 1, device=device) * 0.20 + 0.08
    amps = torch.rand(batch_size, 3, 1, 1, device=device) * 1.6 - 0.8
    for blob in range(3):
        cx = centers[:, blob, 0].view(batch_size, 1, 1)
        cy = centers[:, blob, 1].view(batch_size, 1, 1)
        radius2 = (x.unsqueeze(0) - cx) ** 2 + (y.unsqueeze(0) - cy) ** 2
        field[:, 0] = field[:, 0] + amps[:, blob] * torch.exp(-radius2 / widths[:, blob].pow(2).clamp_min(1e-6))
    return field


def generate_heat_sequence(cfg: Config, device: torch.device, target_steps: int | None = None) -> Tensor:
    target_steps = int(cfg.field_target_steps) if target_steps is None else int(target_steps)
    total_steps = int(cfg.field_input_steps) + target_steps
    if total_steps < 2:
        raise ValueError("field_input_steps + field_target_steps must be at least 2")
    current = make_initial_field(cfg.batch_size, field_grid_shape(cfg), device)
    frames = [current]
    for _ in range(total_steps - 1):
        current = diffuse_step(current, cfg.field_diffusion_rate, cfg.field_decay)
        frames.append(current)
    return torch.stack(frames, dim=1)


def make_grid_field_adjacency(token_height: int, token_width: int, device: torch.device) -> Tensor:
    n_nodes = token_height * token_width
    adjacency = torch.zeros(n_nodes, n_nodes, device=device)
    for r in range(token_height):
        for c in range(token_width):
            i = r * token_width + c
            adjacency[i, i] = 1.0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < token_height and 0 <= nc < token_width:
                    adjacency[i, nr * token_width + nc] = 1.0
    return adjacency


def make_field_adjacency(cfg: Config, device: torch.device) -> Tensor:
    mode = getattr(cfg, "field_adjacency_mode", "grid")
    token_height, token_width = field_token_shape(cfg)
    n_nodes = token_height * token_width

    def node_index(row: int, col: int) -> int:
        return row * token_width + col

    if mode == "grid":
        return make_grid_field_adjacency(token_height, token_width, device)
    if mode == "moore":
        adjacency = torch.zeros(n_nodes, n_nodes, device=device)
        for r in range(token_height):
            for c in range(token_width):
                i = node_index(r, c)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < token_height and 0 <= nc < token_width:
                            adjacency[i, node_index(nr, nc)] = 1.0
        return adjacency
    if mode == "line":
        adjacency = torch.zeros(n_nodes, n_nodes, device=device)
        for r in range(token_height):
            for c in range(token_width):
                i = node_index(r, c)
                for other_c in range(token_width):
                    adjacency[i, node_index(r, other_c)] = 1.0
                for other_r in range(token_height):
                    adjacency[i, node_index(other_r, c)] = 1.0
        return adjacency
    if mode == "torus":
        adjacency = torch.zeros(n_nodes, n_nodes, device=device)
        for r in range(token_height):
            for c in range(token_width):
                i = node_index(r, c)
                adjacency[i, i] = 1.0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = (r + dr) % token_height
                    nc = (c + dc) % token_width
                    adjacency[i, node_index(nr, nc)] = 1.0
        return adjacency
    if mode == "full":
        return torch.ones(n_nodes, n_nodes, device=device)
    raise ValueError(f"Unsupported field_adjacency_mode: {mode}")


def make_radius_visibility(token_height: int, token_width: int, radius: int, device: torch.device) -> Tensor:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    n_nodes = token_height * token_width
    visibility = torch.zeros(n_nodes, n_nodes, device=device)
    for r in range(token_height):
        for c in range(token_width):
            i = r * token_width + c
            for other_r in range(token_height):
                for other_c in range(token_width):
                    distance = abs(r - other_r) + abs(c - other_c)
                    if distance <= radius:
                        visibility[i, other_r * token_width + other_c] = 1.0
    return visibility


def make_field_input_visibility(cfg: Config, device: torch.device) -> Tensor:
    scope = getattr(cfg, "field_input_scope", "global")
    token_height, token_width = field_token_shape(cfg)
    n_nodes = token_height * token_width
    if scope == "global":
        return torch.ones(n_nodes, n_nodes, device=device)
    if scope in {"local", "radius1"}:
        return make_grid_field_adjacency(token_height, token_width, device)
    if scope == "radius2":
        return make_radius_visibility(token_height, token_width, 2, device)
    if scope == "radius4":
        return make_radius_visibility(token_height, token_width, 4, device)
    raise ValueError(f"Unsupported field_input_scope: {scope}")


def make_field_input_channel_mask(cfg: Config, device: torch.device) -> Tensor:
    mask = torch.zeros(cfg.hidden_dim, device=device)
    channels = max(1, int(getattr(cfg, "field_output_dim", 1)))
    visible_channels = min(cfg.hidden_dim, channels * 2)
    if visible_channels > 0:
        mask[:visible_channels] = 1.0
    return mask


def _normalize_patch_size(patch_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(patch_size, tuple):
        return int(patch_size[0]), int(patch_size[1])
    return int(patch_size), int(patch_size)


def patchify_field(field: Tensor, patch_size: int | tuple[int, int]) -> Tensor:
    if field.ndim != 4:
        raise ValueError(f"Expected field [B,C,H,W], got {tuple(field.shape)}")
    batch_size, channels, height, width = field.shape
    patch_height, patch_width = _normalize_patch_size(patch_size)
    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(f"Expected field divisible by patch_size, got field={tuple(field.shape)} patch={patch_size}")
    patch_rows = height // patch_height
    patch_cols = width // patch_width
    patches = field.view(batch_size, channels, patch_rows, patch_height, patch_cols, patch_width)
    patches = patches.mean(dim=(3, 5))
    return patches.permute(0, 2, 3, 1).reshape(batch_size, patch_rows * patch_cols, channels)


def unpatchify_field(patches: Tensor, patch_size: int | tuple[int, int], grid_size: int | tuple[int, int]) -> Tensor:
    if patches.ndim != 3:
        raise ValueError(f"Expected patches [B,N,C], got {tuple(patches.shape)}")
    batch_size, n_nodes, channels = patches.shape
    patch_height, patch_width = _normalize_patch_size(patch_size)
    if isinstance(grid_size, tuple):
        height, width = int(grid_size[0]), int(grid_size[1])
    else:
        height = width = int(grid_size)
    patch_rows = height // patch_height
    patch_cols = width // patch_width
    if n_nodes != patch_rows * patch_cols:
        raise ValueError(f"Expected {patch_rows * patch_cols} patches for grid_size={grid_size}, got {n_nodes}")
    field = patches.view(batch_size, patch_rows, patch_cols, channels).permute(0, 3, 1, 2)
    return field.repeat_interleave(patch_height, dim=2).repeat_interleave(patch_width, dim=3)


def make_field_batch(cfg: Config, device: torch.device) -> TensorBatch:
    configure_field_nodes(cfg)
    if cfg.field_dataset != "synthetic_heat":
        from wca.data.field.real_cache import make_real_field_batch

        return make_real_field_batch(cfg, device)

    target_steps = choose_field_target_steps(cfg, device)
    sequence = generate_heat_sequence(cfg, device, target_steps=target_steps)
    inputs = sequence[:, : cfg.field_input_steps]
    target_index = int(cfg.field_input_steps) + target_steps - 1
    target = sequence[:, target_index]
    patch_shape = field_patch_shape(cfg)
    current_tokens = patchify_field(inputs[:, -1], patch_shape).squeeze(-1)
    previous_tokens = patchify_field(inputs[:, -2] if cfg.field_input_steps > 1 else inputs[:, -1], patch_shape).squeeze(-1)
    target_tokens = patchify_field(target, patch_shape).squeeze(-1)

    H = torch.zeros(cfg.batch_size, cfg.n_nodes, cfg.hidden_dim, device=device)
    H[:, :, 0] = current_tokens
    if cfg.hidden_dim > 1:
        H[:, :, 1] = previous_tokens
    coord_start = inject_field_horizon_conditioning(H, cfg, target_steps, 2)
    if cfg.hidden_dim > coord_start + 1:
        token_height, token_width = field_token_shape(cfg)
        y_coords = torch.linspace(-1.0, 1.0, token_height, device=device)
        x_coords = torch.linspace(-1.0, 1.0, token_width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        H[:, :, coord_start] = xx.reshape(1, cfg.n_nodes)
        H[:, :, coord_start + 1] = yy.reshape(1, cfg.n_nodes)

    adjacency = make_field_adjacency(cfg, device).unsqueeze(0).expand(cfg.batch_size, cfg.n_nodes, cfg.n_nodes).clone()
    input_visibility = (
        make_field_input_visibility(cfg, device).unsqueeze(0).expand(cfg.batch_size, cfg.n_nodes, cfg.n_nodes).clone()
    )
    input_visibility_channels = make_field_input_channel_mask(cfg, device)
    return {
        "H": H,
        "adjacency": adjacency,
        "input_visibility": input_visibility,
        "input_visibility_channels": input_visibility_channels,
        "target_idx": torch.zeros(cfg.batch_size, dtype=torch.long, device=device),
        "label": target_tokens,
        "field_input": inputs,
        "field_target": target,
        "field_target_index": torch.full((cfg.batch_size,), target_index, dtype=torch.long, device=device),
        "field_target_steps_actual": torch.full((cfg.batch_size,), target_steps, dtype=torch.long, device=device),
        "field_prediction_baseline": current_tokens,
        "field_previous_tokens": previous_tokens,
        "source_sign": torch.zeros(cfg.batch_size, device=device),
        "distractor_sign": torch.zeros(cfg.batch_size, device=device),
        "raw_distance": torch.zeros(cfg.batch_size, device=device),
    }
