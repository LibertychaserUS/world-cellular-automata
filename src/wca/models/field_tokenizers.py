from __future__ import annotations

import torch
from torch import Tensor, nn

def normalize_token_dim(hidden_dim: int, requested: int = 0) -> int:
    token_dim = int(requested or max(8, min(int(hidden_dim) // 2, int(hidden_dim) - 6)))
    if token_dim <= 0:
        raise ValueError(f"field token_dim must be positive, got {token_dim}")
    if token_dim >= int(hidden_dim):
        raise ValueError(f"field token_dim={token_dim} must be smaller than hidden_dim={hidden_dim}")
    return token_dim


def extract_field_patches(field: Tensor, patch_size: int | tuple[int, int]) -> Tensor:
    """Extract raw field patches as [B,N,C,Ph,Pw] without averaging."""

    if field.ndim != 4:
        raise ValueError(f"Expected field [B,C,H,W], got {tuple(field.shape)}")
    batch_size, channels, height, width = field.shape
    if isinstance(patch_size, tuple):
        patch_height, patch_width = int(patch_size[0]), int(patch_size[1])
    else:
        patch_height = patch_width = int(patch_size)
    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(f"Expected field divisible by patch_size, got field={tuple(field.shape)} patch={patch_size}")
    patch_rows = height // patch_height
    patch_cols = width // patch_width
    patches = field.view(batch_size, channels, patch_rows, patch_height, patch_cols, patch_width)
    return patches.permute(0, 2, 4, 1, 3, 5).reshape(
        batch_size,
        patch_rows * patch_cols,
        channels,
        patch_height,
        patch_width,
    )


def extract_field_sequence_patches(field_input: Tensor, patch_size: int | tuple[int, int]) -> Tensor:
    """Extract input-window patches as [B,N,T*C,Ph,Pw]."""

    if field_input.ndim != 5:
        raise ValueError(f"Expected field_input [B,T,C,H,W], got {tuple(field_input.shape)}")
    batch_size, steps, channels, height, width = field_input.shape
    flattened = field_input.reshape(batch_size, steps * channels, height, width)
    return extract_field_patches(flattened, patch_size)


class ConvStemTokenizer(nn.Module):
    """Patch-local convolutional encoder for field windows."""

    def __init__(self, input_steps: int, channels: int, token_dim: int, width: int = 0) -> None:
        super().__init__()
        hidden = int(width or max(16, token_dim))
        in_channels = int(input_steps) * int(channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, token_dim),
        )

    def forward(self, field_input: Tensor, patch_size: int | tuple[int, int]) -> Tensor:
        patches = extract_field_sequence_patches(field_input, patch_size)
        batch_size, n_nodes, channels, patch_height, patch_width = patches.shape
        encoded = self.encoder(patches.reshape(batch_size * n_nodes, channels, patch_height, patch_width))
        return encoded.view(batch_size, n_nodes, -1)


class MLPStemTokenizer(nn.Module):
    """Flattened patch encoder used to test learnability without convolutional locality."""

    def __init__(
        self,
        input_steps: int,
        channels: int,
        patch_size: int | tuple[int, int],
        token_dim: int,
        width: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(patch_size, tuple):
            patch_height, patch_width = int(patch_size[0]), int(patch_size[1])
        else:
            patch_height = patch_width = int(patch_size)
        input_dim = int(input_steps) * int(channels) * patch_height * patch_width
        hidden = int(width or max(32, token_dim * 2))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
        )

    def forward(self, field_input: Tensor, patch_size: int | tuple[int, int]) -> Tensor:
        patches = extract_field_sequence_patches(field_input, patch_size)
        batch_size, n_nodes, channels, patch_height, patch_width = patches.shape
        encoded = self.encoder(patches.reshape(batch_size * n_nodes, channels * patch_height * patch_width))
        return encoded.view(batch_size, n_nodes, -1)


class FieldPatchTokenDecoder(nn.Module):
    """Decode WCA readout tokens back to node-wise field channels."""

    def __init__(self, token_dim: int, output_dim: int, width: int = 0) -> None:
        super().__init__()
        hidden = int(width or max(16, token_dim))
        self.decoder = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.decoder(tokens)


def build_field_tokenizer(cfg: object) -> nn.Module:
    tokenizer = str(getattr(cfg, "field_tokenizer", "patch_mean"))
    token_dim = normalize_token_dim(int(getattr(cfg, "hidden_dim")), int(getattr(cfg, "field_token_dim", 0)))
    channels = int(getattr(cfg, "field_output_dim", 1))
    input_steps = int(getattr(cfg, "field_input_steps", 1))
    width = int(getattr(cfg, "field_tokenizer_width", 0))
    if tokenizer == "conv_stem":
        return ConvStemTokenizer(input_steps, channels, token_dim, width=width)
    if tokenizer == "mlp_stem":
        return MLPStemTokenizer(input_steps, channels, _field_patch_shape(cfg), token_dim, width=width)
    if tokenizer == "native_cell_state":
        raise NotImplementedError(
            "native_cell_state is a planned V25b+ branch. Use conv_stem/mlp_stem first to isolate tokenizer effects."
        )
    raise ValueError(f"Unsupported learnable field_tokenizer={tokenizer!r}")


def _field_patch_shape(cfg: object) -> tuple[int, int]:
    patch_height = int(getattr(cfg, "field_patch_height", 0) or getattr(cfg, "field_patch_size", 0))
    patch_width = int(getattr(cfg, "field_patch_width", 0) or getattr(cfg, "field_patch_size", 0))
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"field patch height/width must be positive, got {patch_height}x{patch_width}")
    return patch_height, patch_width
