from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor, nn

from wca.models.field_tokenizers import FieldPatchTokenDecoder, build_field_tokenizer, normalize_token_dim
from wca.models.rws_nca import FullRecursiveWorldStateNCA
from wca.schemas import TensorBatch


def _tokens_for_readout(tokens: Tensor) -> Tensor:
    if tokens.shape[-1] == 1:
        return tokens.squeeze(-1)
    return tokens


def _field_patch_shape(cfg: Any) -> tuple[int, int]:
    patch_height = int(getattr(cfg, "field_patch_height", 0) or getattr(cfg, "field_patch_size", 0))
    patch_width = int(getattr(cfg, "field_patch_width", 0) or getattr(cfg, "field_patch_size", 0))
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError(f"field patch height/width must be positive, got {patch_height}x{patch_width}")
    return patch_height, patch_width


def _field_token_shape(cfg: Any) -> tuple[int, int]:
    height = int(getattr(cfg, "field_grid_height", 0) or getattr(cfg, "field_grid_size", 0))
    width = int(getattr(cfg, "field_grid_width", 0) or getattr(cfg, "field_grid_size", 0))
    patch_height, patch_width = _field_patch_shape(cfg)
    if height <= 0 or width <= 0:
        raise ValueError(f"field grid height/width must be positive, got {height}x{width}")
    if height % patch_height != 0 or width % patch_width != 0:
        raise ValueError(f"field grid {height}x{width} must be divisible by patch {patch_height}x{patch_width}")
    return height // patch_height, width // patch_width


def _max_field_target_steps(cfg: Any) -> int:
    explicit_max = int(getattr(cfg, "field_horizon_max_steps", 0) or 0)
    if explicit_max > 0:
        return explicit_max
    raw_choices = str(getattr(cfg, "field_target_steps_choices", "") or "")
    choices = [int(item.strip()) for item in raw_choices.split(",") if item.strip()]
    return max(choices) if choices else int(getattr(cfg, "field_target_steps", 1))


def _inject_field_horizon_conditioning(H: Tensor, cfg: Any, target_steps: int, start_channel: int) -> int:
    if not bool(getattr(cfg, "field_horizon_conditioning", False)):
        return start_channel
    feature_count = 4
    if H.shape[-1] < start_channel + feature_count:
        raise ValueError(
            "field_horizon_conditioning requires enough hidden channels for four horizon features. "
            f"hidden_dim={H.shape[-1]}, required={start_channel + feature_count}."
        )
    max_steps = max(1, _max_field_target_steps(cfg))
    horizon = torch.tensor(float(target_steps), device=H.device, dtype=H.dtype)
    normalized = horizon / float(max_steps)
    log_scaled = torch.log1p(horizon) / torch.log1p(torch.tensor(float(max_steps), device=H.device, dtype=H.dtype))
    phase = normalized * torch.pi
    features = torch.stack([normalized, torch.sin(phase), torch.cos(phase), log_scaled])
    H[:, :, start_channel : start_channel + feature_count] = features.view(1, 1, feature_count)
    return start_channel + feature_count


class FieldTokenizerWCA(nn.Module):
    """Learnable field-token interface around the protected dense WCA core."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_nodes = int(cfg.n_nodes)
        self.hidden_dim = int(cfg.hidden_dim)
        self.output_dim = int(getattr(cfg, "field_output_dim", 1))
        self.token_dim = normalize_token_dim(self.hidden_dim, int(getattr(cfg, "field_token_dim", 0)))
        self.tokenizer_name = str(getattr(cfg, "field_tokenizer", "patch_mean"))
        self.tokenizer_only = bool(getattr(cfg, "field_tokenizer_only", False))
        if self.tokenizer_name == "patch_mean":
            raise ValueError("FieldTokenizerWCA is only for learnable tokenizers; patch_mean uses the baseline WCA path.")

        self.tokenizer = build_field_tokenizer(cfg)
        self.core = (
            None
            if self.tokenizer_only
            else FullRecursiveWorldStateNCA(
                n_nodes=self.n_nodes,
                hidden_dim=self.hidden_dim,
                edge_dim=int(cfg.edge_dim),
                inner_steps=int(cfg.inner_steps),
                pair_chunk_size=int(cfg.pair_chunk_size),
                output_dim=self.token_dim,
                activation_checkpoint_inner=bool(cfg.activation_checkpoint_inner),
            )
        )
        self.decoder = FieldPatchTokenDecoder(
            self.token_dim,
            self.output_dim,
            width=int(getattr(cfg, "field_decoder_width", 0)),
        )

    def _build_H(self, batch: TensorBatch, encoded_tokens: Tensor, target_steps: int) -> Tensor:
        if encoded_tokens.shape[:2] != (batch["H"].shape[0], self.n_nodes):
            raise ValueError(f"Expected encoded tokens [B,{self.n_nodes},D], got {tuple(encoded_tokens.shape)}")
        if encoded_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {encoded_tokens.shape[-1]}")

        H = torch.zeros(
            encoded_tokens.shape[0],
            self.n_nodes,
            self.hidden_dim,
            device=encoded_tokens.device,
            dtype=encoded_tokens.dtype,
        )
        H[:, :, : self.token_dim] = encoded_tokens
        coord_start = _inject_field_horizon_conditioning(H, self.cfg, target_steps, self.token_dim)
        if self.hidden_dim > coord_start + 1:
            token_height, token_width = _field_token_shape(self.cfg)
            y_coords = torch.linspace(-1.0, 1.0, token_height, device=H.device, dtype=H.dtype)
            x_coords = torch.linspace(-1.0, 1.0, token_width, device=H.device, dtype=H.dtype)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
            H[:, :, coord_start] = xx.reshape(1, self.n_nodes)
            H[:, :, coord_start + 1] = yy.reshape(1, self.n_nodes)
        return H

    def _input_visibility_channels(self, batch: TensorBatch) -> Tensor:
        mask = torch.zeros(self.hidden_dim, device=batch["H"].device, dtype=batch["H"].dtype)
        mask[: self.token_dim] = 1.0
        return mask

    def _field_baseline(self, delta: Tensor, batch: TensorBatch) -> Tensor:
        baseline = batch["field_prediction_baseline"].to(device=delta.device, dtype=delta.dtype)
        previous = batch.get("field_previous_tokens")
        if bool(getattr(self.cfg, "field_tendency_baseline", False)) and isinstance(previous, Tensor):
            previous = previous.to(device=delta.device, dtype=delta.dtype)
            horizon = batch.get("field_target_steps_actual")
            if isinstance(horizon, Tensor):
                horizon = horizon.to(device=delta.device, dtype=delta.dtype).view(-1, 1)
                while horizon.ndim < baseline.ndim:
                    horizon = horizon.unsqueeze(-1)
            else:
                horizon = delta.new_tensor(float(getattr(self.cfg, "field_target_steps", 1)))
            baseline = baseline + float(getattr(self.cfg, "field_tendency_scale", 1.0)) * horizon * (baseline - previous)
        if baseline.ndim == 2 and delta.ndim == 3 and delta.shape[-1] == 1:
            baseline = baseline.unsqueeze(-1)
        return baseline

    def forward(self, batch: TensorBatch, outer_steps: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        field_input = batch["field_input"]
        if field_input.ndim != 5:
            raise ValueError(f"Expected batch['field_input'] [B,T,C,H,W], got {tuple(field_input.shape)}")
        patch_shape = _field_patch_shape(self.cfg)
        encoded_tokens = self.tokenizer(field_input, patch_shape)
        horizon = batch.get("field_target_steps_actual")
        target_steps = int(horizon[0].detach().item()) if isinstance(horizon, Tensor) else int(getattr(self.cfg, "field_target_steps", 1))
        H = self._build_H(batch, encoded_tokens, target_steps)
        visibility_channels = self._input_visibility_channels(batch)
        diagnostics: Dict[str, Tensor] = {
            "field_H": H,
            "field_encoded_tokens": encoded_tokens,
            "field_core_executed": H.new_tensor(0.0),
        }
        if self.core is not None and outer_steps > 0:
            H_final, core_diagnostics = self.core(
                H,
                batch["adjacency"],
                outer_steps,
                input_visibility=batch.get("input_visibility"),
                input_visibility_channels=visibility_channels,
            )
            diagnostics.update(core_diagnostics)
            diagnostics["field_core_executed"] = H.new_tensor(1.0)
            raw_tokens = self.core.predict_all_nodes(H_final)
        else:
            raw_tokens = encoded_tokens
        delta = self.decoder(raw_tokens)
        if bool(getattr(self.cfg, "field_residual_readout", False)):
            prediction = self._field_baseline(delta, batch) + float(getattr(self.cfg, "field_residual_scale", 1.0)) * delta
        else:
            prediction = delta
        prediction = _tokens_for_readout(prediction)
        return prediction, diagnostics

    def parameter_breakdown(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))

        tokenizer_params = count(self.tokenizer)
        core_params = 0 if self.core is None else count(self.core)
        decoder_params = count(self.decoder)
        return {
            "field_tokenizer_params": tokenizer_params,
            "wca_core_params": core_params,
            "field_decoder_params": decoder_params,
            "total_params": tokenizer_params + core_params + decoder_params,
        }
