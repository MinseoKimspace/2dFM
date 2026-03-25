"""Transformer backbone for FM/iMF vector field prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.time_embed import TimeEmbeddingMLP


@dataclass
class EncoderOutput:
    """Structured output for the encoder stack."""

    tokens: torch.Tensor
    hidden_states: list[torch.Tensor] | None = None


@dataclass
class TransformerOutput:
    """Optional structured return type for the full backbone."""

    sample: torch.Tensor
    tokens: torch.Tensor | None = None
    hidden_states: list[torch.Tensor] | None = None


def _init_linear_conservative(module: nn.Module) -> None:
    """Initialize linear layers with a small standard deviation."""
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(1)
        std = (0.1 / float(fan_in)) ** 0.5
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, cond_dim: int | None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.ada_proj = None

        if cond_dim is not None:
            self.ada_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, 6 * d_model),
            )
            nn.init.zeros_(self.ada_proj[-1].weight)
            nn.init.zeros_(self.ada_proj[-1].bias)

    def forward(self, x:torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_attn = scale_attn = gate_attn = None
        shift_mlp = scale_mlp = gate_mlp = None

        if cond is not None and self.ada_proj is not None:
            shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_proj(cond).chunk   (6, dim=-1)

        h = self.norm1(x)
        if shift_attn is not None:
            h = _modulate(h, shift_attn, scale_attn)

        attn_out = self.self_attn(h, h, h, need_weights=False)[0]

        if gate_attn is not None:
            attn_out = gate_attn.unsqueeze(1) * attn_out
        
        x = x + self.dropout1(attn_out)

        h = self.norm2(x)

        if shift_mlp is not None:
            h = _modulate(h, shift_mlp, scale_mlp)
        
        mlp_out = self.linear2(self.dropout(self.act(self.linear1(h))))
        
        if gate_mlp is not None:
            mlp_out = gate_mlp.unsqueeze(1) * mlp_out
        
        x = x + self.dropout2(mlp_out)

        return x
class VectorFieldTransformer(nn.Module):
    """Patch-based Transformer that predicts FM/iMF vector fields."""

    def __init__(
        self,
        input_dim: int = 784,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 1024,
        patch_size: int = 2,
        image_size: int = 28,
        in_channels: int = 1,
        time_embed_dim: int = 128,
        dropout: float = 0.0,
        variant: str = "fm",
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()

        if variant not in {"fm", "imf"}:
            raise ValueError(f"Unknown variant: {variant}")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if in_channels * image_size * image_size != input_dim:
            raise ValueError(
                "input_dim must match in_channels * image_size * image_size "
                f"(got {input_dim} vs {in_channels}*{image_size}*{image_size})."
            )

        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.num_layers = int(num_layers)
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.patch_dim = int(in_channels * patch_size * patch_size)
        patches_per_side = image_size // patch_size
        self.num_tokens = int(patches_per_side * patches_per_side)
        self.variant = variant

        self.patch_in = nn.Linear(self.patch_dim, model_dim)
        self.token_pos = nn.Parameter(torch.zeros(1, self.num_tokens, model_dim))

        self.start_time_embed = TimeEmbeddingMLP(time_embed_dim)
        self.now_time_embed = TimeEmbeddingMLP(time_embed_dim)
        self.time_fuse = nn.Sequential(
            nn.Linear(2 * time_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList(
            [
                AdaLNTransformerBlock(
                    d_model=model_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    cond_dim=cond_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = None
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.v_head = nn.Linear(model_dim, self.patch_dim)
        self.u_head = nn.Linear(model_dim, self.patch_dim)

        self.apply(_init_linear_conservative)
        nn.init.normal_(self.token_pos, mean=0.0, std=0.02)
        if self.variant == "imf":
            nn.init.normal_(self.u_head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.u_head.bias)

    def prepare_tokens(
        self,
        x_t: torch.Tensor,
        t_start: torch.Tensor,
        t_now: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return token embeddings before the Transformer stack."""
        if t_now is None:
            t_now = t_start
            t_start = torch.zeros_like(t_now)

        batch_size = x_t.size(0)
        x_2d = x_t.view(batch_size, self.in_channels, self.image_size, self.image_size)
        x_tokens = F.unfold(
            x_2d,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        tokens = self.patch_in(x_tokens)

        t_start_emb = self.start_time_embed(t_start)
        t_now_emb = self.now_time_embed(t_now)
        time_cond = self.time_fuse(torch.cat([t_start_emb, t_now_emb], dim=1))
        return tokens + self.token_pos + time_cond.unsqueeze(1)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        return_hidden_states: bool = False,
        start_layer: int = 0,
        end_layer: int | None = None,
        cond: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """Run the encoder stack and optionally collect per-layer hidden states."""
        hidden_states: list[torch.Tensor] | None = [] if return_hidden_states else None
        encoded = tokens

        layers = self.layers[start_layer:end_layer]
        for layer in layers:
            encoded = layer(encoded, cond=cond)
            if hidden_states is not None:
                hidden_states.append(encoded)

        if self.final_norm is not None:
            encoded = self.final_norm(encoded)

        return EncoderOutput(tokens=encoded, hidden_states=hidden_states)

    def decode_tokens(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Project final tokens back to flattened sample space."""
        out = self.v_head(tokens) if self.variant == "fm" else self.u_head(tokens)
        out_cols = out.transpose(1, 2)
        out_2d = F.fold(
            out_cols,
            output_size=(self.image_size, self.image_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return out_2d.reshape(tokens.size(0), self.input_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t_start: torch.Tensor,
        t_now: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_tokens: bool = False,
        return_dict: bool = True,
    ) -> torch.Tensor | TransformerOutput:
        """
        Default behavior returns the flattened sample tensor.

        When `return_hidden_states` or `return_tokens` is enabled, this can
        return a structured `TransformerOutput`.
        """
        tokens = self.prepare_tokens(x_t, t_start, t_now)
        encoder_output = self.encode_tokens(tokens, return_hidden_states=return_hidden_states)
        sample = self.decode_tokens(encoder_output.tokens)

        if not return_hidden_states and not return_tokens:
            return sample

        if not return_dict:
            raise ValueError("Structured transformer outputs require return_dict=True.")

        return TransformerOutput(
            sample=sample,
            tokens=encoder_output.tokens if return_tokens else None,
            hidden_states=encoder_output.hidden_states if return_hidden_states else None,
        )
