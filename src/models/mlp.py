from __future__ import annotations

import torch
import torch.nn as nn

from models.time_embed import TimeEmbeddingMLP


def _init_linear_conservative(module: nn.Module) -> None:
    """Linear layers are initialized conservatively for stable early training."""
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(1)
        std = (0.1 / float(fan_in)) ** 0.5
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VectorFieldMLP(nn.Module):
    """MLP that predicts FM v or iMF u."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 512,
        num_layers: int = 3,
        time_embed_dim: int = 128,
        dropout: float = 0.0,
        variant: str = "fm",
        position_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if variant not in {"fm", "imf"}:
            raise ValueError(f"Unknown variant: {variant}")

        self.variant = variant
        pos_dim = int(position_embed_dim) if position_embed_dim is not None else int(input_dim)

        self.position_embed = nn.Sequential(
            nn.Linear(input_dim, pos_dim),
            nn.SiLU(),
            nn.Linear(pos_dim, pos_dim),
        )
        self.start_time_embed = TimeEmbeddingMLP(time_embed_dim)
        self.now_time_embed = TimeEmbeddingMLP(time_embed_dim)

        layers = []
        in_dim = pos_dim + time_embed_dim + time_embed_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.v_head = nn.Linear(hidden_dim, input_dim)
        self.u_head = nn.Linear(hidden_dim, input_dim)
        self.apply(_init_linear_conservative)
        if self.variant == "imf":
            nn.init.normal_(self.u_head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.u_head.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t_start: torch.Tensor,
        t_now: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if t_now is None:
            t_now = t_start
            t_start = torch.zeros_like(t_now)

        x_emb = self.position_embed(x_t)
        t_start_emb = self.start_time_embed(t_start)
        t_now_emb = self.now_time_embed(t_now)
        h = torch.cat([x_emb, t_start_emb, t_now_emb], dim=1)
        h = self.backbone(h)

        _ = return_dict

        if self.variant == "fm":
            return self.v_head(h)
        return self.u_head(h)
