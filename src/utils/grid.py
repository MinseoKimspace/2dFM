from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def flat_to_image(x_flat: torch.Tensor) -> torch.Tensor:
    """Convert [B, 784] in [-1, 1] to [B, 1, 28, 28] in [0, 1]."""
    x = x_flat.view(-1, 1, 28, 28)
    return (x.clamp(-1.0, 1.0) + 1.0) * 0.5


def make_grid_from_flat(x_flat: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    images = flat_to_image(x_flat)
    return make_grid(images, nrow=nrow, padding=2)


def save_grid_from_flat(x_flat: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = flat_to_image(x_flat)
    save_image(images, str(path), nrow=nrow, padding=2)
