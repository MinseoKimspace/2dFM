from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from fm.paths import sample_linear_path


def fm_velocity_loss(model: torch.nn.Module, x1: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    FM objective:
      L = E || v_theta(x_t, t) - (x1 - x0) ||_2^2
    """
    _, x_t, t_now, target_v = sample_linear_path(x1)
    t_start = torch.zeros_like(t_now)
    v_hat = model(x_t, t_start, t_now)
    loss = F.mse_loss(v_hat, target_v)
    metrics = {"loss": float(loss.detach().item()), "v_mse": float(loss.detach().item())}
    return loss, metrics
