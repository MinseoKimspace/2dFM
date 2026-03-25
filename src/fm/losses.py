from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from fm.paths import sample_linear_path
from models.pooled_transformer import semantic_consistency_loss, collapse_regularization_loss

def fm_velocity_loss(model: torch.nn.Module, x1: torch.Tensor, w_sc: float = 0.2, w_cr: float = 0.2) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    FM objective:
      L = E || v_theta(x_t, t) - (x1 - x0) ||_2^2
    """
    _, x_t, t_now, target_v = sample_linear_path(x1)
    t_start = torch.zeros_like(t_now)
    sc_loss = torch.zeros((), device=x1.device, dtype=x1.dtype)
    cr_loss = torch.zeros((), device=x1.device, dtype=x1.dtype)

    if hasattr(model, "forward_with_aux"):
        output = model.forward_with_aux(x_t, t_start, t_now)
        pred = output.sample
        base_loss = F.mse_loss(pred, target_v)
        sc_loss = semantic_consistency_loss(output.early_pred, output.late_shared)
        cr_loss = collapse_regularization_loss(output.early_shared, output.late_shared)
        loss = base_loss + w_sc * sc_loss + w_cr * cr_loss
    else:
        v_hat = model(x_t, t_start, t_now)
        base_loss = F.mse_loss(v_hat, target_v)
        loss = base_loss

    metrics = {
        "loss": float(loss.detach().item()),
        "v_mse": float(base_loss.detach().item()),
        "semantic_consistency_loss": float(sc_loss.detach().item()),
        "collapse_regularization_loss": float(cr_loss.detach().item()),
    }
    return loss, metrics
