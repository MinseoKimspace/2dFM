from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import torch

from imf.paths import linear_path, sample_time_pair
from models.pooled_transformer import collapse_regularization_loss, semantic_consistency_loss
from torch.nn.attention import SDPBackend, sdpa_kernel


def imf_velocity_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
    semantic_weight=0.0,
    collapse_weight=0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch = x1.size(0)

    r, t = sample_time_pair(batch=batch, device=x1.device, dtype=x1.dtype)
    e = torch.randn_like(x1)
    z = linear_path(x1, e, t)

    V = compute_V(model=model, z=z, r=r, t=t)

    base_loss = torch.nn.functional.mse_loss(V, e - x1)
    loss = base_loss
    sc_loss = torch.zeros((), device=x1.device, dtype=x1.dtype)
    cr_loss = torch.zeros((), device=x1.device, dtype=x1.dtype)

    if hasattr(model, "forward_with_aux") and (semantic_weight > 0 or collapse_weight > 0):
        aux = model.forward_with_aux(z, r, t)
        sc_loss = semantic_consistency_loss(aux.early_pred, aux.late_shared)
        cr_loss = collapse_regularization_loss(aux.early_shared, aux.late_shared)
        loss = base_loss + semantic_weight * sc_loss + collapse_weight * cr_loss

    metrics = {
        "loss": float(loss.detach().item()),
        "v_mse": float(base_loss.detach().item()),
        "semantic_consistency_loss": float(sc_loss.detach().item()),
        "collapse_regularization_loss": float(cr_loss.detach().item()),
    }
    return loss, metrics


def compute_V(model, z, r, t):
    with sdpa_kernel(SDPBackend.MATH):
        # Keep JVP in FP32 to avoid AMP + JVP graph/grad instability.
        jvp_ctx = torch.amp.autocast(device_type="cuda", enabled=False) if z.is_cuda else nullcontext()
        with jvp_ctx:
            z_f32 = z.float()
            r_f32 = r.float()
            t_f32 = t.float()

            with torch.no_grad():
                v = model(z_f32, t_f32, t_f32)
                tangent_r = torch.zeros_like(r_f32)
                tangent_t = torch.ones_like(t_f32)
                _, dudt = torch.autograd.functional.jvp(
                    model,
                    (z_f32, r_f32, t_f32),
                    (v, tangent_r, tangent_t),
                    create_graph=False,
                )

        u = model(z, r, t)
        V = u + (t - r) * dudt.to(dtype=u.dtype).detach()
        return V
