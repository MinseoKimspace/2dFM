from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from imf.paths import linear_path, sample_time_pair
from models.pooled_transformer import collapse_regularization_loss, semantic_consistency_loss
from torch.nn.attention import SDPBackend, sdpa_kernel


def _flatten_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1)


def _cosine_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = _flatten_batch(pred)
    target_flat = _flatten_batch(target)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8)
    return 1.0 - cosine.mean()


def _norm_ratio(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_norm = _flatten_batch(pred).norm(dim=1)
    target_norm = _flatten_batch(target).norm(dim=1).clamp_min(eps)
    return (pred_norm / target_norm).mean()


def _mean_sample_norm(x: torch.Tensor) -> torch.Tensor:
    return _flatten_batch(x).norm(dim=1).mean()


def imf_velocity_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
    semantic_weight: float = 0.0,
    collapse_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch = x1.size(0)

    r, t = sample_time_pair(batch=batch, device=x1.device, dtype=x1.dtype)
    e = torch.randn_like(x1)
    z = linear_path(x1, e, t)
    target = e - x1

    V, u, correction = compute_V(model=model, z=z, r=r, t=t, return_parts=True)

    base_loss = F.mse_loss(V, target)
    loss = base_loss
    sc_loss = torch.zeros((), device=x1.device, dtype=x1.dtype)
    cr_loss = torch.zeros((), device=x1.device, dtype=x1.dtype)

    if hasattr(model, "forward_with_aux") and (semantic_weight > 0 or collapse_weight > 0):
        aux = model.forward_with_aux(z, r, t)
        sc_loss = semantic_consistency_loss(aux.early_pred, aux.late_shared)
        cr_loss = collapse_regularization_loss(aux.early_shared, aux.late_shared)
        loss = base_loss + semantic_weight * sc_loss + collapse_weight * cr_loss

    with torch.no_grad():
        u_mse = F.mse_loss(u, target)
        u_cos_err = _cosine_error(u, target)
        u_norm_ratio = _norm_ratio(u, target)
        V_cos_err = _cosine_error(V, target)
        V_norm_ratio = _norm_ratio(V, target)
        corr_norm_mean = _mean_sample_norm(correction)
        corr_ratio = corr_norm_mean / _mean_sample_norm(u).clamp_min(1e-8)
        delta_mean = (t - r).mean()

    metrics = {
        "loss": float(loss.detach().item()),
        "v_mse": float(base_loss.detach().item()),
        "semantic_consistency_loss": float(sc_loss.detach().item()),
        "collapse_regularization_loss": float(cr_loss.detach().item()),
        "u_mse": float(u_mse.item()),
        "u_cos_err": float(u_cos_err.item()),
        "u_norm_ratio": float(u_norm_ratio.item()),
        "V_cos_err": float(V_cos_err.item()),
        "V_norm_ratio": float(V_norm_ratio.item()),
        "corr_norm_mean": float(corr_norm_mean.item()),
        "corr_ratio": float(corr_ratio.item()),
        "delta_mean": float(delta_mean.item()),
    }
    return loss, metrics


def compute_V(
    model: torch.nn.Module,
    z: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    return_parts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        correction = (t - r) * dudt.to(dtype=u.dtype).detach()
        V = u + correction
        if return_parts:
            return V, u, correction
        return V
