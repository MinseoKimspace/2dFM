from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import torch

from imf.paths import linear_path, sample_time_pair
from torch.nn.attention import sdpa_kernel, SDPBackend

def imf_velocity_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch = x1.size(0)

    r, t = sample_time_pair(batch=batch, device=x1.device, dtype=x1.dtype)
    e = torch.randn_like(x1)

    z = linear_path(x1, e, t)

    V = compute_V(model=model, z=z, r=r, t=t)
    
    loss = torch.nn.functional.mse_loss(V, e - x1)
    metrics = {"loss": float(loss.detach().item()), "v_mse": float(loss.detach().item())}

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