from __future__ import annotations

import torch

from imf.losses import compute_V


@torch.no_grad()
def euler_sample_imf(
    model: torch.nn.Module,
    num_samples: int,
    dim: int,
    device: torch.device,
    nfe: int = 1,
    x_init: torch.Tensor | None = None,
) -> torch.Tensor:
    if nfe <= 0:
        raise ValueError("nfe must be positive.")
    
    x = x_init.clone().to(device) if x_init is not None else torch.randn(num_samples, dim, device=device)
    dt = 1.0 / float(nfe)

    model.eval()
    for i in range(nfe):
        t = torch.full((x.size(0), 1), 1.0 - i / float(nfe), device=device, dtype=x.dtype)
        r = torch.zeros_like(t)
        V = compute_V(model, z=x, r=r, t=t)
        x = x - dt * V
    return x



@torch.no_grad()
def one_step_sample_imf(
    model: torch.nn.Module,
    num_samples: int,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    return euler_sample_imf(
        model=model,
        num_samples=num_samples,
        dim=dim,
        device=device,
        nfe=1,
    )
