from __future__ import annotations

import torch


@torch.no_grad()
def euler_sample_fm(
    model: torch.nn.Module,
    num_samples: int,
    dim: int,
    device: torch.device,
    nfe: int = 20,
    x_init: torch.Tensor | None = None,
) -> torch.Tensor:
    if nfe <= 0:
        raise ValueError("nfe must be positive.")

    x = x_init.clone().to(device) if x_init is not None else torch.randn(num_samples, dim, device=device)
    dt = 1.0 / float(nfe)

    model.eval()
    for i in range(nfe):
        t_now = torch.full((x.size(0), 1), i / float(nfe), device=device, dtype=x.dtype)
        t_start = torch.zeros_like(t_now)
        v_hat = model(x, t_start, t_now)
        x = x + dt * v_hat
    return x
