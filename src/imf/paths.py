from __future__ import annotations

import torch


def _sample_logit_normal(
    batch: int,
    device,
    dtype,
    mu: float = -0.4,
    sigma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    z = torch.randn(batch, 1, device=device, dtype=dtype) * sigma + mu
    u = torch.sigmoid(z)
    # eps < u < 1-eps
    return u.clamp(min=eps, max=1.0 - eps)


def sample_time_pair(
    batch: int,
    device,
    dtype,
    mu: float = -0.4,
    sigma: float = 1.0,
    neq_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:

    t1 = _sample_logit_normal(batch=batch, device=device, dtype=dtype, mu=mu, sigma=sigma)
    t2 = _sample_logit_normal(batch=batch, device=device, dtype=dtype, mu=mu, sigma=sigma)

    r = torch.minimum(t1, t2)
    t = torch.maximum(t1, t2)

    eq_ratio = 1.0 - float(neq_ratio)
    if eq_ratio > 0.0:
        eq_mask = torch.rand(batch, 1, device=device) < eq_ratio
        r = torch.where(eq_mask, t, r)

    return r, t


def linear_path(x: torch.Tensor, e: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (1.0 - t) * x + t * e

