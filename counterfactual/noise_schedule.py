from __future__ import annotations

import math
from typing import Literal

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Create a cosine-based beta schedule as proposed in Nichol & Dhariwal (2021)."""

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, min=1e-4, max=0.999)


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def build_beta_schedule(
    schedule: Literal["linear", "cosine"],
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    if schedule == "cosine":
        return cosine_beta_schedule(timesteps)
    if schedule == "linear":
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    raise ValueError(f"Unknown beta schedule: {schedule}")


def prepare_diffusion_terms(betas: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pre-compute diffusion coefficients used throughout DDPM training/sampling."""

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(torch.clamp(1.0 / alphas_cumprod - 1.0, min=1e-12))

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_variance = torch.clip(posterior_variance, min=1e-20)
    posterior_log_variance_clipped = torch.log(posterior_variance)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
    }
