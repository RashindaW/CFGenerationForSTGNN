from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DiffusionConfig:
    """Configuration for the denoising diffusion prior."""

    timesteps: int = 1000
    beta_schedule: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    base_channels: int = 64
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    time_embedding_dim: int = 256
    dropout: float = 0.1
    loss_type: str = "l2"


@dataclass
class GuidanceConfig:
    """Hyper-parameters for forecast-aligned gradient guidance."""

    lambda_scale: float = 1.0
    eta: float = 0.05
    temporal_weight: float = 1e-3
    spatial_weight: float = 1e-3
    control_energy_weight: float = 0.0
    rate_limit: Optional[float] = None
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
    mask_strategy: str = "controls_only"
    max_grad_norm: Optional[float] = 10.0


@dataclass
class CounterfactualConfig:
    """Sampling options for generating counterfactual past trajectories."""

    samples_per_case: int = 10
    max_reverse_steps: Optional[int] = None
    ddim_eta: float = 0.0
    warm_start_noise_steps: int = 0
    lambda_schedule: Tuple[float, float] = (0.5, 5.0)
