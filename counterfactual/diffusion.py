from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import DiffusionConfig
from .noise_schedule import build_beta_schedule, prepare_diffusion_terms

try:
    from typing import TYPE_CHECKING
except ImportError:
    TYPE_CHECKING = False

if TYPE_CHECKING:
    from .guidance import ForecastGuidance


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather values from 1-D tensor `a` at indices `t` and reshape to `x_shape`."""

    out = a.gather(0, t)
    return out.view(t.size(0), *([1] * (len(x_shape) - 1)))


class DiffusionModel(nn.Module):
    """DDPM wrapper with guidance-aware sampling utilities."""

    def __init__(self, network: nn.Module, config: DiffusionConfig, gpu_ids: Optional[List[int]] = None) -> None:
        super().__init__()
        self.network = network
        self.device_ids = gpu_ids if gpu_ids and len(gpu_ids) > 1 and torch.cuda.is_available() else None
        self.config = config
        betas = build_beta_schedule(config.beta_schedule, config.timesteps, config.beta_start, config.beta_end)
        terms = prepare_diffusion_terms(betas)
        for name, tensor in terms.items():
            self.register_buffer(name, tensor)
        self.timesteps = config.timesteps

    @staticmethod
    def _prepare_adjacency(adjacency: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        adjacency = adjacency.to(device)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.size(0) == 1 and batch_size > 1:
            adjacency = adjacency.expand(batch_size, -1, -1)
        return adjacency

    @staticmethod
    def _prepare_temporal_context(
        temporal_context: Optional[torch.Tensor], batch_size: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if temporal_context is None:
            return None
        temporal_context = temporal_context.to(device)
        if temporal_context.size(0) == 1 and batch_size > 1:
            temporal_context = temporal_context.expand(batch_size, -1, -1, -1)
        return temporal_context

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        adjacency: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        adjacency = self._prepare_adjacency(adjacency, batch_size, x.device)
        temporal_context = self._prepare_temporal_context(temporal_context, batch_size, x.device)
        if self.device_ids:
            return nn.parallel.data_parallel(
                self.network,
                (x, timesteps, adjacency, temporal_context),
                device_ids=self.device_ids,
            )
        return self.network(x, timesteps, adjacency=adjacency, temporal_context=temporal_context)

    # --------------------- Training utilities --------------------- #
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_recip_alphas_cumprod_t = _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps

    def predict_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alphas_cumprod_t * x0) / torch.clamp(sqrt_one_minus_alphas_cumprod_t, min=1e-8)

    def training_loss(
        self,
        x_start: torch.Tensor,
        adjacency: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x_start.size(0)
        device = x_start.device
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.forward(x_noisy, t, adjacency=adjacency, temporal_context=temporal_context)
        if self.config.loss_type == "l1":
            return F.l1_loss(predicted_noise, noise)
        return F.mse_loss(predicted_noise, noise)

    # --------------------- Sampling utilities --------------------- #
    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        betas_t = _extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t / torch.clamp(sqrt_one_minus_alphas_cumprod_t, min=1e-8) * eps)
        posterior_variance_t = _extract(self.posterior_variance, t, x.shape)
        return model_mean, posterior_variance_t

    def p_sample(
        self,
        x: torch.Tensor,
        timestep_index: int,
        adjacency: torch.Tensor,
        guidance: Optional["ForecastGuidance"] = None,
        temporal_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = x.size(0)
        t = torch.full((batch,), timestep_index, device=x.device, dtype=torch.long)
        eps = self.forward(x, t, adjacency=adjacency, temporal_context=temporal_context)
        x0 = self.predict_x0_from_eps(x, t, eps)
        if guidance is not None:
            x0 = guidance(x0)
            if timestep_index > 0:
                noise = torch.randn_like(x)
                x = self.q_sample(x0, t, noise)
            else:
                x = x0
            eps = self.predict_eps_from_x0(x, t, x0)
        model_mean, posterior_variance = self.p_mean_variance(x, t, eps)
        if timestep_index == 0:
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance) * noise

    def _build_reverse_schedule(self, max_steps: Optional[int]) -> List[int]:
        if max_steps is None or max_steps >= self.timesteps:
            return list(range(self.timesteps - 1, -1, -1))
        linspace = torch.linspace(self.timesteps - 1, 0, max_steps, dtype=torch.float32)
        steps = torch.unique_consecutive(linspace.round().long()).tolist()
        if steps[-1] != 0:
            steps.append(0)
        return steps

    def p_sample_loop(
        self,
        shape: torch.Size,
        adjacency: torch.Tensor,
        guidance: Optional["ForecastGuidance"] = None,
        temporal_context: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        initial_x: Optional[torch.Tensor] = None,
        progress_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        device = self.betas.device
        if initial_x is not None:
            x = initial_x.to(device)
        else:
            x = torch.randn(shape, device=device)
        for idx, timestep in enumerate(self._build_reverse_schedule(max_steps)):
            x = self.p_sample(x, timestep, adjacency=adjacency, guidance=guidance, temporal_context=temporal_context)
            if progress_callback is not None:
                progress_callback(idx, x)
        return x
