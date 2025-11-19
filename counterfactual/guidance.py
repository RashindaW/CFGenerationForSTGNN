from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .config import GuidanceConfig


def prepare_forecaster_input(x: torch.Tensor) -> torch.Tensor:
    """Reorder diffusion samples into the (B, C, N, T) layout expected by ST-GNNs."""

    return x.permute(0, 3, 2, 1).contiguous()


def compute_graph_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(dim=-1)
    if adjacency.dim() == 2:
        return torch.diag(degree) - adjacency
    # batched adjacency
    diag = torch.diag_embed(degree)
    return diag - adjacency


def temporal_smoothness(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    diffs = x[:, 1:] - x[:, :-1]
    if mask is not None:
        diffs = diffs * mask[:, 1:]
    return torch.mean(diffs.pow(2))


def spatial_coherence(x: torch.Tensor, laplacian: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    b, t, n, f = x.shape
    values = x if mask is None else x * mask
    x_flat = values.reshape(b * t, n, f)
    if laplacian.dim() == 2:
        lap = laplacian.unsqueeze(0)
    else:
        lap = laplacian
    quad = torch.einsum("bnf,bnm,bmf->", x_flat, lap, x_flat)
    denom = float(b * t * f)
    return quad / max(denom, 1.0)


def control_energy(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    diffs = torch.abs(x[:, 1:] - x[:, :-1])
    if mask is not None:
        diffs = diffs * mask[:, 1:]
    return torch.mean(diffs)


def _broadcast_bounds(bounds: Optional[torch.Tensor | float], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if bounds is None:
        return None
    if isinstance(bounds, torch.Tensor):
        return bounds.to(reference.device, reference.dtype)
    return torch.tensor(bounds, device=reference.device, dtype=reference.dtype)


class ForecastGuidance:
    """Applies masked gradient guidance based on a differentiable ST-GNN forecaster."""

    def __init__(
        self,
        forecaster: nn.Module,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
        adjacency: torch.Tensor,
        config: GuidanceConfig,
        lower_bounds: Optional[torch.Tensor | float] = None,
        upper_bounds: Optional[torch.Tensor | float] = None,
        baseline: Optional[torch.Tensor] = None,
        anchor_weights: Optional[torch.Tensor] = None,
    ) -> None:
        device = next(forecaster.parameters()).device
        self.forecaster = forecaster
        self.target = target.to(device).float()
        self.mask = mask.to(device) if mask is not None else None
        self.config = config
        self.adjacency = adjacency.to(device)
        self.laplacian = compute_graph_laplacian(self.adjacency)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.baseline = baseline.to(device).float().unsqueeze(0) if baseline is not None else None
        if anchor_weights is not None:
            weights = anchor_weights.to(device).float().view(1, 1, -1)
        else:
            weights = None
        self.anchor_weights = weights

        self.forecaster.eval()

    def _apply_bounds(self, x: torch.Tensor) -> torch.Tensor:
        lower = _broadcast_bounds(self.lower_bounds, x)
        upper = _broadcast_bounds(self.upper_bounds, x)
        if lower is not None or upper is not None:
            x = torch.clamp(x, min=lower if lower is not None else None, max=upper if upper is not None else None)
        return x

    def _apply_rate_limit(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.rate_limit is None:
            return x
        diffs = x[:, 1:] - x[:, :-1]
        rate = self.config.rate_limit
        clamped = torch.clamp(diffs, min=-rate, max=rate)
        x_adj = x.clone()
        x_adj[:, 1:] = x_adj[:, :-1] + clamped
        return x_adj

    def _apply_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            return tensor
        return tensor * self.mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.detach()
            x.requires_grad_(True)
            prediction = self.forecaster(prepare_forecaster_input(x))
            loss = F.mse_loss(prediction, self.target)
            if (
                self.baseline is not None
                and self.anchor_weights is not None
                and self.config.anchor_loss_scale > 0
            ):
                anchor_term = (prediction - self.baseline).pow(2)
                anchor_term = anchor_term * self.anchor_weights
                loss = loss + self.config.anchor_loss_scale * anchor_term.mean()
            if self.config.temporal_weight > 0:
                loss = loss + self.config.temporal_weight * temporal_smoothness(x, self.mask)
            if self.config.spatial_weight > 0:
                loss = loss + self.config.spatial_weight * spatial_coherence(x, self.laplacian, self.mask)
            if self.config.control_energy_weight > 0:
                loss = loss + self.config.control_energy_weight * control_energy(x, self.mask)

            grad = torch.autograd.grad(loss, x)[0]
            if self.config.max_grad_norm is not None:
                grad_norm = grad.norm().clamp(min=1e-8)
                max_norm = torch.tensor(self.config.max_grad_norm, device=grad.device, dtype=grad.dtype)
                scale = torch.clamp(max_norm / grad_norm, max=1.0)
                grad = grad * scale
            grad = self._apply_mask(grad)
            x = x - self.config.eta * self.config.lambda_scale * grad
            x = self._apply_bounds(x)
            x = self._apply_rate_limit(x)
            return x.detach()
