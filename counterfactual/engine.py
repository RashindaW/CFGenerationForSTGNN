from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING

import torch

from .diffusion import DiffusionModel

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from .guidance import ForecastGuidance


class DiffusionTrainer:
    """Simple trainer wrapper used for fitting the diffusion prior."""

    def __init__(
        self,
        diffusion: DiffusionModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        adjacency: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None,
    ) -> None:
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
        self.adjacency = adjacency.to(device)
        self.temporal_context = temporal_context.to(device) if temporal_context is not None else None

    def _prepare_batch(self, batch: Iterable[torch.Tensor]) -> torch.Tensor:
        x, *_ = batch
        return x.to(self.device).float()

    def train_epoch(self, dataloader: "DataLoader") -> float:
        self.diffusion.train()
        total_loss = 0.0
        total_steps = 0
        for batch in dataloader:
            x = self._prepare_batch(batch)
            loss = self.diffusion.training_loss(x, adjacency=self.adjacency, temporal_context=self.temporal_context)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_steps += 1
        return total_loss / max(total_steps, 1)

    @torch.no_grad()
    def evaluate_epoch(self, dataloader: "DataLoader") -> float:
        self.diffusion.eval()
        total_loss = 0.0
        total_steps = 0
        for batch in dataloader:
            x = self._prepare_batch(batch)
            loss = self.diffusion.training_loss(x, adjacency=self.adjacency, temporal_context=self.temporal_context)
            total_loss += loss.item()
            total_steps += 1
        return total_loss / max(total_steps, 1)


class CounterfactualGenerator:
    """Samples counterfactual past trajectories from a trained diffusion model."""

    def __init__(
        self,
        diffusion: DiffusionModel,
        adjacency: torch.Tensor,
        device: torch.device,
        temporal_context: Optional[torch.Tensor] = None,
    ) -> None:
        self.diffusion = diffusion
        self.adjacency = adjacency
        self.device = device
        self.temporal_context = temporal_context

    def generate(
        self,
        sample_shape: torch.Size,
        guidance: "ForecastGuidance",
        num_samples: int = 1,
        max_steps: Optional[int] = None,
        warm_start: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        adjacency = self.adjacency.to(self.device)
        temporal_context = self.temporal_context
        if temporal_context is not None:
            temporal_context = temporal_context.to(self.device)
        expanded_shape = torch.Size((num_samples, *sample_shape))
        if warm_start is not None and warm_start.shape != expanded_shape:
            raise ValueError("warm_start shape must match the requested sampling shape.")
        samples = self.diffusion.p_sample_loop(
            expanded_shape,
            adjacency=adjacency,
            guidance=guidance,
            temporal_context=temporal_context,
            max_steps=max_steps,
            initial_x=warm_start,
        )
        return samples
