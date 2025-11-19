from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING

import torch

from .diffusion import DiffusionModel

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from .guidance import ForecastGuidance


class EMA:
    """Exponential Moving Average of model parameters for stable sampling."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None) -> None:
        self.decay = decay
        self.device = device
        self.shadow_params = []
        
        # Create shadow copies of model parameters
        for param in model.parameters():
            if param.requires_grad:
                self.shadow_params.append(param.data.clone().detach())
    
    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA parameters with current model parameters."""
        for shadow_param, param in zip(self.shadow_params, [p for p in model.parameters() if p.requires_grad]):
            shadow_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Copy EMA parameters to model."""
        for shadow_param, param in zip(self.shadow_params, [p for p in model.parameters() if p.requires_grad]):
            param.data.copy_(shadow_param)
    
    def state_dict(self) -> dict:
        """Return state dict for saving."""
        return {
            "decay": self.decay,
            "shadow_params": self.shadow_params,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict."""
        self.decay = state_dict["decay"]
        self.shadow_params = state_dict["shadow_params"]


class DiffusionTrainer:
    """Simple trainer wrapper used for fitting the diffusion prior."""

    def __init__(
        self,
        diffusion: DiffusionModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        adjacency: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None,
        grad_clip: Optional[float] = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
    ) -> None:
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
        self.adjacency = adjacency.to(device)
        self.temporal_context = temporal_context.to(device) if temporal_context is not None else None
        self.grad_clip = grad_clip
        self.use_ema = use_ema
        # EMA should track only the UNet network, not the diffusion wrapper
        self.ema = EMA(diffusion.network, decay=ema_decay, device=device) if use_ema else None

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
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.grad_clip)
            self.optimizer.step()
            
            # Update EMA after each training step (only track network parameters)
            if self.ema is not None:
                self.ema.update(self.diffusion.network)
            
            total_loss += loss.item()
            total_steps += 1
        return total_loss / max(total_steps, 1)

    @torch.no_grad()
    def evaluate_epoch(self, dataloader: "DataLoader") -> float:
        self.diffusion.eval()
        
        # Use EMA parameters for evaluation if available (only for network)
        if self.ema is not None:
            # Store current network parameters
            original_params = [p.data.clone() for p in self.diffusion.network.parameters() if p.requires_grad]
            # Copy EMA parameters to network
            self.ema.copy_to(self.diffusion.network)
        
        total_loss = 0.0
        total_steps = 0
        for batch in dataloader:
            x = self._prepare_batch(batch)
            loss = self.diffusion.training_loss(x, adjacency=self.adjacency, temporal_context=self.temporal_context)
            total_loss += loss.item()
            total_steps += 1
        
        # Restore original network parameters
        if self.ema is not None:
            for param, original_param in zip([p for p in self.diffusion.network.parameters() if p.requires_grad], original_params):
                param.data.copy_(original_param)
        
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
