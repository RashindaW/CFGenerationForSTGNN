from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Standard transformer-style sinusoidal embedding for diffusion step indices."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = timesteps.device
        exponent = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -exponent)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        return embeddings


class GraphConvolution(nn.Module):
    """Simple graph convolution via adjacency propagation + channel mixing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N), adjacency: (N, N) or (B, N, N)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if adjacency.size(0) == 1 and x.size(0) > 1:
            adjacency = adjacency.expand(x.size(0), -1, -1)

        b, c, t, n = x.shape
        x_perm = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        neighbor_sum = torch.einsum("btnc,bnm->btmc", x_perm, adjacency)
        out = self.linear(neighbor_sum)
        return out.permute(0, 3, 1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.graph_conv = GraphConvolution(out_channels)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.dropout = nn.Dropout(dropout)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        if time_emb is not None:
            time_cond = self.time_mlp(time_emb).view(time_emb.size(0), -1, 1, 1)
            h = h + time_cond
        h = self.graph_conv(h, adjacency)
        h = self.norm1(h)
        h = torch.nn.functional.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = torch.nn.functional.silu(h)
        return h + self.residual(x)


def temporal_downsample(x: torch.Tensor) -> torch.Tensor:
    if x.size(2) % 2 == 1:
        x = torch.nn.functional.pad(x, (0, 0, 0, 1))
    return torch.nn.functional.avg_pool2d(x, kernel_size=(2, 1), stride=(2, 1))


def temporal_upsample(x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.block1 = ResidualBlock(in_channels, out_channels, time_embedding_dim, dropout)
        self.block2 = ResidualBlock(out_channels, out_channels, time_embedding_dim, dropout)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.block1(x, time_emb, adjacency)
        h = self.block2(h, time_emb, adjacency)
        skip = h
        down = temporal_downsample(h)
        return down, skip


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.block1 = ResidualBlock(in_channels + skip_channels, out_channels, time_embedding_dim, dropout)
        self.block2 = ResidualBlock(out_channels, out_channels, time_embedding_dim, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = temporal_upsample(x, target_size=skip.shape[-2:])
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, time_emb, adjacency)
        x = self.block2(x, time_emb, adjacency)
        return x


class SpatioTemporalUNet(nn.Module):
    """A compact U-Net operating on (time × nodes) grids with graph convolutions."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        channel_multipliers: Iterable[int] = (1, 2, 4),
        time_embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )

        self.input_projection = nn.Conv2d(in_channels, base_channels, kernel_size=1)

        downs = []
        skips = []
        prev_channels = base_channels
        for multiplier in channel_multipliers:
            out_channels = base_channels * multiplier
            downs.append(DownsampleBlock(prev_channels, out_channels, time_embedding_dim, dropout))
            skips.append(out_channels)
            prev_channels = out_channels
        self.downs = nn.ModuleList(downs)
        self.skip_channels = skips

        self.mid_block1 = ResidualBlock(prev_channels, prev_channels, time_embedding_dim, dropout)
        self.mid_block2 = ResidualBlock(prev_channels, prev_channels, time_embedding_dim, dropout)

        ups = []
        for skip_channels in reversed(self.skip_channels):
            ups.append(UpsampleBlock(prev_channels, skip_channels, skip_channels, time_embedding_dim, dropout))
            prev_channels = skip_channels
        self.ups = nn.ModuleList(ups)

        self.output_projection = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        adjacency: torch.Tensor,
        temporal_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, F) noisy past windows.
            timesteps: (B,) diffusion step indices.
            adjacency: (N, N) or (B, N, N) graph adjacency.
            temporal_context: optional additive context broadcastable to x.
        """

        if temporal_context is not None:
            x = x + temporal_context

        h = x.permute(0, 3, 1, 2).contiguous()  # (B, F, T, N)
        h = self.input_projection(h)
        time_emb = self.time_embed(timesteps)

        skips = []
        current = h
        for down in self.downs:
            current, skip = down(current, time_emb, adjacency)
            skips.append(skip)

        current = self.mid_block1(current, time_emb, adjacency)
        current = self.mid_block2(current, time_emb, adjacency)

        for up, skip in zip(self.ups, reversed(skips)):
            current = up(current, skip, time_emb, adjacency)

        out = self.output_projection(current)
        out = out.permute(0, 2, 3, 1).contiguous()  # back to (B, T, N, F)
        return out
