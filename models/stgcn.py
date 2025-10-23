from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dense_to_sparse


class TemporalConv(nn.Module):
    """Temporal gated convolution used in the original STGCN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, nodes, channels)
        x = x.permute(0, 3, 2, 1).contiguous()  # -> (batch, channels, nodes, time)
        p = self.conv_1(x)
        q = torch.sigmoid(self.conv_2(x))
        pq = p * q
        h = F.relu(pq + self.conv_3(x))
        h = h.permute(0, 3, 2, 1).contiguous()  # -> (batch, time, nodes, channels)
        return h


class STConv(nn.Module):
    """Spatio-temporal convolution block (Temporal -> Graph -> Temporal)."""

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        k_order: int,
        normalization: str = "sym",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.temporal_conv1 = TemporalConv(in_channels, hidden_channels, kernel_size)
        self.graph_conv = ChebConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=k_order,
            normalization=normalization,
            bias=bias,
        )
        self.temporal_conv2 = TemporalConv(hidden_channels, out_channels, kernel_size)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        t = self.temporal_conv1(x)
        batch_size, seq_len, num_nodes, hidden = t.shape
        t = t.reshape(batch_size * seq_len, num_nodes, hidden)
        out = t.new_zeros(batch_size * seq_len, num_nodes, hidden)
        for idx in range(batch_size * seq_len):
            out[idx] = self.graph_conv(t[idx], edge_index, edge_weight)
        t = out.reshape(batch_size, seq_len, num_nodes, hidden)
        t = F.relu(t)
        t = self.temporal_conv2(t)
        t = t.permute(0, 2, 1, 3).contiguous()
        t = self.batch_norm(t)
        t = t.permute(0, 2, 1, 3).contiguous()
        return t


@dataclass
class STGCNConfig:
    num_nodes: int
    in_channels: int
    hidden_channels: int = 64
    horizon: int = 12
    num_layers: int = 2
    temporal_kernel: int = 3
    k_order: int = 3
    normalization: str = "sym"
    bias: bool = True


class STGCN(nn.Module):
    """Spatio-temporal graph convolutional network following the TGT implementation."""

    def __init__(self, config: STGCNConfig, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.config = config

        edge_index, edge_weight = dense_to_sparse(adjacency)
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        blocks = []
        in_channels = config.in_channels
        for _ in range(config.num_layers):
            blocks.append(
                STConv(
                    num_nodes=config.num_nodes,
                    in_channels=in_channels,
                    hidden_channels=config.hidden_channels,
                    out_channels=config.hidden_channels,
                    kernel_size=config.temporal_kernel,
                    k_order=config.k_order,
                    normalization=config.normalization,
                    bias=config.bias,
                )
            )
            in_channels = config.hidden_channels
        self.blocks = nn.ModuleList(blocks)

        self.final_temporal = TemporalConv(
            in_channels=config.hidden_channels,
            out_channels=config.hidden_channels,
            kernel_size=config.temporal_kernel,
        )
        self.linear = nn.Linear(config.hidden_channels, config.horizon)

    def _resolve_graph(self, adjacency: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if adjacency is None:
            return self.edge_index, self.edge_weight
        edge_index, edge_weight = dense_to_sparse(adjacency)
        return edge_index.to(self.edge_index.device), edge_weight.to(self.edge_weight.device)

    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, features, nodes, steps)
        edge_index, edge_weight = self._resolve_graph(adjacency)
        out = x.permute(0, 3, 2, 1).contiguous()  # -> (batch, time, nodes, channels)
        for block in self.blocks:
            out = block(out, edge_index, edge_weight)
        out = self.final_temporal(out)
        out = out[:, -1, :, :]  # last temporal slice: (batch, num_nodes, hidden)
        out = self.linear(out)  # (batch, num_nodes, horizon)
        return out
