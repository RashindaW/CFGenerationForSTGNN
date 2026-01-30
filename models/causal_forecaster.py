from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation {name}")


def normalize_adjacency(adj: torch.Tensor, mode: str = "sym", add_self_loops: bool = True) -> torch.Tensor:
    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError("adjacency must have shape (N, N)")
    adj = adj.float()
    if add_self_loops:
        eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        adj = adj + eye
    degree = adj.sum(dim=1)
    if mode == "sym":
        inv_sqrt = degree.clamp_min(1e-8).pow(-0.5)
        norm = inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)
        return norm
    if mode == "row":
        inv = degree.clamp_min(1e-8).pow(-1.0)
        return inv.unsqueeze(1) * adj
    raise ValueError(f"Unsupported adjacency normalization {mode}")


class LSTMTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lag, nodes, features)
        batch_size, lag, num_nodes, feat_dim = x.shape
        x = x.reshape(batch_size * num_nodes, lag, feat_dim)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        return h_last.reshape(batch_size, num_nodes, -1)


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.activation = _parse_activation(activation)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, steps)
        padding = (self.kernel_size - 1) * self.dilation
        out = F.pad(x, (padding, 0))
        out = self.conv(out)
        out = self.activation(out)
        out = self.dropout(out)
        residual = x if self.residual is None else self.residual(x)
        return out + residual


class TCNTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dilation_base: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = input_dim
        for layer_idx in range(num_layers):
            dilation = dilation_base**layer_idx
            layers.append(
                TemporalConvBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation,
                )
            )
            in_channels = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lag, nodes, features)
        batch_size, lag, num_nodes, feat_dim = x.shape
        x = x.reshape(batch_size * num_nodes, lag, feat_dim).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = x[..., -1]
        return x.reshape(batch_size, num_nodes, -1)


class TransformerTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_lag: int,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional = nn.Parameter(torch.zeros(1, max_lag, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lag, nodes, features)
        batch_size, lag, num_nodes, feat_dim = x.shape
        if lag > self.positional.size(1):
            raise ValueError(f"lag {lag} exceeds positional embedding length {self.positional.size(1)}")
        x = x.reshape(batch_size * num_nodes, lag, feat_dim)
        x = self.input_proj(x)
        x = x + self.positional[:, :lag, :]
        x = self.encoder(x)
        x = x[:, -1, :]
        return x.reshape(batch_size, num_nodes, -1)


def build_temporal_encoder(
    name: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    kernel_size: int,
    dilation_base: int,
    activation: str,
    num_heads: int,
    max_lag: int,
) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMTemporalEncoder(input_dim, hidden_dim, num_layers, dropout)
    if name == "tcn":
        return TCNTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
            activation=activation,
        )
    if name == "transformer":
        return TransformerTemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_lag=max_lag,
        )
    raise ValueError(f"Unsupported temporal encoder {name}")


class SpatialGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(h)
        return torch.matmul(adj, support)


class SpatialGNNStack(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        layers = []
        norms = []
        current_dim = in_dim
        for _ in range(num_layers):
            layers.append(SpatialGCNLayer(current_dim, hidden_dim))
            norms.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)
        self.activation = _parse_activation(activation)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = h
        for layer, norm in zip(self.layers, self.norms):
            residual = out
            out = layer(out, adj)
            out = norm(out)
            out = self.activation(out)
            out = self.dropout(out)
            if out.shape == residual.shape:
                out = out + residual
        return out


class CausalCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target: torch.Tensor, source: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # target: (batch, N_t, D), source: (batch, N_s, D), adj: (N_s, N_t)
        batch_size, num_t, _ = target.shape
        num_s = source.size(1)
        q = self.q_proj(target).view(batch_size, num_t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(source).view(batch_size, num_s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(source).view(batch_size, num_s, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask: mask positions where NO edge exists
        adj_t = adj.T
        mask = adj_t <= 0  # True where no edge (correct)

        # For isolated nodes (no incoming edges), ALL positions get masked
        # → softmax produces uniform tiny values, which is correct behavior
        # The CausalFusion residual (out = h_manip + attn_out) preserves information

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_t, self.embed_dim)
        out = self.out_proj(out)

        # Note: We no longer zero out isolated nodes - the CausalFusion residual
        # connection (out = h_manip + attn_out) preserves learned representations
        return out


class CausalFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, rounds: int) -> None:
        super().__init__()
        self.rounds = rounds
        self.attn = CausalCrossAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h_control: torch.Tensor, h_manip: torch.Tensor, adj_cm: torch.Tensor) -> torch.Tensor:
        out = h_manip
        for _ in range(self.rounds):
            attn_out = self.attn(out, h_control, adj_cm)
            out = self.norm(out + attn_out)
        return out


class PredictionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        layers = []
        current_dim = hidden_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(_parse_activation(activation))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


@dataclass
class CausalForecasterConfig:
    num_nodes: int
    in_channels: int
    hidden_dim: int
    horizon: int
    lag: int
    control_indices: List[int]
    manipulated_indices: List[int]
    target_indices: List[int]
    temporal_encoder: str = "lstm"
    temporal_layers: int = 1
    temporal_dropout: float = 0.1
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    spatial_layers: int = 2
    spatial_dropout: float = 0.1
    spatial_norm: str = "row"
    attention_heads: int = 4
    attention_dropout: float = 0.1
    fusion_rounds: int = 1
    decoder_layers: int = 2
    decoder_dropout: float = 0.1
    activation: str = "relu"
    target_channel: int = 0
    control_last_weight: Optional[float] = None  # Percentage (0-100) of weight for last lag timestep of control nodes
    use_residual: bool = True  # If True, use residual prediction (y = last_value + delta); if False, direct prediction


class CausalDualStreamForecaster(nn.Module):
    def __init__(self, config: CausalForecasterConfig, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        self.num_nodes = config.num_nodes
        self.horizon = config.horizon
        self.target_channel = config.target_channel

        device = adjacency.device
        control_idx = torch.tensor(config.control_indices, dtype=torch.long, device=device)
        manip_idx = torch.tensor(config.manipulated_indices, dtype=torch.long, device=device)
        target_idx = torch.tensor(config.target_indices, dtype=torch.long, device=device)
        self.register_buffer("control_indices", control_idx)
        self.register_buffer("manipulated_indices", manip_idx)
        self.register_buffer("target_indices", target_idx)

        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("adjacency must have shape (N, N)")
        if adjacency.size(0) != config.num_nodes:
            raise ValueError("adjacency size does not match num_nodes")
        adj = adjacency.float()
        adj_cm = adj.index_select(0, control_idx).index_select(1, manip_idx)
        adj_mm = adj.index_select(0, manip_idx).index_select(1, manip_idx)
        self.register_buffer("adj_cm", adj_cm)
        self.register_buffer("adj_mm", adj_mm)
        self.register_buffer("adj_mm_norm", normalize_adjacency(adj_mm, mode=config.spatial_norm, add_self_loops=True))

        # Build control lag weights if specified
        control_lag_weights = self._build_control_lag_weights(config.lag, config.control_last_weight, device)
        if control_lag_weights is not None:
            self.register_buffer("control_lag_weights", control_lag_weights)
        else:
            self.control_lag_weights = None

        self.control_encoder = build_temporal_encoder(
            name=config.temporal_encoder,
            input_dim=config.in_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.temporal_layers,
            dropout=config.temporal_dropout,
            kernel_size=config.tcn_kernel_size,
            dilation_base=config.tcn_dilation_base,
            activation=config.activation,
            num_heads=config.attention_heads,
            max_lag=config.lag,
        )
        self.manip_encoder = build_temporal_encoder(
            name=config.temporal_encoder,
            input_dim=config.in_channels,
            hidden_dim=config.hidden_dim,
            num_layers=config.temporal_layers,
            dropout=config.temporal_dropout,
            kernel_size=config.tcn_kernel_size,
            dilation_base=config.tcn_dilation_base,
            activation=config.activation,
            num_heads=config.attention_heads,
            max_lag=config.lag,
        )
        self.spatial_gnn = SpatialGNNStack(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.spatial_layers,
            dropout=config.spatial_dropout,
            activation=config.activation,
        )
        self.causal_fusion = CausalFusion(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            rounds=config.fusion_rounds,
        )
        self.decoder = PredictionDecoder(
            hidden_dim=config.hidden_dim,
            output_dim=config.horizon,
            num_layers=config.decoder_layers,
            dropout=config.decoder_dropout,
            activation=config.activation,
        )

        self.loss_node_indices = self.manipulated_indices

    @staticmethod
    def _build_control_lag_weights(
        lag: int, control_last_weight: Optional[float], device: torch.device
    ) -> Optional[torch.Tensor]:
        """Build lag weights for control nodes.

        Args:
            lag: Number of lag timesteps.
            control_last_weight: Percentage (0-100) of weight assigned to the last lag timestep.
                Remaining weight is distributed equally among earlier timesteps.
            device: Device to create tensor on.

        Returns:
            Tensor of shape (lag,) with weights, or None if control_last_weight is None.
        """
        if control_last_weight is None:
            return None
        if control_last_weight < 0.0 or control_last_weight > 100.0:
            raise ValueError("control_last_weight must be between 0 and 100.")
        if lag <= 1:
            return torch.ones(1, dtype=torch.float32, device=device)

        last_share = control_last_weight / 100.0
        other_share = (1.0 - last_share) / (lag - 1)
        weights = torch.full((lag,), other_share, dtype=torch.float32, device=device)
        weights[-1] = last_share
        return weights

    def forward_components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (batch, features, nodes, lag)
        x = x.permute(0, 3, 2, 1).contiguous()  # (batch, lag, nodes, features)
        x_control = x.index_select(2, self.control_indices)  # (batch, lag, num_control, features)
        x_manip = x.index_select(2, self.manipulated_indices)

        # Apply control lag weights if specified
        # Weights shape: (lag,) -> broadcast to (1, lag, 1, 1)
        if self.control_lag_weights is not None:
            weights = self.control_lag_weights.view(1, -1, 1, 1)
            x_control = x_control * weights

        h_control = self.control_encoder(x_control)
        h_manip = self.manip_encoder(x_manip)
        h_manip = self.spatial_gnn(h_manip, self.adj_mm_norm)
        h_fused = self.causal_fusion(h_control, h_manip, self.adj_cm)
        m_pred = self.decoder(h_fused)
        if self.target_indices.numel() > 0:
            y_pred = m_pred.index_select(1, self.target_indices)
        else:
            y_pred = m_pred
        return {"M_pred": m_pred, "Y_pred": y_pred, "X_control": x_control, "X_manip": x_manip}

    def forward_manipulated_only(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only manipulated node predictions.

        Args:
            x: Input tensor of shape (batch, features, nodes, lag).

        Returns:
            Predictions for manipulated nodes only, shape (batch, num_manip, horizon).
        """
        parts = self.forward_components(x)
        m_delta = parts["M_pred"]  # (batch, num_manip, horizon)

        if self.config.use_residual:
            # Get last input value for manipulated nodes
            manip_channel = min(self.target_channel, parts["X_manip"].size(-1) - 1)
            last_manip = parts["X_manip"][:, -1, :, manip_channel]  # (batch, num_manip)

            # Residual prediction: last_value + learned_delta
            m_pred = last_manip.unsqueeze(-1) + m_delta  # (batch, num_manip, horizon)
        else:
            # Direct prediction: decoder output is the final prediction
            m_pred = m_delta

        return m_pred

    def forward(self, x: torch.Tensor, manipulated_only: bool = False) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, features, nodes, lag).
            manipulated_only: If True, return only manipulated node predictions
                with shape (batch, num_manip, horizon). If False, return full
                predictions with shape (batch, num_nodes, horizon).

        Returns:
            Predictions tensor.
        """
        if manipulated_only:
            return self.forward_manipulated_only(x)

        parts = self.forward_components(x)
        m_delta = parts["M_pred"]  # Now represents delta/change from last input (or direct pred if use_residual=False)

        if self.config.use_residual:
            # Get last input value for manipulated nodes
            # X_manip shape: (batch, lag, num_manip, features)
            manip_channel = min(self.target_channel, parts["X_manip"].size(-1) - 1)
            last_manip = parts["X_manip"][:, -1, :, manip_channel]  # (batch, num_manip)

            # Residual prediction: last_value + learned_delta
            m_pred = last_manip.unsqueeze(-1) + m_delta  # (batch, num_manip, horizon)
        else:
            # Direct prediction: decoder output is the final prediction
            m_pred = m_delta

        # Control nodes: use last known value (unchanged)
        control_channel = min(self.target_channel, parts["X_control"].size(-1) - 1)
        control_last = parts["X_control"][:, -1, :, control_channel]
        control_pred = control_last.unsqueeze(-1).repeat(1, 1, self.horizon)

        # Construct full prediction tensor
        full_pred = x.new_zeros((x.size(0), self.num_nodes, self.horizon))
        full_pred.index_copy_(1, self.manipulated_indices, m_pred)
        full_pred.index_copy_(1, self.control_indices, control_pred)
        return full_pred


def parse_index_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    tokens = [token for token in value.replace(",", " ").split() if token.strip()]
    return [int(token) for token in tokens]


def load_feature_names(dataset_dir: str | Path) -> Optional[List[str]]:
    base = Path(dataset_dir)
    candidates = [
        base / "feature_names.txt",
        base / "causal" / "feature_names.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
    return None


def infer_node_groups(
    feature_names: Optional[Sequence[str]],
    num_nodes: int,
    control_nodes: Optional[Sequence[int]] = None,
    manip_nodes: Optional[Sequence[int]] = None,
) -> tuple[List[int], List[int]]:
    if control_nodes is not None:
        control = list(control_nodes)
    else:
        control = []
        if feature_names:
            control = [idx for idx, name in enumerate(feature_names) if name.lower().startswith("xmv_")]
    if manip_nodes is not None:
        manip = list(manip_nodes)
    else:
        manip = []
        if feature_names:
            manip = [idx for idx, name in enumerate(feature_names) if name.lower().startswith("xmeas_")]
    if not control and not manip:
        raise ValueError("Unable to infer control/manipulated nodes; provide --control_nodes or --manip_nodes.")
    if not control:
        control = [idx for idx in range(num_nodes) if idx not in set(manip)]
    if not manip:
        manip = [idx for idx in range(num_nodes) if idx not in set(control)]
    overlap = set(control) & set(manip)
    if overlap:
        raise ValueError(f"Control/manipulated node sets overlap: {sorted(overlap)}")
    return control, manip
