from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.graphwavenet import GraphWaveNet
from models.stgcn import STGCN, STGCNConfig
from preprocessing.data_reader import TemporalDatasetBundle, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spatio-temporal models on traffic datasets.")
    parser.add_argument("--model", type=str, choices=["stgcn", "graphwavenet"], default="stgcn")
    parser.add_argument("--dataset", type=str, choices=["METRLA", "PEMSBAY"], default="METRLA")
    parser.add_argument("--data_root", type=str, default=None, help="Path to dataset root directory.")
    parser.add_argument("--lag", type=int, default=12, help="Number of historical steps.")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2, help="Number of STGCN blocks.")
    parser.add_argument("--temporal_kernel", type=int, default=3)
    parser.add_argument("--cheb_k", type=int, default=3, help="Chebyshev polynomial order for STGCN.")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_channel", type=int, default=0, help="Feature channel to forecast.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--output", type=str, default=None, help="Optional path to save trained weights.")
    # Graph WaveNet specific
    parser.add_argument("--skip_channels", type=int, default=256)
    parser.add_argument("--end_channels", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--disable_gcn", action="store_true")
    parser.add_argument("--disable_adaptive_adj", action="store_true")
    return parser.parse_args()


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sym_normalized_adjacency(adj: torch.Tensor) -> torch.Tensor:
    device = adj.device
    adj = adj + torch.eye(adj.size(0), device=device)
    degree = adj.sum(1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    diag = torch.diag(degree_inv_sqrt)
    return diag @ adj @ diag

def build_dataloaders(bundle: TemporalDatasetBundle, batch_size: int, num_workers: int) -> Dict[str, DataLoader]:
    return {
        "train": DataLoader(bundle.train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(bundle.val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(bundle.test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def build_model(args: argparse.Namespace, bundle: TemporalDatasetBundle, device: torch.device) -> torch.nn.Module:
    adjacency = bundle.adjacency.to(device)
    if args.model == "stgcn":
        config = STGCNConfig(
            num_nodes=bundle.num_nodes,
            in_channels=bundle.num_features,
            hidden_channels=args.hidden_channels,
            horizon=args.horizon,
            num_layers=args.num_layers,
            temporal_kernel=args.temporal_kernel,
            k_order=args.cheb_k,
        )
        model = STGCN(config, adjacency=adjacency)
        return model.to(device)

    supports = [sym_normalized_adjacency(adjacency).to(device)]
    model = GraphWaveNet(
        device=device,
        num_nodes=bundle.num_nodes,
        dropout=args.dropout,
        supports=supports,
        gcn_bool=not args.disable_gcn,
        addaptadj=not args.disable_adaptive_adj,
        aptinit=supports[0],
        in_dim=bundle.num_features,
        out_dim=args.horizon,
        residual_channels=args.hidden_channels,
        dilation_channels=args.hidden_channels,
        skip_channels=args.skip_channels,
        end_channels=args.end_channels,
        kernel_size=args.kernel_size,
        blocks=args.blocks,
        layers=args.layers,
    )
    return model.to(device)


def prepare_batch(batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    x = x.to(device).float()  # (batch, lag, nodes, features)
    y = y.to(device).float()  # (batch, horizon, nodes)
    x = x.permute(0, 3, 2, 1).contiguous()  # (batch, features, nodes, lag)
    target = y.permute(0, 2, 1).contiguous()  # (batch, nodes, horizon)
    return x, target


def forward_pass(
    model: torch.nn.Module,
    x: torch.Tensor,
    model_type: str,
) -> torch.Tensor:
    if model_type == "graphwavenet":
        pad_len = max(0, getattr(model, "receptive_field", 1) - x.size(-1))
        padded_x = nn.functional.pad(x, (pad_len, 0, 0, 0))
        output = model(padded_x)
        output = output.transpose(1, 3)  # (batch, time_out, nodes, horizon)
        if output.size(1) == 1:
            output = output.squeeze(1)  # (batch, nodes, horizon)
        else:
            output = output.mean(dim=1)
        return output

    output = model(x)
    if output.dim() == 4 and output.size(1) == 1:
        output = output.squeeze(1)
    elif output.dim() == 4:
        output = output.mean(dim=1)
    return output


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    return mae, rmse


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    num_batches = 0

    for batch in loader:
        x, target = prepare_batch(batch, device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            prediction = forward_pass(model, x, model_type)
            loss = criterion(prediction, target)
            if is_train:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        mae, rmse = compute_metrics(prediction, target)
        total_loss += loss.item()
        total_mae += mae
        total_rmse += rmse
        num_batches += 1

    return {
        "loss": total_loss / max(1, num_batches),
        "mae": total_mae / max(1, num_batches),
        "rmse": total_rmse / max(1, num_batches),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    data_root = Path(args.data_root) if args.data_root else None

    bundle = load_dataset(
        dataset=args.dataset,
        lag=args.lag,
        horizon=args.horizon,
        data_root=data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        target_channel=args.target_channel,
    )
    loaders = build_dataloaders(bundle, args.batch_size, args.num_workers)
    model = build_model(args, bundle, device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state: Dict[str, Any] | None = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model,
            loaders["train"],
            device,
            args.model,
            criterion,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_stats = run_epoch(model, loaders["val"], device, args.model, criterion)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_stats['loss']:.4f} MAE: {train_stats['mae']:.4f} RMSE: {train_stats['rmse']:.4f} | "
            f"Val Loss: {val_stats['loss']:.4f} MAE: {val_stats['mae']:.4f} RMSE: {val_stats['rmse']:.4f}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience and patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_stats = run_epoch(model, loaders["test"], device, args.model, criterion)
    print(
        f"Test | Loss: {test_stats['loss']:.4f} MAE: {test_stats['mae']:.4f} RMSE: {test_stats['rmse']:.4f}"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_type": args.model,
                "dataset": args.dataset,
                "config": {
                    "args": vars(args),
                    "stgcn_config": asdict(model.config) if args.model == "stgcn" else None,
                },
            },
            output_path,
        )
        print(f"Model checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
