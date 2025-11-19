from __future__ import annotations

import argparse
import csv
import shlex
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import train as forecaster_module
from counterfactual import (
    CounterfactualGenerator,
    DiffusionConfig,
    DiffusionModel,
    DiffusionTrainer,
    ForecastGuidance,
    GuidanceConfig,
    SpatioTemporalUNet,
)
from counterfactual.guidance import prepare_forecaster_input
from preprocessing.data_reader import TemporalDatasetBundle, load_dataset
from train import build_dataloaders, train_pipeline, test_pipeline


def add_forecaster_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("forecaster", help="Train or evaluate an ST-GNN forecaster.")
    parser.add_argument("--model", type=str, choices=["stgcn", "graphwavenet", "mstgcn", "astgcn"], default="stgcn")
    parser.add_argument("--dataset", type=str, choices=["METRLA", "PEMSBAY"], default="METRLA")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--temporal_kernel", type=int, default=3)
    parser.add_argument("--cheb_k", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated CUDA device IDs, e.g., '0,1'.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_channel", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--time_strides", type=int, default=3)
    parser.add_argument("--skip_channels", type=int, default=256)
    parser.add_argument("--end_channels", type=int, default=512)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--disable_gcn", action="store_true")
    parser.add_argument("--disable_adaptive_adj", action="store_true")
    parser.set_defaults(handler=run_forecaster_command)
    return parser


def add_diffusion_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("diffusion", help="Train the diffusion prior over past trajectories.")
    parser.add_argument("--dataset", type=str, choices=["METRLA", "PEMSBAY"], default="METRLA")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--lag", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated CUDA device IDs, e.g., '0,1'.")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--target_channel", type=int, default=0)
    parser.add_argument("--diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--diffusion_base_channels", type=int, default=64)
    parser.add_argument("--diffusion_channel_mults", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--diffusion_time_dim", type=int, default=256)
    parser.add_argument("--diffusion_dropout", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, choices=["l1", "l2"], default="l2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold.")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA for model parameters.")
    parser.add_argument("--no_ema", action="store_false", dest="use_ema", help="Disable EMA.")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/diffusion")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume training from a checkpoint.")
    parser.set_defaults(handler=run_diffusion_command)
    return parser


def add_counterfactual_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("counterfactual", help="Generate diffusion-guided counterfactual past windows.")
    parser.add_argument("--forecaster_checkpoint", type=str, required=True)
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None, help="Optional override for stored dataset root.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--mask_path", type=str, default=None, help="Optional path to a numpy mask of shape (T, N, F).")
    parser.add_argument("--target_path", type=str, default=None, help="Optional path to a numpy target (H, N).")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--lambda_scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--temporal_weight", type=float, default=1e-3)
    parser.add_argument("--spatial_weight", type=float, default=1e-3)
    parser.add_argument("--control_weight", type=float, default=0.0)
    parser.add_argument("--rate_limit", type=float, default=None)
    parser.add_argument("--clamp_min", type=float, default=None)
    parser.add_argument("--clamp_max", type=float, default=None)
    parser.add_argument("--lower_bound", type=float, default=None)
    parser.add_argument("--upper_bound", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated CUDA device IDs, e.g., '0,1'.")
    parser.add_argument("--output_path", type=str, default="counterfactual_samples.pt")
    parser.add_argument("--warm_start", action="store_true", help="Warm-start guidance from the observed trajectory.")
    parser.add_argument("--use_predicted_target", action="store_true", help="Use the forecaster's prediction as the guidance target.")
    parser.add_argument(
        "--target_adjust_percent",
        type=float,
        default=0.0,
        help="Percentage change to apply to the final horizon step of the selected node's target.",
    )
    parser.add_argument(
        "--target_adjust_offset",
        type=float,
        default=0.0,
        help="Additive offset (DC shift) to apply to the final horizon step of the selected node's target.",
    )
    parser.add_argument(
        "--target_adjust_node",
        type=int,
        default=0,
        help="Node index whose final horizon value is adjusted (use -1 to apply to all nodes).",
    )
    parser.add_argument(
        "--plot_node",
        type=int,
        default=None,
        help="Node index to visualize; defaults to the adjusted node or 0 if unspecified.",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Optional path to save a plot comparing original and counterfactual predictions.",
    )
    parser.add_argument(
        "--cf_horizon",
        type=int,
        default=None,
        help="Override the number of horizon steps to enforce for counterfactual guidance and plotting.",
    )
    parser.add_argument(
        "--anchor_start_weight",
        type=float,
        default=1.0,
        help="Weight applied to early horizon steps to keep them close to the baseline forecast.",
    )
    parser.add_argument(
        "--anchor_end_weight",
        type=float,
        default=0.05,
        help="Weight applied to the final horizon step for the baseline-anchoring loss.",
    )
    parser.add_argument(
        "--anchor_loss_scale",
        type=float,
        default=1.0,
        help="Overall scale for the baseline anchoring penalty.",
    )
    parser.set_defaults(handler=run_counterfactual_command)
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified driver for ST-GNN training and diffusion-based counterfactuals.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_forecaster_subcommand(subparsers)
    add_diffusion_subcommand(subparsers)
    add_counterfactual_subcommand(subparsers)
    args = parser.parse_args()
    gpus = getattr(args, "gpus", None)
    args.gpu_ids = forecaster_module.parse_gpu_ids(gpus)
    return args


# ------------------------- Diffusion helpers ------------------------- #
def build_temporal_context(length: int, device: torch.device) -> torch.Tensor:
    timeline = torch.linspace(0, 1, steps=length, device=device).view(1, length, 1, 1)
    return timeline


def create_diffusion_config(args: argparse.Namespace) -> DiffusionConfig:
    return DiffusionConfig(
        timesteps=args.diffusion_timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        base_channels=args.diffusion_base_channels,
        channel_multipliers=tuple(args.diffusion_channel_mults),
        time_embedding_dim=args.diffusion_time_dim,
        dropout=args.diffusion_dropout,
        loss_type="l1" if args.loss_type == "l1" else "l2",
    )


def resolve_diffusion_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    run_dir = Path(args.checkpoint_dir) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.dataset.lower()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "diffusion.pt"


def append_csv_row(csv_path: Path, headers: list[str], values: list[float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(headers)
        writer.writerow(values)


def _format_training_command() -> str:
    python_exec = sys.executable or "python"
    try:
        arg_string = shlex.join(sys.argv)
    except AttributeError:
        arg_string = " ".join(shlex.quote(arg) for arg in sys.argv)
    return f"{python_exec} {arg_string}".strip()


def write_training_command_file(directory: Path) -> None:
    command_path = directory / "trainingCommand.txt"
    command_path.write_text(_format_training_command() + "\n")


def save_diffusion_checkpoint(
    path: Path,
    epoch: int,
    diffusion: DiffusionModel,
    optimizer: torch.optim.Optimizer,
    config: DiffusionConfig,
    dataset_meta: Dict[str, Any],
    model_meta: Dict[str, Any],
    metrics: Dict[str, float],
    ema_state: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": diffusion.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "dataset_meta": dataset_meta,
        "model_meta": model_meta,
        "metrics": metrics,
    }
    if ema_state is not None:
        checkpoint["ema_state"] = ema_state
    torch.save(checkpoint, path)


def load_diffusion_checkpoint(path: Path, device: torch.device, gpu_ids: Optional[List[int]] = None) -> Tuple[DiffusionModel, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    config = DiffusionConfig(**checkpoint["config"])
    model_meta = checkpoint["model_meta"]
    network = SpatioTemporalUNet(
        in_channels=model_meta["in_channels"],
        base_channels=model_meta["base_channels"],
        channel_multipliers=tuple(model_meta["channel_multipliers"]),
        time_embedding_dim=model_meta["time_embedding_dim"],
        dropout=model_meta["dropout"],
    ).to(device)
    diffusion = DiffusionModel(network, config, gpu_ids=gpu_ids).to(device)
    diffusion.load_state_dict(checkpoint["model_state"])
    return diffusion, checkpoint


# ------------------------- Command handlers ------------------------- #
def run_forecaster_command(args: argparse.Namespace) -> None:
    if args.mode == "train":
        train_pipeline(args)
    else:
        test_pipeline(args)


def run_diffusion_command(args: argparse.Namespace) -> None:
    device = forecaster_module.resolve_device(args.device, args.gpu_ids)
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
    loaders, _ = build_dataloaders(bundle, args.batch_size, args.num_workers)

    network = SpatioTemporalUNet(
        in_channels=bundle.num_features,
        base_channels=args.diffusion_base_channels,
        channel_multipliers=tuple(args.diffusion_channel_mults),
        time_embedding_dim=args.diffusion_time_dim,
        dropout=args.diffusion_dropout,
    ).to(device)
    diffusion_config = create_diffusion_config(args)
    diffusion = DiffusionModel(network, diffusion_config, gpu_ids=args.gpu_ids).to(device)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = DiffusionTrainer(
        diffusion,
        optimizer,
        device,
        bundle.adjacency.to(device),
        temporal_context=build_temporal_context(args.lag, device),
        grad_clip=args.grad_clip,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
    )

    checkpoint_path = resolve_diffusion_checkpoint_path(args)
    write_training_command_file(checkpoint_path.parent)
    metrics_csv = checkpoint_path.parent / "metrics.csv"
    best_val = float("inf")
    start_epoch = 1
    if args.checkpoint:
        ckpt = torch.load(Path(args.checkpoint), map_location=device)
        diffusion.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("metrics", {}).get("val_loss", best_val)
        
        # Load EMA state if available
        if trainer.ema is not None and "ema_state" in ckpt:
            trainer.ema.load_state_dict(ckpt["ema_state"])
            print(f"Resumed diffusion training with EMA from {args.checkpoint} at epoch {start_epoch}")
        else:
            print(f"Resumed diffusion training from {args.checkpoint} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = trainer.train_epoch(loaders["train"])
        val_loss = trainer.evaluate_epoch(loaders["val"])
        print(f"[Diffusion] Epoch {epoch:03d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
        append_csv_row(
            metrics_csv,
            ["epoch", "train_loss", "val_loss"],
            [epoch, train_loss, val_loss],
        )
        if val_loss < best_val:
            best_val = val_loss
            dataset_meta = {
                "dataset": args.dataset,
                "lag": args.lag,
                "horizon": args.horizon,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "target_channel": args.target_channel,
                "data_root": str(data_root) if data_root else None,
                "num_nodes": bundle.num_nodes,
                "num_features": bundle.num_features,
            }
            model_meta = {
                "in_channels": bundle.num_features,
                "base_channels": args.diffusion_base_channels,
                "channel_multipliers": tuple(args.diffusion_channel_mults),
                "time_embedding_dim": args.diffusion_time_dim,
                "dropout": args.diffusion_dropout,
            }
            metrics = {"train_loss": train_loss, "val_loss": val_loss}
            ema_state = trainer.ema.state_dict() if trainer.ema is not None else None
            save_diffusion_checkpoint(checkpoint_path, epoch, diffusion, optimizer, diffusion_config, dataset_meta, model_meta, metrics, ema_state)
            print(f"Saved best diffusion checkpoint to {checkpoint_path}")


def prepare_mask(lag: int, num_nodes: int, num_features: int, source: Optional[str]) -> torch.Tensor:
    if source is None:
        return torch.ones((lag, num_nodes, num_features), dtype=torch.float32)
    arr = np.load(source)
    mask = torch.from_numpy(arr).float()
    if mask.shape != (lag, num_nodes, num_features):
        raise ValueError(f"Mask shape {mask.shape} does not match (lag, nodes, features) = {(lag, num_nodes, num_features)}")
    return mask


def prepare_target(default_target: torch.Tensor, path: Optional[str]) -> torch.Tensor:
    if path is None:
        return default_target
    arr = np.load(path)
    tensor = torch.from_numpy(arr).float()
    if tensor.shape != default_target.shape:
        raise ValueError(f"Target shape {tensor.shape} does not match expected {default_target.shape}")
    return tensor


def adjust_target(
    target: torch.Tensor,
    percent: float,
    offset: float,
    node_index: int,
) -> torch.Tensor:
    adjusted = target.clone()
    if adjusted.shape[1] == 0:
        return adjusted
    indices: torch.Tensor | slice
    if node_index < 0:
        indices = slice(None)
    elif node_index >= adjusted.shape[0]:
        raise ValueError(f"target_adjust_node {node_index} is out of range for {adjusted.shape[0]} nodes")
    else:
        indices = node_index
    if percent != 0.0:
        factor = 1.0 + percent / 100.0
        adjusted[indices, -1] = adjusted[indices, -1] * factor
    if offset != 0.0:
        adjusted[indices, -1] = adjusted[indices, -1] + offset
    return adjusted


def build_guidance_target(
    baseline: torch.Tensor,
    adjusted_target: torch.Tensor,
    node_index: int,
) -> torch.Tensor:
    if baseline.shape != adjusted_target.shape:
        raise ValueError("baseline and adjusted target must share the same shape")
    horizon = baseline.shape[1]
    if horizon == 0:
        return baseline.clone()
    guidance = baseline.clone()
    ramp = torch.linspace(0.0, 1.0, steps=horizon, dtype=baseline.dtype, device=baseline.device)
    if node_index < 0:
        node_idx = torch.arange(baseline.shape[0], dtype=torch.long)
    elif node_index >= baseline.shape[0]:
        raise ValueError(f"target_adjust_node {node_index} is out of range for {baseline.shape[0]} nodes")
    else:
        node_idx = torch.tensor([node_index], dtype=torch.long)
    if node_idx.numel() == 0:
        return guidance
    delta = adjusted_target[node_idx, -1] - baseline[node_idx, -1]
    guidance[node_idx] = baseline[node_idx] + delta.unsqueeze(1) * ramp
    guidance[node_idx, -1] = adjusted_target[node_idx, -1]
    return guidance


def truncate_horizon(tensor: torch.Tensor, horizon: Optional[int]) -> torch.Tensor:
    if horizon is None or tensor.shape[1] <= horizon:
        return tensor
    if horizon <= 0:
        raise ValueError("cf_horizon must be positive")
    return tensor[:, :horizon].contiguous()


def save_prediction_plot(
    plot_path: Path,
    node_index: int,
    original_prediction: torch.Tensor,
    cf_prediction: torch.Tensor,
    adjusted_target: torch.Tensor,
    ground_truth: torch.Tensor,
    guidance_target: Optional[torch.Tensor] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plot generation.")
        return

    node_index = max(0, min(node_index, original_prediction.shape[0] - 1))
    horizon = original_prediction.shape[1]
    steps = list(range(horizon))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, original_prediction[node_index].detach().cpu().numpy(), label="Original forecast", linewidth=2)
    plt.plot(
        steps,
        cf_prediction[node_index].detach().cpu().numpy(),
        label="Counterfactual forecast (best)",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        steps,
        ground_truth[node_index].detach().cpu().numpy(),
        label="Ground truth",
        linestyle=":",
        linewidth=2,
    )
    if guidance_target is not None:
        plt.plot(
            steps,
            guidance_target[node_index].detach().cpu().numpy(),
            label="Guidance target",
            linestyle="-.",
            linewidth=2,
        )
    plt.scatter(
        [horizon - 1],
        [adjusted_target[node_index, -1].detach().cpu().item()],
        label="Adjusted target (final step)",
        color="red",
    )
    plt.title(f"Node {node_index} horizon forecast comparison")
    plt.xlabel("Horizon step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def select_split_dataset(bundle: TemporalDatasetBundle, split: str):
    if split == "train":
        return bundle.train
    if split == "val":
        return bundle.val
    return bundle.test


def load_forecaster_from_checkpoint(path: Path, device: torch.device, data_root_override: Optional[Path] = None) -> Tuple[torch.nn.Module, TemporalDatasetBundle, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    checkpoint_args = checkpoint.get("config", {}).get("args", {})

    dataset_name = checkpoint.get("dataset") or checkpoint_args.get("dataset")
    dataset_name = dataset_name.upper() if dataset_name else "METRLA"
    lag = checkpoint_args.get("lag", 12)
    horizon = checkpoint_args.get("horizon", 12)
    train_ratio = checkpoint_args.get("train_ratio", 0.7)
    val_ratio = checkpoint_args.get("val_ratio", 0.1)
    target_channel = checkpoint_args.get("target_channel", 0)
    data_root_value = data_root_override or checkpoint_args.get("data_root")
    data_root = Path(data_root_value) if data_root_value else None

    bundle = load_dataset(
        dataset=dataset_name,
        lag=lag,
        horizon=horizon,
        data_root=data_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        target_channel=target_channel,
    )

    args_namespace = argparse.Namespace(**checkpoint_args)
    model = forecaster_module.build_model(args_namespace, bundle, device)
    state_key = "model_state" if "model_state" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    metadata = {
        "lag": lag,
        "horizon": horizon,
        "dataset": dataset_name,
        "target_channel": target_channel,
    }
    return model, bundle, metadata


def run_counterfactual_command(args: argparse.Namespace) -> None:
    device = forecaster_module.resolve_device(args.device, args.gpu_ids)
    forecaster_path = Path(args.forecaster_checkpoint)
    diffusion_path = Path(args.diffusion_checkpoint)

    forecaster, bundle, dataset_meta = load_forecaster_from_checkpoint(
        forecaster_path, device, Path(args.data_root) if args.data_root else None
    )

    diffusion, diffusion_ckpt = load_diffusion_checkpoint(diffusion_path, device, gpu_ids=args.gpu_ids)
    dataset_info = diffusion_ckpt.get("dataset_meta", {})
    if dataset_info.get("dataset") and dataset_info.get("dataset") != dataset_meta["dataset"]:
        print("Warning: Forecaster and diffusion checkpoints were trained on different datasets.")

    dataset_horizon = dataset_meta["horizon"]
    cf_horizon = args.cf_horizon if args.cf_horizon is not None else dataset_horizon
    if cf_horizon <= 0:
        raise ValueError("cf_horizon must be positive")
    if cf_horizon > dataset_horizon:
        raise ValueError(f"cf_horizon {cf_horizon} exceeds dataset horizon {dataset_horizon}")

    split_dataset = select_split_dataset(bundle, args.split)
    if args.sample_index < 0 or args.sample_index >= len(split_dataset):
        raise ValueError(f"sample_index {args.sample_index} is out of range for split {args.split}")
    past_window, target_future = split_dataset[args.sample_index]
    past_window = past_window.float()
    target_future = target_future.permute(1, 0).contiguous().float()

    default_target = truncate_horizon(prepare_target(target_future, args.target_path).float(), cf_horizon)
    past_window_batch = past_window.unsqueeze(0).to(device).float()
    with torch.no_grad():
        baseline_forecast = (
            forecaster(prepare_forecaster_input(past_window_batch))
            .squeeze(0)
            .detach()
            .cpu()
            .float()
        )
    baseline_forecast = truncate_horizon(baseline_forecast, cf_horizon)

    anchor_start = max(args.anchor_start_weight, 0.0)
    anchor_end = max(args.anchor_end_weight, 0.0)
    anchor_weights = torch.linspace(anchor_start, anchor_end, steps=cf_horizon, dtype=torch.float32)

    if args.use_predicted_target:
        target_source = baseline_forecast.clone()
    else:
        target_source = default_target.clone()

    adjusted_target = truncate_horizon(
        adjust_target(target_source, args.target_adjust_percent, args.target_adjust_offset, args.target_adjust_node).float(),
        cf_horizon,
    )
    if args.target_path is not None:
        guidance_target = adjusted_target.clone()
    else:
        guidance_target = build_guidance_target(baseline_forecast, adjusted_target, args.target_adjust_node)
    mask = prepare_mask(past_window.shape[0], bundle.num_nodes, bundle.num_features, args.mask_path)

    sample_shape = torch.Size(past_window.shape)  # (T, N, F)
    mask = mask.to(device)
    target_batched = guidance_target.unsqueeze(0).repeat(args.samples, 1, 1).to(device)
    mask_batched = mask.unsqueeze(0).repeat(args.samples, 1, 1, 1).to(device)

    guidance_config = GuidanceConfig(
        lambda_scale=args.lambda_scale,
        eta=args.eta,
        temporal_weight=args.temporal_weight,
        spatial_weight=args.spatial_weight,
        control_energy_weight=args.control_weight,
        rate_limit=args.rate_limit,
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
        anchor_start_weight=args.anchor_start_weight,
        anchor_end_weight=args.anchor_end_weight,
        anchor_loss_scale=args.anchor_loss_scale,
    )
    guidance = ForecastGuidance(
        forecaster=forecaster,
        target=target_batched,
        mask=mask_batched,
        adjacency=bundle.adjacency.to(device),
        config=guidance_config,
        lower_bounds=args.lower_bound,
        upper_bounds=args.upper_bound,
        baseline=baseline_forecast,
        anchor_weights=anchor_weights,
    )

    generator = CounterfactualGenerator(
        diffusion,
        adjacency=bundle.adjacency.to(device),
        device=device,
        temporal_context=build_temporal_context(dataset_meta["lag"], device),
    )

    warm_start = None
    if args.warm_start:
        warm_start = past_window.unsqueeze(0).repeat(args.samples, 1, 1, 1).to(device)

    samples = generator.generate(
        sample_shape=sample_shape,
        guidance=guidance,
        num_samples=args.samples,
        max_steps=args.max_steps,
        warm_start=warm_start,
    )

    with torch.no_grad():
        cf_preds = forecaster(prepare_forecaster_input(samples))
        cf_preds = cf_preds[:, :, :cf_horizon]
        mse = torch.mean((cf_preds - target_batched) ** 2, dim=(1, 2))
        print(f"Counterfactual guidance MSE per sample: {mse.cpu().numpy()}")

    cf_preds_cpu = cf_preds.detach().cpu().float()
    mse_cpu = mse.detach().cpu()
    best_idx = int(torch.argmin(mse_cpu).item())
    best_cf_prediction = cf_preds_cpu[best_idx]

    plot_node = args.plot_node if args.plot_node is not None else (args.target_adjust_node if args.target_adjust_node >= 0 else 0)
    plot_path = Path(args.plot_path) if args.plot_path else Path(args.output_path).with_name(Path(args.output_path).stem + "_plot.png")
    if adjusted_target.shape[0] > 0 and adjusted_target.shape[1] > 0:
        save_prediction_plot(
            plot_path,
            plot_node,
            baseline_forecast,
            best_cf_prediction,
            adjusted_target,
            default_target,
            guidance_target,
        )

    output = {
        "samples": samples.cpu(),
        "target": target_batched.cpu(),
        "mask": mask_batched.cpu(),
        "guidance": asdict(guidance_config),
        "metadata": {
            "forecaster_checkpoint": str(forecaster_path),
            "diffusion_checkpoint": str(diffusion_path),
            "dataset": dataset_meta,
            "diffusion_dataset": dataset_info,
            "split": args.split,
            "sample_index": args.sample_index,
        },
        "baseline_forecast": baseline_forecast,
        "original_target": default_target,
        "adjusted_target": adjusted_target,
        "target_adjust_percent": args.target_adjust_percent,
        "target_adjust_offset": args.target_adjust_offset,
        "target_adjust_node": args.target_adjust_node,
        "use_predicted_target": args.use_predicted_target,
        "plot_path": str(plot_path),
        "counterfactual_mse": mse_cpu,
        "best_sample_index": best_idx,
        "best_counterfactual_prediction": best_cf_prediction,
        "cf_horizon": cf_horizon,
        "anchor_weights": anchor_weights,
        "guidance_target": guidance_target,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Saved counterfactual samples to {output_path}")


def main() -> None:
    args = parse_args()
    handler = getattr(args, "handler")
    handler(args)


if __name__ == "__main__":
    main()
