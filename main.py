from __future__ import annotations

import argparse
import csv
import math
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
from counterfactual.noise_schedule import build_beta_schedule, prepare_diffusion_terms
from preprocessing.data_reader import TemporalDatasetBundle, load_dataset
from train import build_dataloaders, train_pipeline, test_pipeline


def add_forecaster_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("forecaster", help="Train or evaluate an ST-GNN forecaster.")
    parser.add_argument("--model", type=str, choices=["stgcn", "graphwavenet", "mstgcn", "astgcn"], default="stgcn")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["METRLA", "PEMSBAY", "METRLA_15", "METRLA_30"],
        default="METRLA",
    )
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
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["METRLA", "PEMSBAY", "METRLA_15", "METRLA_30"],
        default="METRLA",
    )
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
        "--target_focus_percent",
        type=float,
        default=80.0,
        help="Percentage of forecast loss weight given to the adjusted node; remaining weight is distributed to other nodes by inverse hop distance.",
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
        "--plot_top_cf_window_path",
        type=str,
        default=None,
        help="Optional path to save a grid plot of the top-changed nodes in the past window (original vs counterfactual).",
    )
    parser.add_argument(
        "--plot_top_cf_window_k",
        type=int,
        default=10,
        help="Number of nodes to include in the past-window grid plot (ranked by mean absolute edit).",
    )
    parser.add_argument(
        "--plot_top_cf_window_feature",
        type=int,
        default=0,
        help="Feature index to visualize for the past-window grid plot.",
    )
    parser.add_argument(
        "--plot_lag_steps",
        type=int,
        default=None,
        help="Number of lagged time steps to plot (from most recent backwards). If None, plots all available lag steps.",
    )
    parser.add_argument(
        "--guidance_start_source",
        type=str,
        choices=["baseline", "ground_truth", "lag"],
        default="baseline",
        help="Select the starting point for the guidance trajectory interpolation.",
    )
    parser.add_argument(
        "--guidance_interpolation",
        type=str,
        choices=["linear"],
        default="linear",
        help="Interpolation scheme for guidance trajectory (last point is always fixed).",
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
    parser.add_argument(
        "--anchor_release_power",
        type=float,
        default=1.5,
        help="Exponent that shapes how quickly anchor weights decay toward the horizon (values > 1 hold longer).",
    )
    parser.add_argument(
        "--guidance_impute_steps",
        type=int,
        default=64,
        help="Number of diffusion steps to use when imputing the guidance trajectory between anchors.",
    )
    parser.add_argument(
        "--guidance_impute_schedule",
        type=str,
        choices=["linear", "cosine"],
        default="cosine",
        help="Noise schedule used for diffusion-based guidance imputation.",
    )
    parser.add_argument(
        "--guidance_path_strategy",
        type=str,
        choices=["diffusion", "linear"],
        default="diffusion",
        help="How to construct the guidance trajectory between start (last lag) and adjusted target endpoint.",
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


def _temporal_smooth_1d(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return x
    left = torch.cat([x[:1], x[:-1]])
    right = torch.cat([x[1:], x[-1:]])
    return (left + x + right) / 3.0


def diffusion_impute_path(
    start_value: torch.Tensor,
    end_value: torch.Tensor,
    prior_path: torch.Tensor,
    betas: torch.Tensor,
    smooth_weight: float = 0.65,
    prior_weight: float = 0.35,
) -> torch.Tensor:
    """Diffusion-style imputation of a 1D temporal path with fixed endpoints."""

    device = prior_path.device
    dtype = prior_path.dtype
    horizon = prior_path.shape[0]
    if horizon == 0:
        return prior_path.clone()

    terms = prepare_diffusion_terms(betas)
    terms = {k: v.to(device=device, dtype=dtype) for k, v in terms.items()}

    anchor_mask = torch.zeros(horizon, device=device, dtype=dtype)
    anchor_mask[0] = 1.0
    anchor_mask[-1] = 1.0
    anchor_values = torch.zeros_like(prior_path)
    anchor_values[0] = start_value
    anchor_values[-1] = end_value

    # Initialize with anchors + noisy prior.
    x = anchor_mask * anchor_values + (1 - anchor_mask) * prior_path
    x = terms["sqrt_alphas_cumprod"][-1] * x + terms["sqrt_one_minus_alphas_cumprod"][-1] * torch.randn_like(x)

    smooth_weight = float(min(max(smooth_weight, 0.0), 1.0))
    prior_weight = float(min(max(prior_weight, 0.0), 1.0))
    free_weight = max(0.0, 1.0 - (smooth_weight + prior_weight))

    for t_idx in range(betas.numel() - 1, -1, -1):
        smooth = _temporal_smooth_1d(x)
        x0_pred = prior_weight * prior_path + smooth_weight * smooth + free_weight * x
        x0_pred = anchor_mask * anchor_values + (1 - anchor_mask) * x0_pred

        sqrt_alpha_bar = terms["sqrt_alphas_cumprod"][t_idx]
        sqrt_one_minus_alpha_bar = torch.clamp(terms["sqrt_one_minus_alphas_cumprod"][t_idx], min=1e-8)
        betas_t = terms["betas"][t_idx]
        sqrt_recip_alpha = terms["sqrt_recip_alphas"][t_idx]
        posterior_variance_t = terms["posterior_variance"][t_idx]

        eps = (x - sqrt_alpha_bar * x0_pred) / sqrt_one_minus_alpha_bar
        model_mean = sqrt_recip_alpha * (x - betas_t / sqrt_one_minus_alpha_bar * eps)

        if t_idx == 0:
            x = model_mean
        else:
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_variance_t) * noise

        # Reinforce anchors after each denoise step.
        x = anchor_mask * anchor_values + (1 - anchor_mask) * x

    return x


def build_diffusion_guidance_target(
    baseline: torch.Tensor,
    adjusted_target: torch.Tensor,
    past_window: torch.Tensor,
    target_node: int,
    target_channel: int,
    impute_steps: int = 64,
    beta_schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Form a guidance trajectory by anchoring the first point to the last observed lag value
    and the last point to the adjusted target, then imputing the in-between steps via diffusion.
    """

    if baseline.shape != adjusted_target.shape:
        raise ValueError("baseline and adjusted_target must share the same shape for diffusion guidance")
    horizon = baseline.shape[1]
    if horizon == 0:
        return baseline.clone()

    num_nodes = baseline.shape[0]
    if target_node >= num_nodes or target_node < -1:
        raise ValueError(f"target_adjust_node {target_node} is out of range for {num_nodes} nodes")

    impute_steps = int(max(1, impute_steps))
    betas = build_beta_schedule(beta_schedule, impute_steps, beta_start=beta_start, beta_end=beta_end).to(baseline.device)
    guidance = baseline.clone()

    if target_channel < 0 or target_channel >= past_window.shape[-1]:
        raise ValueError(f"target_channel {target_channel} is out of range for past window features {past_window.shape[-1]}")

    start_values = past_window[-1, :, target_channel].to(baseline.device, baseline.dtype)
    target_nodes = range(num_nodes) if target_node < 0 else [target_node]

    for node in target_nodes:
        start_val = start_values[node]
        end_val = adjusted_target[node, -1].to(baseline.device, baseline.dtype)
        prior_path = baseline[node]
        imputed = diffusion_impute_path(start_val, end_val, prior_path, betas)
        guidance[node] = imputed

    return guidance


def build_linear_guidance_target(
    baseline: torch.Tensor,
    adjusted_target: torch.Tensor,
    past_window: torch.Tensor,
    target_node: int,
    target_channel: int,
) -> torch.Tensor:
    """
    Linearly interpolate between the last observed lag value and the adjusted target endpoint
    for the chosen node(s); other nodes follow the baseline forecast.
    """

    if baseline.shape != adjusted_target.shape:
        raise ValueError("baseline and adjusted_target must share the same shape for linear guidance")
    horizon = baseline.shape[1]
    if horizon == 0:
        return baseline.clone()

    num_nodes = baseline.shape[0]
    if target_node >= num_nodes or target_node < -1:
        raise ValueError(f"target_adjust_node {target_node} is out of range for {num_nodes} nodes")
    if target_channel < 0 or target_channel >= past_window.shape[-1]:
        raise ValueError(f"target_channel {target_channel} is out of range for past window features {past_window.shape[-1]}")

    guidance = baseline.clone()
    start_values = past_window[-1, :, target_channel].to(baseline.device, baseline.dtype)
    target_nodes = range(num_nodes) if target_node < 0 else [target_node]
    steps = torch.linspace(0.0, 1.0, steps=horizon, device=baseline.device, dtype=baseline.dtype)

    for node in target_nodes:
        start_val = start_values[node]
        end_val = adjusted_target[node, -1].to(baseline.device, baseline.dtype)
        interp = start_val + (end_val - start_val) * steps
        interp[-1] = end_val
        guidance[node] = interp

    return guidance


def build_guidance_target(
    baseline: torch.Tensor,
    adjusted_target: torch.Tensor,
    node_index: int,
    start_source: str = "baseline",
    start_target: Optional[torch.Tensor] = None,
    interpolation: str = "linear",
    lag_start: Optional[torch.Tensor] = None,
    lag_length: int = 0,
) -> torch.Tensor:
    if baseline.shape != adjusted_target.shape:
        raise ValueError("baseline and adjusted target must share the same shape")
    if baseline.dim() == 2:
        base = baseline
        target = adjusted_target
    elif baseline.dim() == 3 and baseline.size(-1) == 1:
        base = baseline.squeeze(-1)
        target = adjusted_target.squeeze(-1)
    else:
        raise ValueError("baseline/target must have shape (nodes, horizon) or (nodes, horizon, 1)")
    horizon = base.shape[1]
    if horizon == 0:
        return baseline.clone()
    guidance = base.clone()
    ramp_steps = horizon
    if start_source == "lag":
        ramp_steps = max(lag_length + horizon, horizon)
    ramp_full = torch.linspace(0.0, 1.0, steps=ramp_steps, dtype=base.dtype, device=base.device)
    ramp = ramp_full[-horizon:]
    if node_index < 0:
        node_idx = torch.arange(base.shape[0], dtype=torch.long, device=base.device)
    elif node_index >= base.shape[0]:
        raise ValueError(f"target_adjust_node {node_index} is out of range for {base.shape[0]} nodes")
    else:
        node_idx = torch.tensor([node_index], dtype=torch.long, device=base.device)
    if node_idx.numel() == 0:
        return baseline.clone()

    if start_source == "ground_truth":
        if start_target is None or start_target.shape != base.shape:
            raise ValueError("start_target must be provided with matching shape when start_source='ground_truth'")
        start_values = start_target[node_idx, 0]
    elif start_source == "lag":
        if lag_start is None or lag_start.shape[0] != base.shape[0]:
            raise ValueError("lag_start must be provided with shape (nodes,) when start_source='lag'")
        start_values = lag_start[node_idx]
    elif start_source == "baseline":
        start_values = base[node_idx, 0]
    else:
        raise ValueError(f"Unknown start_source {start_source}")

    end_values = target[node_idx, -1]
    if interpolation == "linear":
        guidance[node_idx] = start_values.unsqueeze(1) + (end_values - start_values).unsqueeze(1) * ramp
    else:
        raise ValueError(f"Unknown interpolation {interpolation}")

    guidance[node_idx, -1] = end_values

    if baseline.dim() == 3:
        guidance = guidance.unsqueeze(-1)
    return guidance


def build_anchor_weights(
    baseline: torch.Tensor,
    guidance_target: torch.Tensor,
    start_weight: float,
    end_weight: float,
    release_power: float,
) -> torch.Tensor:
    if baseline.shape != guidance_target.shape:
        raise ValueError("baseline and guidance_target must share the same shape for anchor weighting")
    horizon = baseline.shape[1]
    start = max(start_weight, 0.0)
    end = max(end_weight, 0.0)
    if horizon == 0:
        return torch.zeros(0, dtype=torch.float32)
    release_power = max(float(release_power), 1e-3)
    per_step_delta = (guidance_target - baseline).abs().amax(dim=0)
    max_delta_value = float(per_step_delta.max().item())
    if max_delta_value <= 1e-8:
        progress = torch.linspace(0.0, 1.0, steps=horizon, dtype=torch.float32)
    else:
        progress = (per_step_delta / max_delta_value).clamp(0.0, 1.0)
    progress = progress.to(torch.float32).pow(release_power)
    start_tensor = torch.full_like(progress, start)
    end_tensor = torch.full_like(progress, end)
    weights = torch.lerp(start_tensor, end_tensor, progress)
    return weights.clamp_min(0.0)


def compute_hop_node_weights(adjacency: torch.Tensor, target_node: int, focus_percent: float) -> torch.Tensor:
    """Distribute node weights: target gets focus_percent, others share the remainder by inverse hop distance."""

    if adjacency.dim() == 3:
        adjacency = adjacency[0]
    num_nodes = adjacency.shape[0]
    if target_node < 0 or target_node >= num_nodes:
        raise ValueError(f"target_adjust_node {target_node} is out of range for {num_nodes} nodes")

    target_share = float(max(0.0, min(focus_percent / 100.0, 1.0)))
    remaining_share = max(0.0, 1.0 - target_share)

    graph = (adjacency.detach().cpu() > 0).to(torch.bool)
    distances = torch.full((num_nodes,), float("inf"))
    distances[target_node] = 0.0
    frontier = [target_node]
    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            neighbors = torch.nonzero(graph[node], as_tuple=False).flatten().tolist()
            for nb in neighbors:
                if not torch.isfinite(distances[nb]).item():
                    distances[nb] = distances[node] + 1.0
                    next_frontier.append(nb)
        frontier = next_frontier

    inv_dist = torch.zeros(num_nodes, dtype=torch.float32)
    reachable_mask = torch.isfinite(distances) & (distances > 0)
    inv_dist[reachable_mask] = 1.0 / distances[reachable_mask]
    inv_total = float(inv_dist.sum().item())
    if inv_total > 0 and remaining_share > 0:
        inv_dist = inv_dist * (remaining_share / inv_total)
    else:
        inv_dist.zero_()

    weights = inv_dist
    weights[target_node] = target_share
    total = float(weights.sum().item())
    if total <= 0:
        return torch.ones(num_nodes, dtype=torch.float32) / max(num_nodes, 1)
    return weights / total


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
    lag_target: Optional[torch.Tensor] = None,
    plot_lag_steps: Optional[int] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping plot generation.")
        return

    node_index = max(0, min(node_index, original_prediction.shape[0] - 1))
    horizon = original_prediction.shape[1]
    steps = list(range(1, horizon + 1))
    lag_steps: list[int] = []
    lag_values = None
    if lag_target is not None and lag_target.dim() == 2 and lag_target.shape[0] > node_index:
        lag_values = lag_target[node_index].detach().cpu().numpy()
        # Limit the number of lag steps to plot if specified
        if plot_lag_steps is not None and plot_lag_steps > 0:
            num_lag_steps = min(plot_lag_steps, lag_target.shape[1])
            lag_values = lag_values[-num_lag_steps:]  # Take the most recent N steps
        else:
            num_lag_steps = lag_target.shape[1]
        lag_steps = list(range(-num_lag_steps + 1, 1))

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
    if lag_values is not None:
        gt_steps = lag_steps + steps
        gt_values = np.concatenate([lag_values, ground_truth[node_index].detach().cpu().numpy()])
    else:
        gt_steps = steps
        gt_values = ground_truth[node_index].detach().cpu().numpy()
    plt.plot(gt_steps, gt_values, label="Ground truth", linestyle=":", linewidth=2)
    if guidance_target is not None:
        plt.plot(
            steps,
            guidance_target[node_index].detach().cpu().numpy(),
            label="Guidance target",
            linestyle="-.",
            linewidth=2,
        )
    plt.scatter(
        [horizon],
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


def save_top_cf_window_plot(
    plot_path: Path,
    original_window: torch.Tensor,
    cf_window: torch.Tensor,
    top_k: int = 10,
    feature: int = 0,
    ncols: int = 5,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping top-k past window plot.")
        return

    if original_window.shape != cf_window.shape:
        raise ValueError(f"Shape mismatch between original ({original_window.shape}) and counterfactual ({cf_window.shape}) windows")
    lag, num_nodes, num_features = original_window.shape
    if num_nodes == 0 or lag == 0:
        print("Empty window; skipping top-k past window plot.")
        return

    feature_idx = max(0, min(feature, num_features - 1))
    top_k = max(1, min(int(top_k), num_nodes))
    diffs = (cf_window - original_window).abs().mean(dim=(0, 2))
    top_nodes = torch.argsort(diffs, descending=True)[:top_k]

    ncols = max(1, int(ncols))
    nrows = math.ceil(top_k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharex=True)
    axes = axes.flatten()
    time_steps = np.arange(lag)

    for ax, node in zip(axes, top_nodes):
        idx = int(node.item())
        ax.plot(
            time_steps,
            original_window[:, idx, feature_idx].detach().cpu().numpy(),
            label="Original",
            linewidth=2,
        )
        ax.plot(
            time_steps,
            cf_window[:, idx, feature_idx].detach().cpu().numpy(),
            label="Counterfactual",
            linestyle="--",
            linewidth=2,
        )
        ax.set_title(f"Node {idx} | Δ={diffs[idx]:.3f}")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(top_nodes) :]:
        ax.axis("off")

    axes[0].legend()
    fig.suptitle(f"Top {top_k} nodes by mean abs edit (feature {feature_idx})", y=1.02)
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved top-k past-window plot to {plot_path}")


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
    model_type = checkpoint_args.get("model", "stgcn")
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
        "model": model_type,
    }
    return model, bundle, metadata


def run_counterfactual_command(args: argparse.Namespace) -> None:
    device = forecaster_module.resolve_device(args.device, args.gpu_ids)
    forecaster_path = Path(args.forecaster_checkpoint)
    diffusion_path = Path(args.diffusion_checkpoint)

    forecaster, bundle, dataset_meta = load_forecaster_from_checkpoint(
        forecaster_path, device, Path(args.data_root) if args.data_root else None
    )
    model_type = dataset_meta.get("model", "stgcn")

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
    forecaster_input = prepare_forecaster_input(past_window_batch)
    with torch.no_grad():
        baseline_forecast = (
            forecaster_module.forward_pass(forecaster, forecaster_input, model_type)
            .squeeze(0)
            .detach()
            .cpu()
            .float()
        )
    baseline_forecast = truncate_horizon(baseline_forecast, cf_horizon)

    target_ch = dataset_meta.get("target_channel", 0)
    if target_ch < 0 or target_ch >= past_window.shape[-1]:
        raise ValueError(f"target_channel {target_ch} is out of range for past window features {past_window.shape[-1]}")

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
        if args.guidance_path_strategy == "linear":
            guidance_target = build_linear_guidance_target(
                baseline_forecast,
                adjusted_target,
                past_window,
                args.target_adjust_node,
                target_ch,
            )
        else:
            guidance_target = build_diffusion_guidance_target(
                baseline_forecast,
                adjusted_target,
                past_window,
                args.target_adjust_node,
                target_ch,
                impute_steps=args.guidance_impute_steps,
                beta_schedule=args.guidance_impute_schedule,
                beta_start=diffusion.config.beta_start,
                beta_end=diffusion.config.beta_end,
            )
    anchor_start = max(args.anchor_start_weight, 0.0)
    anchor_end = max(args.anchor_end_weight, 0.0)
    if guidance_target is not None:
        anchor_weights = build_anchor_weights(
            baseline_forecast,
            guidance_target,
            anchor_start,
            anchor_end,
            args.anchor_release_power,
        )
    else:
        anchor_weights = torch.linspace(anchor_start, anchor_end, steps=cf_horizon, dtype=torch.float32)
    node_weights = None
    if args.target_adjust_node >= 0:
        node_weights = compute_hop_node_weights(bundle.adjacency, args.target_adjust_node, args.target_focus_percent)
    mask = prepare_mask(past_window.shape[0], bundle.num_nodes, bundle.num_features, args.mask_path)
    if args.target_adjust_node >= 0:
        # Prevent direct edits to the target node; guidance can only modify other nodes.
        mask[:, args.target_adjust_node, :] = 0.0

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
        node_weights=node_weights,
        model_type=model_type,
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
        cf_input = prepare_forecaster_input(samples)
        cf_preds = forecaster_module.forward_pass(forecaster, cf_input, model_type)
        cf_preds = cf_preds[:, :, :cf_horizon]
        node_w = node_weights.to(device) if node_weights is not None else torch.ones(bundle.num_nodes, device=device)
        node_w = node_w / node_w.sum().clamp(min=1e-8)
        diff_sq = (cf_preds - target_batched) ** 2
        per_node = diff_sq.mean(dim=2)
        mse = (per_node * node_w.view(1, -1)).sum(dim=1)
        print(f"Counterfactual guidance weighted MSE per sample: {mse.cpu().numpy()}")

    cf_preds_cpu = cf_preds.detach().cpu().float()
    mse_cpu = mse.detach().cpu()
    best_idx = int(torch.argmin(mse_cpu).item())
    best_cf_prediction = cf_preds_cpu[best_idx]
    best_cf_window = samples[best_idx].detach().cpu().float()

    plot_node = args.plot_node if args.plot_node is not None else (args.target_adjust_node if args.target_adjust_node >= 0 else 0)
    plot_path = Path(args.plot_path) if args.plot_path else Path(args.output_path).with_name(Path(args.output_path).stem + "_plot.png")
    target_ch = dataset_meta.get("target_channel", 0)
    lag_target = None
    if 0 <= target_ch < past_window.shape[-1]:
        lag_target = past_window[:, :, target_ch].permute(1, 0).contiguous()
    if adjusted_target.shape[0] > 0 and adjusted_target.shape[1] > 0:
        save_prediction_plot(
            plot_path,
            plot_node,
            baseline_forecast,
            best_cf_prediction,
            adjusted_target,
            default_target,
            guidance_target,
            lag_target=lag_target,
            plot_lag_steps=args.plot_lag_steps,
        )

    top_window_plot_path = Path(args.plot_top_cf_window_path) if args.plot_top_cf_window_path else None
    if top_window_plot_path is not None:
        save_top_cf_window_plot(
            top_window_plot_path,
            past_window.detach().cpu().float(),
            best_cf_window,
            top_k=args.plot_top_cf_window_k,
            feature=args.plot_top_cf_window_feature,
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
        "target_focus_percent": args.target_focus_percent,
        "use_predicted_target": args.use_predicted_target,
        "plot_path": str(plot_path),
        "top_cf_window_plot_path": str(top_window_plot_path) if top_window_plot_path else None,
        "counterfactual_mse": mse_cpu,
        "best_sample_index": best_idx,
        "best_counterfactual_prediction": best_cf_prediction,
        "best_counterfactual_window": best_cf_window,
        "cf_horizon": cf_horizon,
        "anchor_weights": anchor_weights,
        "anchor_release_power": args.anchor_release_power,
        "guidance_target": guidance_target,
        "node_weights": node_weights.cpu() if node_weights is not None else None,
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
