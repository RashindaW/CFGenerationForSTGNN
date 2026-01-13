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
from counterfactual.subgraph import (
    RandomWalkConfig,
    build_node_weight_vector,
    build_random_walk_subgraphs,
    ensure_subgraph_cache,
    load_subgraph,
    random_walk_subgraph,
    save_subgraphs,
)
from models.causal_forecaster import load_feature_names
from preprocessing.data_reader import TemporalDatasetBundle, load_dataset
from preprocessing.graphwavenet_utils import StandardScaler
from train import build_dataloaders, train_pipeline, test_pipeline


def add_forecaster_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("forecaster", help="Train or evaluate an ST-GNN forecaster.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["stgcn", "graphwavenet", "mstgcn", "astgcn", "causal_forecaster"],
        default="stgcn",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["METRLA", "PEMSBAY", "TEP", "METRLA_15", "METRLA_30", "METRLA_SUB", "METRLA_SUB_15", "METRLA_SUB_30"],
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
    parser.add_argument(
        "--temporal_encoder",
        type=str,
        choices=["lstm", "tcn", "transformer"],
        default="lstm",
        help="Temporal encoder for causal_forecaster.",
    )
    parser.add_argument("--temporal_layers", type=int, default=1, help="Temporal encoder layers for causal_forecaster.")
    parser.add_argument("--attention_heads", type=int, default=4, help="Attention heads for causal_forecaster.")
    parser.add_argument("--decoder_layers", type=int, default=2, help="Decoder MLP layers for causal_forecaster.")
    parser.add_argument("--fusion_rounds", type=int, default=1, help="Causal fusion rounds for causal_forecaster.")
    parser.add_argument(
        "--spatial_norm",
        type=str,
        choices=["sym", "row"],
        default="sym",
        help="Adjacency normalization for causal_forecaster spatial GNN.",
    )
    parser.add_argument(
        "--control_nodes",
        type=str,
        default=None,
        help="Comma-separated global node indices used as control nodes.",
    )
    parser.add_argument(
        "--manip_nodes",
        type=str,
        default=None,
        help="Comma-separated global node indices used as manipulated nodes.",
    )
    parser.add_argument(
        "--target_nodes",
        type=str,
        default=None,
        help="Comma-separated global node indices used as target nodes (subset of manipulated).",
    )
    parser.add_argument("--tcn_dilation_base", type=int, default=2, help="TCN dilation base for causal_forecaster.")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated CUDA device IDs, e.g., '0,1'.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_channel", type=int, default=0)
    parser.add_argument(
        "--loss_focus",
        type=str,
        choices=["full", "last"],
        default="full",
        help="Compute loss/metrics over the full horizon or only the final step.",
    )
    parser.add_argument(
        "--lag_last_weight_percent",
        type=float,
        default=None,
        help="Percent of input weight assigned to the last lag step; remaining weight is spread across earlier steps.",
    )
    parser.add_argument(
        "--neighbor_only_inputs",
        action="store_true",
        help="Use only neighbor history by pre-aggregating inputs with an adjacency matrix that excludes self loops.",
    )
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
        choices=["METRLA", "PEMSBAY", "TEP", "METRLA_15", "METRLA_30", "METRLA_SUB", "METRLA_SUB_15", "METRLA_SUB_30"],
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


def add_subgraph_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("subgraphs", help="Precompute random-walk subgraphs for each node.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["METRLA", "PEMSBAY", "TEP", "METRLA_15", "METRLA_30", "METRLA_SUB", "METRLA_SUB_15", "METRLA_SUB_30"],
        default="METRLA",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Root directory containing dataset folders.")
    parser.add_argument("--output_dir", type=str, default=None, help="Override subgraph output directory.")
    parser.add_argument("--num_walks", type=int, default=128, help="Number of random walks per node.")
    parser.add_argument("--walk_length", type=int, default=8, help="Steps per walk.")
    parser.add_argument("--restart_prob", type=float, default=0.15, help="Restart probability during walks.")
    parser.add_argument("--top_k", type=int, default=16, help="Keep the top-k visited neighbors per node.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild and overwrite any existing cache.")
    parser.set_defaults(handler=run_subgraph_command)
    return parser


def add_counterfactual_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("counterfactual", help="Generate diffusion-guided counterfactual past windows.")
    parser.add_argument("--forecaster_checkpoint", type=str, required=True)
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)
    parser.add_argument(
        "--short_forecaster_checkpoint",
        type=str,
        default=None,
        help="One-step forecaster checkpoint used for iterative counterfactual guidance.",
    )
    parser.add_argument("--data_root", type=str, default=None, help="Optional override for stored dataset root.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--mask_path", type=str, default=None, help="Optional path to a numpy mask of shape (T, N, F).")
    parser.add_argument("--target_path", type=str, default=None, help="Optional path to a numpy target (H, N).")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--iterative_guidance",
        action="store_true",
        help="Iteratively edit only the last lag step using a one-step forecaster.",
    )
    parser.add_argument(
        "--iterative_steps",
        type=int,
        default=None,
        help="Number of iterative steps (defaults to cf_horizon).",
    )
    parser.add_argument("--lambda_scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--temporal_weight", type=float, default=1e-3)
    parser.add_argument("--spatial_weight", type=float, default=1e-3)
    parser.add_argument("--control_weight", type=float, default=0.0)
    parser.add_argument("--rate_limit", type=float, default=None)
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
        help="Percentage of forecast loss weight given to the adjusted node; remaining weight is distributed according to the chosen node-loss strategy.",
    )
    parser.add_argument(
        "--node_loss_strategy",
        type=str,
        choices=["subgraph", "hop", "uniform"],
        default="subgraph",
        help="How to distribute forecast loss across nodes for guidance.",
    )
    parser.add_argument(
        "--subgraph_dir",
        type=str,
        default=None,
        help="Directory containing per-node subgraph .npy files (defaults to <data_root>/<dataset>/subgraphs_random_walk).",
    )
    parser.add_argument("--subgraph_num_walks", type=int, default=128, help="Random walks per node for subgraph cache.")
    parser.add_argument("--subgraph_walk_length", type=int, default=8, help="Steps per random walk.")
    parser.add_argument("--subgraph_restart_prob", type=float, default=0.15, help="Restart probability for random walks.")
    parser.add_argument("--subgraph_top_k", type=int, default=16, help="Top-k visited neighbors to keep per node.")
    parser.add_argument(
        "--subgraph_spillover_percent",
        type=float,
        default=5.0,
        help="Percent of remaining (non-target) weight spilled uniformly outside the subgraph.",
    )
    parser.add_argument("--subgraph_seed", type=int, default=42, help="Seed for building subgraph cache.")
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
        "--metrics_path",
        type=str,
        default=None,
        help="Optional path to save iterative counterfactual metrics CSV (defaults to <output_path>_metrics.csv).",
    )
    parser.add_argument(
        "--metrics_plot_path",
        type=str,
        default=None,
        help="Optional path to save iterative counterfactual metrics plot (defaults to <output_path>_metrics.png).",
    )
    parser.add_argument(
        "--metrics_eps",
        type=float,
        default=1e-6,
        help="Absolute-change threshold for counting edited nodes in metrics.",
    )
    parser.add_argument(
        "--metrics_input",
        type=str,
        choices=["baseline", "guidance", "adjusted", "original", "actions"],
        default="baseline",
        help="Which series to treat as the input for metrics in iterative guidance.",
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
        "--guidance_impute_samples",
        type=int,
        default=10,
        help="Number of imputed guidance trajectories to sample per node and select the smoothest.",
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
        choices=["diffusion", "linear", "dc_shift", "step"],
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
    add_subgraph_subcommand(subparsers)
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


def write_training_command_file(directory: Path, filename: str = "trainingCommand.txt") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    command_path = directory / filename
    command_path.write_text(_format_training_command() + "\n")
    return command_path


def safe_torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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
    checkpoint = safe_torch_load(path, device)
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
    command_dir = Path(__file__).resolve().parent / "training_commands" / checkpoint_path.parent.name
    write_training_command_file(command_dir)
    metrics_csv = checkpoint_path.parent / "metrics.csv"
    best_val = float("inf")
    start_epoch = 1
    if args.checkpoint:
        ckpt = safe_torch_load(Path(args.checkpoint), device)
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


def run_subgraph_command(args: argparse.Namespace) -> None:
    dataset_dir = resolve_dataset_dir(args.dataset, args.data_root)
    adj_path = dataset_dir / "adj_mat.npy"
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency file not found at {adj_path}")

    adjacency = np.load(adj_path)
    cache_dir = Path(args.output_dir) if args.output_dir else default_subgraph_dir(args.dataset, args.data_root)
    config = RandomWalkConfig(
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        restart_prob=args.restart_prob,
        top_k=args.top_k,
        seed=args.seed,
    )
    if args.overwrite:
        subgraphs = build_random_walk_subgraphs(adjacency, config)
        save_subgraphs(subgraphs, cache_dir, config)
        print(f"Rebuilt subgraph cache at {cache_dir}")
    else:
        ensure_subgraph_cache(adjacency, cache_dir, config)
        print(f"Subgraph cache ready at {cache_dir}")


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


def inverse_target_scale(tensor: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    """Inverse scale a (nodes, horizon) tensor for the target channel."""

    arr = tensor.detach().cpu().numpy()
    arr = scaler.inverse_transform(arr)
    return torch.from_numpy(arr).to(tensor.dtype)


def inverse_target_feature(tensor: torch.Tensor, target_channel: int, scaler: StandardScaler) -> torch.Tensor:
    """Inverse scale only the target channel of a (T, N, F) tensor."""

    arr = tensor.detach().cpu().numpy()
    arr[..., target_channel] = scaler.inverse_transform(arr[..., target_channel])
    return torch.from_numpy(arr).to(tensor.dtype)


def _temporal_smooth_1d(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return x
    left = torch.cat([x[:1], x[:-1]])
    right = torch.cat([x[1:], x[-1:]])
    return (left + x + right) / 3.0


def _temporal_roughness_1d(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return x.new_tensor(0.0)
    diffs = x[1:] - x[:-1]
    return torch.mean(diffs.pow(2))


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
    num_samples: int = 10,
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
    num_samples = int(max(1, num_samples))
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
        best_path = None
        best_score = None
        for _ in range(num_samples):
            imputed = diffusion_impute_path(start_val, end_val, prior_path, betas)
            score = float(_temporal_roughness_1d(imputed).item())
            if best_score is None or score < best_score:
                best_score = score
                best_path = imputed
        guidance[node] = prior_path if best_path is None else best_path

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


def build_dc_shift_guidance_target(
    baseline: torch.Tensor,
    adjusted_target: torch.Tensor,
    past_window: torch.Tensor,
    target_node: int,
    target_channel: int,
) -> torch.Tensor:
    """
    Use a flat guidance path at the adjusted target endpoint for the chosen node(s);
    other nodes follow the baseline forecast.
    """

    if baseline.shape != adjusted_target.shape:
        raise ValueError("baseline and adjusted_target must share the same shape for DC-shift guidance")
    horizon = baseline.shape[1]
    if horizon == 0:
        return baseline.clone()

    num_nodes = baseline.shape[0]
    if target_node >= num_nodes or target_node < -1:
        raise ValueError(f"target_adjust_node {target_node} is out of range for {num_nodes} nodes")
    if target_channel < 0 or target_channel >= past_window.shape[-1]:
        raise ValueError(f"target_channel {target_channel} is out of range for past window features {past_window.shape[-1]}")

    guidance = baseline.clone()
    target_nodes = range(num_nodes) if target_node < 0 else [target_node]

    for node in target_nodes:
        end_val = adjusted_target[node, -1].to(baseline.device, baseline.dtype)
        guidance[node] = end_val

    return guidance


def build_step_guidance_target(
    baseline: torch.Tensor,
    adjusted_target: torch.Tensor,
    past_window: torch.Tensor,
    target_node: int,
    target_channel: int,
) -> torch.Tensor:
    """
    Hold the last observed lag value for half the horizon, then step to the adjusted target.
    """

    if baseline.shape != adjusted_target.shape:
        raise ValueError("baseline and adjusted_target must share the same shape for step guidance")
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

    if horizon == 1:
        for node in target_nodes:
            guidance[node, 0] = adjusted_target[node, -1].to(baseline.device, baseline.dtype)
        return guidance

    step_idx = max(1, horizon // 2)
    for node in target_nodes:
        start_val = start_values[node]
        end_val = adjusted_target[node, -1].to(baseline.device, baseline.dtype)
        guidance[node, :step_idx] = start_val
        guidance[node, step_idx:] = end_val

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


def resolve_dataset_dir(dataset: str, data_root: Optional[str | Path]) -> Path:
    base = Path(data_root) if data_root is not None else Path(__file__).resolve().parent / "preprocessing" / "data"
    return base / dataset.upper()


def default_subgraph_dir(dataset: str, data_root: Optional[str | Path]) -> Path:
    return resolve_dataset_dir(dataset, data_root) / "subgraphs_random_walk"


def resolve_tep_control_nodes(dataset: str, data_root: Optional[str | Path], num_nodes: int) -> np.ndarray:
    if dataset.upper() != "TEP":
        return np.array([], dtype=np.int64)
    dataset_dir = resolve_dataset_dir(dataset, data_root)
    feature_names = load_feature_names(dataset_dir)
    control_nodes: List[int] = []
    if feature_names:
        control_nodes = [idx for idx, name in enumerate(feature_names) if name.lower().startswith("xmv_")]
    if not control_nodes:
        start = max(num_nodes - 11, 0)
        control_nodes = list(range(start, num_nodes))
        if control_nodes:
            print("Warning: TEP control node names not found; defaulting to the last 11 nodes.")
    return np.array(control_nodes, dtype=np.int64)


def prepare_node_weights(
    adjacency: torch.Tensor,
    target_node: int,
    focus_percent: float,
    strategy: str,
    dataset: str,
    data_root: Optional[str | Path],
    subgraph_args: argparse.Namespace,
) -> Optional[torch.Tensor]:
    if target_node < 0:
        return None

    adj_for_weights = adjacency[0] if adjacency.dim() == 3 else adjacency
    num_nodes = adj_for_weights.shape[0]
    if target_node >= num_nodes:
        raise ValueError(f"target_adjust_node {target_node} is out of range for {num_nodes} nodes")
    if strategy == "uniform":
        weights = torch.ones(num_nodes, dtype=torch.float32)
        return weights / weights.sum().clamp(min=1e-8)

    if strategy == "hop":
        return compute_hop_node_weights(adjacency, target_node, focus_percent)

    if strategy == "subgraph" and dataset.upper() == "TEP":
        control_nodes = resolve_tep_control_nodes(dataset, data_root, num_nodes)
        if target_node >= 0:
            control_nodes = control_nodes[control_nodes != target_node]
        if control_nodes.size == 0:
            weights = torch.ones(num_nodes, dtype=torch.float32)
            return weights / weights.sum().clamp(min=1e-8)
        spillover_fraction = max(0.0, subgraph_args.subgraph_spillover_percent / 100.0)
        subgraph_weights = np.ones(control_nodes.size, dtype=np.float32)
        return build_node_weight_vector(
            num_nodes=num_nodes,
            target_node=target_node,
            target_share=focus_percent / 100.0,
            subgraph_nodes=control_nodes,
            subgraph_weights=subgraph_weights,
            spillover_fraction=spillover_fraction,
        )

    subgraph_cache = Path(subgraph_args.subgraph_dir) if subgraph_args.subgraph_dir else default_subgraph_dir(dataset, data_root)
    rw_config = RandomWalkConfig(
        num_walks=subgraph_args.subgraph_num_walks,
        walk_length=subgraph_args.subgraph_walk_length,
        restart_prob=subgraph_args.subgraph_restart_prob,
        top_k=subgraph_args.subgraph_top_k,
        seed=subgraph_args.subgraph_seed,
    )
    adj_np = adjacency.detach().cpu().numpy()
    if adj_np.ndim == 3:
        adj_np = adj_np[0]
    try:
        ensure_subgraph_cache(adj_np, subgraph_cache, rw_config)
    except Exception as exc:  # pragma: no cover - cache writes may be skipped in read-only envs
        print(f"Warning: could not build subgraph cache at {subgraph_cache}: {exc}")
    subgraph_path = subgraph_cache / f"node_{target_node}.npy"
    if subgraph_path.exists():
        nodes, weights = load_subgraph(subgraph_path)
    else:
        nodes, weights = random_walk_subgraph(adj_np, target_node, rw_config)
    spillover_fraction = max(0.0, subgraph_args.subgraph_spillover_percent / 100.0)
    return build_node_weight_vector(
        num_nodes=adj_np.shape[0],
        target_node=target_node,
        target_share=focus_percent / 100.0,
        subgraph_nodes=nodes,
        subgraph_weights=weights,
        spillover_fraction=spillover_fraction,
    )


def load_subgraph_nodes_for_target(
    adjacency: torch.Tensor,
    target_node: int,
    subgraph_cache: Path,
    config: RandomWalkConfig,
    dataset: str,
    data_root: Optional[str | Path],
) -> np.ndarray:
    if target_node < 0:
        return np.array([], dtype=np.int64)
    adj_np = adjacency.detach().cpu().numpy()
    if adj_np.ndim == 3:
        adj_np = adj_np[0]
    if dataset.upper() == "TEP":
        control_nodes = resolve_tep_control_nodes(dataset, data_root, adj_np.shape[0])
        if target_node >= 0:
            control_nodes = control_nodes[control_nodes != target_node]
        return control_nodes
    try:
        ensure_subgraph_cache(adj_np, subgraph_cache, config)
    except Exception as exc:  # pragma: no cover - cache writes may be skipped in read-only envs
        print(f"Warning: could not build subgraph cache at {subgraph_cache}: {exc}")
    subgraph_path = subgraph_cache / f"node_{target_node}.npy"
    if subgraph_path.exists():
        nodes, _ = load_subgraph(subgraph_path)
    else:
        nodes, _ = random_walk_subgraph(adj_np, target_node, config)
    return nodes


def truncate_horizon(tensor: torch.Tensor, horizon: Optional[int]) -> torch.Tensor:
    if horizon is None or tensor.shape[1] <= horizon:
        return tensor
    if horizon <= 0:
        raise ValueError("cf_horizon must be positive")
    return tensor[:, :horizon].contiguous()


def select_metrics_input(
    source: str,
    baseline: Optional[torch.Tensor],
    guidance: Optional[torch.Tensor],
    adjusted: Optional[torch.Tensor],
    original: Optional[torch.Tensor],
    actions: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    options = {
        "baseline": baseline,
        "guidance": guidance,
        "adjusted": adjusted,
        "original": original,
        "actions": actions,
    }
    selected = options.get(source)
    if selected is None:
        print(f"Metrics input '{source}' is not available; skipping metrics.")
    return selected


def build_counterfactual_metrics(
    input_series: torch.Tensor,
    output_series: torch.Tensor,
    eps: float,
) -> Dict[str, Any]:
    if input_series.shape != output_series.shape:
        raise ValueError(
            f"Metrics shape mismatch: input {tuple(input_series.shape)} vs output {tuple(output_series.shape)}"
        )
    if input_series.dim() != 2:
        raise ValueError("Metrics inputs must be shaped (nodes, steps)")

    input_series = input_series.detach().cpu().float()
    output_series = output_series.detach().cpu().float()
    steps = output_series.shape[1]

    diff = output_series - input_series
    abs_diff = diff.abs()
    changed_vs_input = (abs_diff > eps).sum(dim=0).to(torch.int64)
    l2_vs_input = torch.sqrt((diff ** 2).sum(dim=0))
    avg_l2_vs_input = torch.where(
        changed_vs_input > 0,
        l2_vs_input / changed_vs_input.to(l2_vs_input.dtype),
        torch.zeros_like(l2_vs_input),
    )

    if steps > 1:
        diff_prev = output_series[:, 1:] - output_series[:, :-1]
        abs_prev = diff_prev.abs()
        changed_vs_prev = torch.zeros(steps, dtype=torch.int64)
        changed_vs_prev[0] = changed_vs_input[0]
        changed_vs_prev[1:] = (abs_prev > eps).sum(dim=0).to(torch.int64)
        l2_vs_prev = torch.zeros(steps, dtype=l2_vs_input.dtype)
        l2_vs_prev[0] = l2_vs_input[0]
        l2_vs_prev[1:] = torch.sqrt((diff_prev ** 2).sum(dim=0))
    else:
        changed_vs_prev = changed_vs_input.clone()
        l2_vs_prev = l2_vs_input.clone()

    avg_l2_vs_prev = torch.where(
        changed_vs_prev > 0,
        l2_vs_prev / changed_vs_prev.to(l2_vs_prev.dtype),
        torch.zeros_like(l2_vs_prev),
    )

    total_changed_vs_input = int(changed_vs_input.sum().item())
    total_changed_vs_prev = int(changed_vs_prev.sum().item())
    total_l2_vs_input = float(l2_vs_input.sum().item())
    total_l2_vs_prev = float(l2_vs_prev.sum().item())
    total_avg_l2_vs_input = total_l2_vs_input / total_changed_vs_input if total_changed_vs_input > 0 else 0.0
    total_avg_l2_vs_prev = total_l2_vs_prev / total_changed_vs_prev if total_changed_vs_prev > 0 else 0.0

    rows = []
    for idx in range(steps):
        rows.append(
            {
                "step": idx + 1,
                "changed_nodes_vs_input": int(changed_vs_input[idx].item()),
                "changed_nodes_vs_prev": int(changed_vs_prev[idx].item()),
                "l2_vs_input": float(l2_vs_input[idx].item()),
                "avg_l2_per_changed_vs_input": float(avg_l2_vs_input[idx].item()),
                "l2_vs_prev": float(l2_vs_prev[idx].item()),
                "avg_l2_per_changed_vs_prev": float(avg_l2_vs_prev[idx].item()),
            }
        )

    summary = {
        "step": "total",
        "changed_nodes_vs_input": total_changed_vs_input,
        "changed_nodes_vs_prev": total_changed_vs_prev,
        "l2_vs_input": total_l2_vs_input,
        "avg_l2_per_changed_vs_input": total_avg_l2_vs_input,
        "l2_vs_prev": total_l2_vs_prev,
        "avg_l2_per_changed_vs_prev": total_avg_l2_vs_prev,
    }

    return {
        "steps": list(range(1, steps + 1)),
        "rows": rows,
        "summary": summary,
        "changed_vs_input": changed_vs_input,
        "changed_vs_prev": changed_vs_prev,
        "l2_vs_input": l2_vs_input,
        "l2_vs_prev": l2_vs_prev,
        "avg_l2_vs_input": avg_l2_vs_input,
    }


def save_counterfactual_metrics_csv(metrics_path: Path, metrics: Dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "changed_nodes_vs_input",
        "changed_nodes_vs_prev",
        "l2_vs_input",
        "avg_l2_per_changed_vs_input",
        "l2_vs_prev",
        "avg_l2_per_changed_vs_prev",
    ]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics["rows"]:
            writer.writerow(row)
        writer.writerow(metrics["summary"])
    print(f"Saved counterfactual metrics table to {metrics_path}")


def save_counterfactual_metrics_plot(metrics_plot_path: Path, metrics: Dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping counterfactual metrics plot.")
        return

    steps = metrics["steps"]
    changed_vs_input = metrics["changed_vs_input"].detach().cpu().numpy()
    changed_vs_prev = metrics["changed_vs_prev"].detach().cpu().numpy()
    l2_vs_input = metrics["l2_vs_input"].detach().cpu().numpy()
    avg_l2_vs_input = metrics["avg_l2_vs_input"].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(steps, changed_vs_input, marker="o", label="Changed vs input")
    axes[0].plot(steps, changed_vs_prev, marker="s", label="Changed vs previous")
    axes[0].set_ylabel("Changed nodes")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, l2_vs_input, marker="o", label="L2 vs input")
    axes[1].plot(steps, avg_l2_vs_input, marker="s", label="Avg L2 per changed")
    axes[1].set_xlabel("Iterative step")
    axes[1].set_ylabel("L2 distance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    metrics_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(metrics_plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved counterfactual metrics plot to {metrics_plot_path}")


def save_prediction_plot(
    plot_path: Path,
    node_index: int,
    original_prediction: torch.Tensor,
    cf_prediction: torch.Tensor,
    adjusted_target: torch.Tensor,
    ground_truth: torch.Tensor,
    guidance_target: Optional[torch.Tensor] = None,
    step_target: Optional[torch.Tensor] = None,
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
    if step_target is not None:
        plt.scatter(
            steps,
            step_target[node_index].detach().cpu().numpy(),
            label="Step target",
            marker="x",
            color="black",
            zorder=3,
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


def save_iterative_cf_window_plots(
    plot_path: Path,
    lag_values: torch.Tensor,
    horizon_ground_truth: torch.Tensor,
    horizon_counterfactual: torch.Tensor,
    target_node: int,
    per_page: int = 10,
    ncols: int = 5,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping iterative counterfactual plots.")
        return []

    if lag_values.dim() != 2:
        raise ValueError(f"lag_values must be (lag, nodes); got {lag_values.shape}")
    if horizon_ground_truth.dim() != 2 or horizon_counterfactual.dim() != 2:
        raise ValueError("horizon_ground_truth and horizon_counterfactual must be (nodes, horizon)")
    if horizon_ground_truth.shape != horizon_counterfactual.shape:
        raise ValueError("ground-truth and counterfactual horizons must have the same shape")
    if lag_values.shape[1] != horizon_ground_truth.shape[0]:
        raise ValueError("lag_values node count must match horizon node count")

    lag_len, num_nodes = lag_values.shape
    horizon_len = horizon_ground_truth.shape[1]
    nodes = list(range(num_nodes))
    if 0 <= target_node < num_nodes:
        nodes.remove(target_node)

    per_page = max(1, int(per_page))
    ncols = max(1, int(ncols))
    pages = math.ceil(len(nodes) / per_page)
    saved_paths: List[Path] = []

    lag_steps = np.arange(-lag_len + 1, 1)
    horizon_steps = np.arange(1, horizon_len + 1)
    full_steps = np.concatenate([lag_steps, horizon_steps])

    for page_idx in range(pages):
        page_nodes = nodes[page_idx * per_page : (page_idx + 1) * per_page]
        if not page_nodes:
            continue
        nrows = math.ceil(len(page_nodes) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax, node in zip(axes, page_nodes):
            lag_series = lag_values[:, node].detach().cpu().numpy()
            gt_series = horizon_ground_truth[node].detach().cpu().numpy()
            cf_series = horizon_counterfactual[node].detach().cpu().numpy()
            gt_full = np.concatenate([lag_series, gt_series])

            ax.plot(full_steps, gt_full, color="green", linestyle=":", linewidth=2, label="Ground truth")
            ax.plot(horizon_steps, cf_series, linewidth=3, label="Counterfactual")
            ax.set_title(f"Node {node}")
            ax.grid(True, alpha=0.3)

        for ax in axes[len(page_nodes) :]:
            ax.axis("off")

        axes[0].legend()
        title = "Counterfactual horizon vs ground truth (excluding target node)"
        fig.suptitle(title, y=1.02)
        fig.tight_layout()

        if pages == 1:
            save_path = plot_path
        elif page_idx == 0:
            save_path = plot_path
        else:
            save_path = plot_path.with_name(f"{plot_path.stem}_page{page_idx + 1}{plot_path.suffix}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        saved_paths.append(save_path)
        print(f"Saved iterative counterfactual plot to {save_path}")

    return saved_paths


def save_control_node_action_plots(
    output_dir: Path,
    target_node: int,
    control_nodes: List[int],
    prev_last_lag: torch.Tensor,
    edited_last_lag: torch.Tensor,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping control-node action plots.")
        return []

    if prev_last_lag.shape != edited_last_lag.shape:
        raise ValueError("prev_last_lag and edited_last_lag must have the same shape")
    if prev_last_lag.dim() != 2:
        raise ValueError("prev_last_lag and edited_last_lag must be shaped (nodes, steps)")

    steps = np.arange(1, prev_last_lag.shape[1] + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for node in control_nodes:
        if node == target_node or node < 0 or node >= prev_last_lag.shape[0]:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            steps,
            prev_last_lag[node].detach().cpu().numpy(),
            label="Previous last lag",
            color="tab:blue",
            linewidth=2,
        )
        ax.plot(
            steps,
            edited_last_lag[node].detach().cpu().numpy(),
            label="Edited last lag",
            color="tab:orange",
            linewidth=2,
        )
        ax.set_title(f"Node {node} last-lag edits")
        ax.set_xlabel("Iterative step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_path = output_dir / f"target_{target_node}_node_{node}.png"
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        saved.append(save_path)
    if saved:
        print(f"Saved control-node action plots to {output_dir}")
    return saved


def select_split_dataset(bundle: TemporalDatasetBundle, split: str):
    if split == "train":
        return bundle.train
    if split == "val":
        return bundle.val
    return bundle.test


def load_forecaster_from_checkpoint(path: Path, device: torch.device, data_root_override: Optional[Path] = None) -> Tuple[torch.nn.Module, TemporalDatasetBundle, Dict[str, Any]]:
    checkpoint = safe_torch_load(path, device)
    checkpoint_args = checkpoint.get("config", {}).get("args", {})

    dataset_name = checkpoint.get("dataset") or checkpoint_args.get("dataset")
    dataset_name = dataset_name.upper() if dataset_name else "METRLA"
    model_type = checkpoint_args.get("model", "stgcn")
    lag = checkpoint_args.get("lag", 12)
    horizon = checkpoint_args.get("horizon", 12)
    train_ratio = checkpoint_args.get("train_ratio", 0.7)
    val_ratio = checkpoint_args.get("val_ratio", 0.1)
    target_channel = checkpoint_args.get("target_channel", 0)
    lag_last_weight_percent = checkpoint_args.get("lag_last_weight_percent", None)
    neighbor_only_inputs = checkpoint_args.get("neighbor_only_inputs", False)
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
    lag_weights = forecaster_module.build_lag_weights(lag, lag_last_weight_percent)
    metadata = {
        "lag": lag,
        "horizon": horizon,
        "dataset": dataset_name,
        "target_channel": target_channel,
        "model": model_type,
        "lag_last_weight_percent": lag_last_weight_percent,
        "lag_weights": lag_weights,
        "neighbor_only_inputs": neighbor_only_inputs,
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
            forecaster_module.forward_pass(
                forecaster,
                forecaster_input,
                model_type,
                lag_weights=dataset_meta.get("lag_weights"),
                adjacency=bundle.adjacency,
                neighbor_only_inputs=dataset_meta.get("neighbor_only_inputs", False),
            )
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
        elif args.guidance_path_strategy == "dc_shift":
            guidance_target = build_dc_shift_guidance_target(
                baseline_forecast,
                adjusted_target,
                past_window,
                args.target_adjust_node,
                target_ch,
            )
        elif args.guidance_path_strategy == "step":
            guidance_target = build_step_guidance_target(
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
                num_samples=args.guidance_impute_samples,
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
    subgraph_cache_dir = Path(args.subgraph_dir) if args.subgraph_dir else default_subgraph_dir(dataset_meta["dataset"], args.data_root)
    node_weights = prepare_node_weights(
        adjacency=bundle.adjacency,
        target_node=args.target_adjust_node,
        focus_percent=args.target_focus_percent,
        strategy=args.node_loss_strategy,
        dataset=dataset_meta["dataset"],
        data_root=args.data_root,
        subgraph_args=args,
    )
    base_mask = prepare_mask(past_window.shape[0], bundle.num_nodes, bundle.num_features, args.mask_path)
    if args.target_adjust_node >= 0:
        # Prevent direct edits to the target node; guidance can only modify other nodes.
        base_mask[:, args.target_adjust_node, :] = 0.0

    sample_shape = torch.Size(past_window.shape)  # (T, N, F)

    guidance_config = GuidanceConfig(
        lambda_scale=args.lambda_scale,
        eta=args.eta,
        temporal_weight=args.temporal_weight,
        spatial_weight=args.spatial_weight,
        control_energy_weight=args.control_weight,
        rate_limit=args.rate_limit,
        anchor_start_weight=args.anchor_start_weight,
        anchor_end_weight=args.anchor_end_weight,
        anchor_loss_scale=args.anchor_loss_scale,
    )
    generator = CounterfactualGenerator(
        diffusion,
        adjacency=bundle.adjacency.to(device),
        device=device,
        temporal_context=build_temporal_context(dataset_meta["lag"], device),
    )

    if args.iterative_guidance:
        if not args.short_forecaster_checkpoint:
            raise ValueError("--short_forecaster_checkpoint is required when --iterative_guidance is set.")
        short_forecaster_path = Path(args.short_forecaster_checkpoint)
        short_forecaster, short_bundle, short_meta = load_forecaster_from_checkpoint(
            short_forecaster_path,
            device,
            Path(args.data_root) if args.data_root else None,
        )
        short_model_type = short_meta.get("model", "stgcn")
        if short_meta.get("horizon", 1) != 1:
            raise ValueError("Short-term forecaster must have horizon=1 for iterative guidance.")
        if short_meta.get("lag") != dataset_meta.get("lag"):
            raise ValueError("Short-term forecaster lag does not match the long-horizon forecaster lag.")
        if short_meta.get("dataset") and short_meta.get("dataset") != dataset_meta.get("dataset"):
            print("Warning: Short-term forecaster and long-horizon forecaster were trained on different datasets.")
        if short_bundle.num_nodes != bundle.num_nodes or short_bundle.num_features != bundle.num_features:
            raise ValueError("Short-term forecaster data shape does not match the long-horizon forecaster.")

        iter_steps = args.iterative_steps if args.iterative_steps is not None else cf_horizon
        if iter_steps <= 0:
            raise ValueError("iterative_steps must be positive")
        if iter_steps > cf_horizon:
            raise ValueError(f"iterative_steps {iter_steps} exceeds cf_horizon {cf_horizon}")

        node_w = node_weights.to(device) if node_weights is not None else torch.ones(bundle.num_nodes, device=device)
        node_w = node_w / node_w.sum().clamp(min=1e-8)

        current_window = past_window.to(device).float()
        best_windows: list[torch.Tensor] = []
        mse_per_step: list[float] = []
        best_indices: list[int] = []
        iterative_predictions = torch.zeros((bundle.num_nodes, iter_steps), device=device)
        iterative_actions = torch.zeros((bundle.num_nodes, iter_steps), device=device)
        iterative_prev_last_lag = torch.zeros((bundle.num_nodes, iter_steps), device=device)
        iterative_edited_last_lag = torch.zeros((bundle.num_nodes, iter_steps), device=device)
        iterative_targets = torch.zeros((bundle.num_nodes, iter_steps), device=device)
        if guidance_target is None:
            raise ValueError("guidance_target is required for iterative guidance.")
        target_series = guidance_target.to(device)

        edit_mask = torch.zeros_like(base_mask)
        edit_mask[-1] = 1.0
        edit_mask = edit_mask * base_mask
        edit_mask = edit_mask.to(device)

        # Identify control nodes vs dependent nodes based on edit mask
        # Control nodes: can be perturbed (edit_mask is non-zero at last timestep)
        # Dependent nodes: cannot be perturbed (edit_mask is zero, includes target node)
        node_is_control = (edit_mask[-1].sum(dim=-1) > 0)  # Shape (N,)
        node_is_dependent = ~node_is_control  # Shape (N,)

        for step in range(iter_steps):
            # Step 1: Shift left and duplicate last value for perturbation
            # This creates a window where position T-1 is a copy of position T-2
            shifted_window = torch.cat([current_window[1:], current_window[-1:].clone()], dim=0)

            # Record pre-perturbation value (from the duplicated last position)
            if 0 <= target_ch < shifted_window.shape[-1]:
                iterative_prev_last_lag[:, step] = shifted_window[-1, :, target_ch]

            # Get baseline prediction from shifted window (before perturbation)
            with torch.no_grad():
                step_input = prepare_forecaster_input(shifted_window.unsqueeze(0))
                step_pred = forecaster_module.forward_pass(
                    short_forecaster,
                    step_input,
                    short_model_type,
                    lag_weights=short_meta.get("lag_weights"),
                    adjacency=short_bundle.adjacency,
                    neighbor_only_inputs=short_meta.get("neighbor_only_inputs", False),
                )
                if step_pred.dim() == 2:
                    step_pred = step_pred.unsqueeze(-1)
                step_pred = step_pred[:, :, :1]
            step_pred = step_pred.squeeze(0)
            iterative_actions[:, step] = step_pred.squeeze(-1)

            step_target = target_series[:, step : step + 1]
            iterative_targets[:, step] = step_target.squeeze(-1)

            target_batched = step_target.unsqueeze(0).repeat(args.samples, 1, 1)
            mask_batched = edit_mask.unsqueeze(0).repeat(args.samples, 1, 1, 1)

            baseline_step = step_pred
            anchor_step = anchor_weights[step : step + 1] if anchor_weights is not None else None
            guidance = ForecastGuidance(
                forecaster=short_forecaster,
                target=target_batched,
                mask=mask_batched,
                adjacency=bundle.adjacency.to(device),
                config=guidance_config,
                lower_bounds=args.lower_bound,
                upper_bounds=args.upper_bound,
                baseline=baseline_step,
                anchor_weights=anchor_step,
                node_weights=node_weights,
                neighbor_only_inputs=short_meta.get("neighbor_only_inputs", False),
                model_type=short_model_type,
                lag_weights=short_meta.get("lag_weights"),
            )

            # Step 2: Perturb the shifted window (only last position is editable)
            fixed_values = shifted_window.unsqueeze(0).repeat(args.samples, 1, 1, 1)
            warm_start = fixed_values if args.warm_start else None
            samples = generator.generate(
                sample_shape=sample_shape,
                guidance=guidance,
                num_samples=args.samples,
                max_steps=args.max_steps,
                warm_start=warm_start,
                edit_mask=mask_batched,
                fixed_values=fixed_values,
            )

            # Evaluate samples and select best based on forecast MSE
            with torch.no_grad():
                cf_input = prepare_forecaster_input(samples)
                cf_preds = forecaster_module.forward_pass(
                    short_forecaster,
                    cf_input,
                    short_model_type,
                    lag_weights=short_meta.get("lag_weights"),
                    adjacency=short_bundle.adjacency,
                    neighbor_only_inputs=short_meta.get("neighbor_only_inputs", False),
                )
                if cf_preds.dim() == 2:
                    cf_preds = cf_preds.unsqueeze(-1)
                cf_preds = cf_preds[:, :, :1]
                diff_sq = (cf_preds - target_batched) ** 2
                per_node = diff_sq.mean(dim=2)
                mse = (per_node * node_w.view(1, -1)).sum(dim=1)

            best_idx = int(torch.argmin(mse).item())
            best_indices.append(best_idx)
            best_pred = cf_preds[best_idx]
            iterative_predictions[:, step] = best_pred.squeeze(-1)
            mse_per_step.append(float(mse[best_idx].item()))

            best_window = samples[best_idx]
            if 0 <= target_ch < best_window.shape[-1]:
                iterative_edited_last_lag[:, step] = best_window[-1, :, target_ch]
            best_windows.append(best_window.detach().cpu())

            # Step 3: Construct f*(T) - hybrid of perturbed control values and forecasted dependent values
            # - Control nodes: keep perturbed values from best_window[-1] (all features)
            # - Dependent nodes: use forecast for target channel, perturbed values for other channels
            #   (other channels of dependent nodes are unchanged since they weren't editable)
            next_step = best_window[-1].clone()  # Start with all perturbed values (correct for control nodes)
            if 0 <= target_ch < next_step.shape[-1]:
                # Only update dependent nodes' target channel with forecast
                # Control nodes keep their perturbed values for all features
                next_step[node_is_dependent, target_ch] = best_pred[node_is_dependent, 0]
            current_window = torch.cat([current_window[1:], next_step.unsqueeze(0)], dim=0)

        iterative_predictions_cpu = iterative_predictions.detach().cpu()
        iterative_prev_last_lag_cpu = iterative_prev_last_lag.detach().cpu()
        iterative_edited_last_lag_cpu = iterative_edited_last_lag.detach().cpu()
        iterative_targets_cpu = iterative_targets.detach().cpu()
        final_window = current_window.detach().cpu()
        baseline_slice = baseline_forecast[:, :iter_steps]
        adjusted_slice = adjusted_target[:, :iter_steps]
        default_slice = default_target[:, :iter_steps]
        guidance_slice = guidance_target[:, :iter_steps] if guidance_target is not None else None

        baseline_plot = inverse_target_scale(baseline_slice, bundle.scaler)
        iterative_plot = inverse_target_scale(iterative_predictions_cpu, bundle.scaler)
        adjusted_target_plot = inverse_target_scale(adjusted_slice, bundle.scaler)
        default_target_plot = inverse_target_scale(default_slice, bundle.scaler)
        guidance_target_plot = inverse_target_scale(guidance_slice, bundle.scaler) if guidance_slice is not None else None
        iterative_targets_plot = inverse_target_scale(iterative_targets_cpu, bundle.scaler)
        prev_last_lag_plot = inverse_target_scale(iterative_prev_last_lag_cpu, bundle.scaler)
        edited_last_lag_plot = inverse_target_scale(iterative_edited_last_lag_cpu, bundle.scaler)

        plot_node = args.plot_node if args.plot_node is not None else (args.target_adjust_node if args.target_adjust_node >= 0 else 0)
        plot_path = Path(args.plot_path) if args.plot_path else Path(args.output_path).with_name(Path(args.output_path).stem + "_plot.png")
        lag_target = None
        if 0 <= target_ch < past_window.shape[-1]:
            lag_target = past_window[:, :, target_ch].permute(1, 0).contiguous()
        lag_target_plot = inverse_target_scale(lag_target, bundle.scaler) if lag_target is not None else None

        if iterative_predictions_cpu.shape[1] > 0:
            save_prediction_plot(
                plot_path,
                plot_node,
                baseline_plot,
                iterative_plot,
                adjusted_target_plot,
                default_target_plot,
                guidance_target_plot,
                step_target=iterative_targets_plot,
                lag_target=lag_target_plot,
                plot_lag_steps=args.plot_lag_steps,
            )

        top_window_plot_path = Path(args.plot_top_cf_window_path) if args.plot_top_cf_window_path else None
        top_window_plot_paths: Optional[List[Path]] = None
        if top_window_plot_path is not None:
            lag_values = past_window[:, :, target_ch].permute(1, 0).contiguous().cpu()
            lag_values = inverse_target_scale(lag_values, bundle.scaler).permute(1, 0)
            horizon_ground_truth = truncate_horizon(target_future, iter_steps).cpu()
            horizon_ground_truth = inverse_target_scale(horizon_ground_truth, bundle.scaler)
            horizon_counterfactual = iterative_predictions_cpu[:, :iter_steps]
            horizon_counterfactual = inverse_target_scale(horizon_counterfactual, bundle.scaler)
            top_window_plot_paths = save_iterative_cf_window_plots(
                top_window_plot_path,
                lag_values,
                horizon_ground_truth,
                horizon_counterfactual,
                target_node=args.target_adjust_node,
                per_page=10,
                ncols=5,
            )

        control_action_dir = Path(args.output_path).with_name("control_node_actions")
        control_action_paths: Optional[List[Path]] = None
        if args.target_adjust_node >= 0 and iter_steps > 0:
            rw_config = RandomWalkConfig(
                num_walks=args.subgraph_num_walks,
                walk_length=args.subgraph_walk_length,
                restart_prob=args.subgraph_restart_prob,
                top_k=args.subgraph_top_k,
                seed=args.subgraph_seed,
            )
            control_nodes = load_subgraph_nodes_for_target(
                bundle.adjacency,
                args.target_adjust_node,
                subgraph_cache_dir,
                rw_config,
                dataset_meta["dataset"],
                args.data_root,
            )
            control_nodes_list = [int(node) for node in control_nodes.tolist()]
            if control_nodes_list:
                control_action_paths = save_control_node_action_plots(
                    control_action_dir,
                    args.target_adjust_node,
                    control_nodes_list,
                    prev_last_lag_plot,
                    edited_last_lag_plot,
                )

        metrics_path: Optional[Path] = None
        metrics_plot_path: Optional[Path] = None
        metrics_input = select_metrics_input(
            args.metrics_input,
            baseline_slice,
            guidance_slice,
            adjusted_slice,
            default_slice,
            iterative_actions,
        )
        if metrics_input is not None:
            output_base = Path(args.output_path)
            metrics_path = Path(args.metrics_path) if args.metrics_path else output_base.with_name(f"{output_base.stem}_metrics.csv")
            metrics_plot_path = (
                Path(args.metrics_plot_path) if args.metrics_plot_path else output_base.with_name(f"{output_base.stem}_metrics.png")
            )
            metrics = build_counterfactual_metrics(metrics_input, iterative_predictions_cpu, args.metrics_eps)
            save_counterfactual_metrics_csv(metrics_path, metrics)
            if metrics_plot_path is not None:
                save_counterfactual_metrics_plot(metrics_plot_path, metrics)

        output = {
            "mode": "iterative",
            "samples": None,
            "target": iterative_targets_cpu.unsqueeze(0),
            "mask": edit_mask.unsqueeze(0).cpu(),
            "guidance": asdict(guidance_config),
            "metadata": {
                "forecaster_checkpoint": str(forecaster_path),
                "short_forecaster_checkpoint": str(short_forecaster_path),
                "diffusion_checkpoint": str(diffusion_path),
                "dataset": dataset_meta,
                "diffusion_dataset": dataset_info,
                "split": args.split,
                "sample_index": args.sample_index,
            },
            "baseline_forecast": baseline_slice,
            "original_target": default_slice,
            "adjusted_target": adjusted_slice,
            "target_adjust_percent": args.target_adjust_percent,
            "target_adjust_offset": args.target_adjust_offset,
            "target_adjust_node": args.target_adjust_node,
            "target_focus_percent": args.target_focus_percent,
            "node_loss_strategy": args.node_loss_strategy,
            "subgraph_cache_dir": str(subgraph_cache_dir),
            "subgraph_params": {
                "num_walks": args.subgraph_num_walks,
                "walk_length": args.subgraph_walk_length,
                "restart_prob": args.subgraph_restart_prob,
                "top_k": args.subgraph_top_k,
                "spillover_percent": args.subgraph_spillover_percent,
                "seed": args.subgraph_seed,
            },
            "use_predicted_target": args.use_predicted_target,
            "plot_path": str(plot_path),
            "top_cf_window_plot_path": str(top_window_plot_path) if top_window_plot_path else None,
            "top_cf_window_plot_paths": [str(path) for path in top_window_plot_paths] if top_window_plot_paths else None,
            "metrics_path": str(metrics_path) if metrics_path else None,
            "metrics_plot_path": str(metrics_plot_path) if metrics_plot_path else None,
            "control_node_action_dir": str(control_action_dir) if control_action_paths else None,
            "control_node_action_paths": [str(path) for path in control_action_paths] if control_action_paths else None,
            "metrics_input": args.metrics_input,
            "metrics_eps": args.metrics_eps,
            "counterfactual_mse": None,
            "best_sample_index": None,
            "best_counterfactual_prediction": iterative_predictions_cpu,
            "best_counterfactual_window": final_window,
            "cf_horizon": cf_horizon,
            "iterative_steps": iter_steps,
            "anchor_weights": anchor_weights,
            "anchor_release_power": args.anchor_release_power,
            "guidance_target": guidance_slice,
            "node_weights": node_weights.cpu() if node_weights is not None else None,
            "iterative": {
                "predictions": iterative_predictions_cpu,
                "targets": iterative_targets_cpu,
                "actions": iterative_actions.detach().cpu(),
                "prev_last_lag": iterative_prev_last_lag_cpu,
                "edited_last_lag": iterative_edited_last_lag_cpu,
                "mse_per_step": torch.tensor(mse_per_step, dtype=torch.float32),
                "best_indices": best_indices,
                "best_windows": torch.stack(best_windows) if best_windows else None,
                "final_window": final_window,
            },
        }
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output, output_path)
        print(f"Saved iterative counterfactual samples to {output_path}")
        return

    mask = base_mask.to(device)
    target_batched = guidance_target.unsqueeze(0).repeat(args.samples, 1, 1).to(device)
    mask_batched = mask.unsqueeze(0).repeat(args.samples, 1, 1, 1).to(device)
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
        neighbor_only_inputs=dataset_meta.get("neighbor_only_inputs", False),
        model_type=model_type,
        lag_weights=dataset_meta.get("lag_weights"),
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
        cf_preds = forecaster_module.forward_pass(
            forecaster,
            cf_input,
            model_type,
            lag_weights=dataset_meta.get("lag_weights"),
            adjacency=bundle.adjacency,
            neighbor_only_inputs=dataset_meta.get("neighbor_only_inputs", False),
        )
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
    # Prepare unscaled versions for plotting.
    baseline_plot = inverse_target_scale(baseline_forecast, bundle.scaler)
    best_cf_prediction_plot = inverse_target_scale(best_cf_prediction, bundle.scaler)
    adjusted_target_plot = inverse_target_scale(adjusted_target, bundle.scaler)
    default_target_plot = inverse_target_scale(default_target, bundle.scaler)
    guidance_target_plot = inverse_target_scale(guidance_target, bundle.scaler) if guidance_target is not None else None

    plot_node = args.plot_node if args.plot_node is not None else (args.target_adjust_node if args.target_adjust_node >= 0 else 0)
    plot_path = Path(args.plot_path) if args.plot_path else Path(args.output_path).with_name(Path(args.output_path).stem + "_plot.png")
    target_ch = dataset_meta.get("target_channel", 0)
    lag_target = None
    if 0 <= target_ch < past_window.shape[-1]:
        lag_target = past_window[:, :, target_ch].permute(1, 0).contiguous()
    lag_target_plot = inverse_target_scale(lag_target, bundle.scaler) if lag_target is not None else None
    if adjusted_target.shape[0] > 0 and adjusted_target.shape[1] > 0:
        save_prediction_plot(
            plot_path,
            plot_node,
            baseline_plot,
            best_cf_prediction_plot,
            adjusted_target_plot,
            default_target_plot,
            guidance_target_plot,
            lag_target=lag_target_plot,
            plot_lag_steps=args.plot_lag_steps,
        )

    top_window_plot_path = Path(args.plot_top_cf_window_path) if args.plot_top_cf_window_path else None
    if top_window_plot_path is not None:
        past_plot = past_window.detach().cpu().float()
        best_cf_window_plot = best_cf_window
        if args.plot_top_cf_window_feature == target_ch:
            past_plot = inverse_target_feature(past_plot, target_ch, bundle.scaler)
            best_cf_window_plot = inverse_target_feature(best_cf_window_plot, target_ch, bundle.scaler)
        save_top_cf_window_plot(
            top_window_plot_path,
            past_plot,
            best_cf_window_plot,
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
        "node_loss_strategy": args.node_loss_strategy,
        "subgraph_cache_dir": str(subgraph_cache_dir),
        "subgraph_params": {
            "num_walks": args.subgraph_num_walks,
            "walk_length": args.subgraph_walk_length,
            "restart_prob": args.subgraph_restart_prob,
            "top_k": args.subgraph_top_k,
            "spillover_percent": args.subgraph_spillover_percent,
            "seed": args.subgraph_seed,
        },
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
