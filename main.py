from __future__ import annotations

import argparse
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
    parser.add_argument("--output_path", type=str, default="counterfactual_samples.pt")
    parser.add_argument("--warm_start", action="store_true", help="Warm-start guidance from the observed trajectory.")
    parser.set_defaults(handler=run_counterfactual_command)
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified driver for ST-GNN training and diffusion-based counterfactuals.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_forecaster_subcommand(subparsers)
    add_diffusion_subcommand(subparsers)
    add_counterfactual_subcommand(subparsers)
    return parser.parse_args()


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


def save_diffusion_checkpoint(
    path: Path,
    epoch: int,
    diffusion: DiffusionModel,
    optimizer: torch.optim.Optimizer,
    config: DiffusionConfig,
    dataset_meta: Dict[str, Any],
    model_meta: Dict[str, Any],
    metrics: Dict[str, float],
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
    torch.save(checkpoint, path)


def load_diffusion_checkpoint(path: Path, device: torch.device) -> Tuple[DiffusionModel, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    config = DiffusionConfig(**checkpoint["config"])
    model_meta = checkpoint["model_meta"]
    network = SpatioTemporalUNet(
        in_channels=model_meta["in_channels"],
        base_channels=model_meta["base_channels"],
        channel_multipliers=tuple(model_meta["channel_multipliers"]),
        time_embedding_dim=model_meta["time_embedding_dim"],
        dropout=model_meta["dropout"],
    )
    diffusion = DiffusionModel(network, config).to(device)
    diffusion.load_state_dict(checkpoint["model_state"])
    return diffusion, checkpoint


# ------------------------- Command handlers ------------------------- #
def run_forecaster_command(args: argparse.Namespace) -> None:
    if args.mode == "train":
        train_pipeline(args)
    else:
        test_pipeline(args)


def run_diffusion_command(args: argparse.Namespace) -> None:
    device = forecaster_module.resolve_device(args.device)
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

    network = SpatioTemporalUNet(
        in_channels=bundle.num_features,
        base_channels=args.diffusion_base_channels,
        channel_multipliers=tuple(args.diffusion_channel_mults),
        time_embedding_dim=args.diffusion_time_dim,
        dropout=args.diffusion_dropout,
    )
    diffusion_config = create_diffusion_config(args)
    diffusion = DiffusionModel(network, diffusion_config).to(device)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    trainer = DiffusionTrainer(
        diffusion,
        optimizer,
        device,
        bundle.adjacency.to(device),
        temporal_context=build_temporal_context(args.lag, device),
    )

    checkpoint_path = resolve_diffusion_checkpoint_path(args)
    best_val = float("inf")
    start_epoch = 1
    if args.checkpoint:
        ckpt = torch.load(Path(args.checkpoint), map_location=device)
        diffusion.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("metrics", {}).get("val_loss", best_val)
        print(f"Resumed diffusion training from {args.checkpoint} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = trainer.train_epoch(loaders["train"])
        val_loss = trainer.evaluate_epoch(loaders["val"])
        print(f"[Diffusion] Epoch {epoch:03d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")
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
            save_diffusion_checkpoint(checkpoint_path, epoch, diffusion, optimizer, diffusion_config, dataset_meta, model_meta, metrics)
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
    device = forecaster_module.resolve_device(args.device)
    forecaster_path = Path(args.forecaster_checkpoint)
    diffusion_path = Path(args.diffusion_checkpoint)

    forecaster, bundle, dataset_meta = load_forecaster_from_checkpoint(forecaster_path, device, Path(args.data_root) if args.data_root else None)

    diffusion, diffusion_ckpt = load_diffusion_checkpoint(diffusion_path, device)
    dataset_info = diffusion_ckpt.get("dataset_meta", {})
    if dataset_info.get("dataset") and dataset_info.get("dataset") != dataset_meta["dataset"]:
        print("Warning: Forecaster and diffusion checkpoints were trained on different datasets.")

    split_dataset = select_split_dataset(bundle, args.split)
    if args.sample_index < 0 or args.sample_index >= len(split_dataset):
        raise ValueError(f"sample_index {args.sample_index} is out of range for split {args.split}")
    past_window, target_future = split_dataset[args.sample_index]
    target_future = target_future.permute(1, 0).contiguous()

    target_tensor = prepare_target(target_future, args.target_path)
    mask = prepare_mask(past_window.shape[0], bundle.num_nodes, bundle.num_features, args.mask_path)

    sample_shape = torch.Size(past_window.shape)  # (T, N, F)
    mask = mask.to(device)
    target_batched = target_tensor.unsqueeze(0).repeat(args.samples, 1, 1).to(device)
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
    )
    guidance = ForecastGuidance(
        forecaster=forecaster,
        target=target_batched,
        mask=mask_batched,
        adjacency=bundle.adjacency.to(device),
        config=guidance_config,
        lower_bounds=args.lower_bound,
        upper_bounds=args.upper_bound,
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
        mse = torch.mean((cf_preds - target_batched) ** 2, dim=(1, 2))
        print(f"Counterfactual guidance MSE per sample: {mse.cpu().numpy()}")

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
