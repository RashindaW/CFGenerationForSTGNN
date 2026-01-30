from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

import main
import train
from models.causal_forecaster import load_feature_names

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("plotly is required. Install it with `pip install plotly`.") from exc


def ensure_main_unpickle_hooks() -> None:
    import __main__ as main_module

    for name in (
        "run_forecaster_command",
        "run_diffusion_command",
        "run_subgraph_command",
        "run_counterfactual_command",
    ):
        if not hasattr(main_module, name) and hasattr(main, name):
            setattr(main_module, name, getattr(main, name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot test-set predictions, ground truth, and residuals for a forecaster checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the forecaster checkpoint (.pt).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Optional dataset root override (defaults to checkpoint setting).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for Plotly HTML files.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--include_control",
        action="store_true",
        help="Include control nodes in plots (default excludes them when available).",
    )
    return parser.parse_args()


def _to_index_list(value: Optional[torch.Tensor | Sequence[int]]) -> List[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.detach().cpu().flatten().tolist()]
    return [int(v) for v in value]


def resolve_control_indices(model: torch.nn.Module, num_nodes: int) -> List[int]:
    model_ref = train.unwrap_model(model)
    control_indices = _to_index_list(getattr(model_ref, "control_indices", None))
    if control_indices:
        return control_indices
    loss_indices = _to_index_list(getattr(model_ref, "loss_node_indices", None))
    if loss_indices:
        loss_set = set(loss_indices)
        return [idx for idx in range(num_nodes) if idx not in loss_set]
    return []


def build_node_labels(dataset_dir: Path, num_nodes: int) -> List[str]:
    feature_names = load_feature_names(dataset_dir)
    labels = []
    for idx in range(num_nodes):
        if feature_names and idx < len(feature_names):
            labels.append(feature_names[idx])
        else:
            labels.append(f"Node {idx}")
    return labels


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, target = train.prepare_batch(batch, device)
            prediction = train.forward_pass(
                model,
                x,
                model_type,
            )
            preds.append(prediction.cpu())
            targets.append(target.cpu())
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def flatten_horizon(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError(f"Expected (samples, nodes, horizon); got {pred.shape} and {target.shape}.")
    if pred.size(-1) > 1:
        pred = pred[..., -1]
        target = target[..., -1]
    else:
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
    return pred, target


def build_node_dropdown_figure(
    steps: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    residual: np.ndarray,
    node_labels: Sequence[str],
    title: str,
) -> "go.Figure":
    num_nodes = prediction.shape[1]
    traces = []
    for node_idx in range(num_nodes):
        visible = node_idx == 0
        traces.append(
            go.Scatter(
                x=steps,
                y=prediction[:, node_idx],
                mode="lines",
                name="Prediction",
                line=dict(color="#1f77b4"),
                visible=visible,
            )
        )
        traces.append(
            go.Scatter(
                x=steps,
                y=ground_truth[:, node_idx],
                mode="lines",
                name="Ground truth",
                line=dict(color="#2ca02c"),
                visible=visible,
            )
        )
        traces.append(
            go.Scatter(
                x=steps,
                y=residual[:, node_idx],
                mode="lines",
                name="Residual",
                line=dict(color="#d62728", dash="dot"),
                visible=visible,
                yaxis="y2",
            )
        )

    buttons = []
    for node_idx in range(num_nodes):
        visibility = [False] * (num_nodes * 3)
        base = node_idx * 3
        visibility[base : base + 3] = [True, True, True]
        label = node_labels[node_idx]
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"{title} - {label}"},
                ],
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title} - {node_labels[0]}",
        xaxis=dict(title="Test index"),
        yaxis=dict(title="Value"),
        yaxis2=dict(title="Residual", overlaying="y", side="right", showgrid=False),
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
            )
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=80, t=80, b=60),
        hovermode="x unified",
    )
    return fig


def main_entry() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = train.resolve_device(args.device, None)

    ensure_main_unpickle_hooks()
    forecaster, bundle, meta = main.load_forecaster_from_checkpoint(
        checkpoint_path, device, Path(args.data_root) if args.data_root else None
    )
    dataset_name = meta.get("dataset", "TEP")
    if meta.get("lag") != 20 or meta.get("horizon") != 1:
        print(
            f"Warning: checkpoint uses lag={meta.get('lag')} horizon={meta.get('horizon')}; "
            "requested lag=20 horizon=1."
        )
    dataset_dir = train.resolve_dataset_dir(dataset_name, Path(args.data_root) if args.data_root else None)

    default_dir = checkpoint_path.parent.name or checkpoint_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else Path("plotly_forecaster_plots") / default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders, _ = train.build_dataloaders(bundle, args.batch_size, num_workers=0)
    control_indices = resolve_control_indices(forecaster, bundle.num_nodes)
    if control_indices and not args.include_control:
        control_set = set(control_indices)
        node_indices = [idx for idx in range(bundle.num_nodes) if idx not in control_set]
        print(f"Excluding {len(control_indices)} control nodes from plots.")
    else:
        node_indices = list(range(bundle.num_nodes))

    labels = build_node_labels(dataset_dir, bundle.num_nodes)
    labels = [labels[idx] for idx in node_indices]

    def save_split_plot(split_name: str, loader: torch.utils.data.DataLoader) -> None:
        prediction, ground_truth = collect_predictions(
            forecaster,
            loader,
            device,
            meta.get("model", "causal_forecaster"),
        )
        prediction, ground_truth = flatten_horizon(prediction, ground_truth)

        pred_np = prediction.numpy()
        gt_np = ground_truth.numpy()
        pred_np = bundle.scaler.inverse_transform(pred_np)
        gt_np = bundle.scaler.inverse_transform(gt_np)
        residual_np = pred_np - gt_np

        pred_np = pred_np[:, node_indices]
        gt_np = gt_np[:, node_indices]
        residual_np = residual_np[:, node_indices]

        steps = np.arange(pred_np.shape[0])
        title = f"{dataset_name} {split_name} forecast (lag={meta.get('lag')}, horizon={meta.get('horizon')})"
        fig = build_node_dropdown_figure(steps, pred_np, gt_np, residual_np, labels, title)

        html_path = output_dir / f"forecast_ground_truth_residual_by_node_{split_name}.html"
        fig.write_html(str(html_path), include_plotlyjs="inline")
        print(f"Saved Plotly forecast comparison to {html_path}")

    save_split_plot("val", loaders["val"])
    save_split_plot("test", loaders["test"])


if __name__ == "__main__":
    main_entry()
