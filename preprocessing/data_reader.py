from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .graphwavenet_utils import StandardScaler


@dataclass
class TemporalDatasetBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    adjacency: torch.Tensor
    scaler: StandardScaler
    num_nodes: int
    num_features: int


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, lag: int, horizon: int, target_channel: int = 0) -> None:
        if data.shape[0] < lag + horizon:
            raise ValueError("Not enough data points to create sequences with the given lag and horizon.")
        self.data = data.astype(np.float32)
        self.lag = lag
        self.horizon = horizon
        self.target_channel = target_channel

    def __len__(self) -> int:
        return self.data.shape[0] - self.lag - self.horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.lag]  # (lag, nodes, features)
        y = self.data[idx + self.lag : idx + self.lag + self.horizon, :, self.target_channel]  # (horizon, nodes)
        x_tensor = torch.from_numpy(np.copy(x))
        y_tensor = torch.from_numpy(np.copy(y))
        return x_tensor, y_tensor


class DataReader:
    DATA_FILES = {
        "METRLA": {"values": "node_values.npy", "adjacency": "adj_mat.npy"},
        "METRLA_15": {"values": "node_values.npy", "adjacency": "adj_mat.npy"},
        "METRLA_30": {"values": "node_values.npy", "adjacency": "adj_mat.npy"},
        "METRLA_SUB": {"values": "node_values.npy", "adjacency": "adj_mat.npy"},
        "METRLA_SUB_15": {"values": "node_values.npy", "adjacency": "adj_mat.npy"},
        "METRLA_SUB_30": {"values": "node_values.npy", "adjacency": "adj_mat.npy"},
        "PEMSBAY": {"values": "pems_node_values.npy", "adjacency": "pems_adj_mat.npy"},
        "TEP": {
            "values": "causal/node_values_train.npy",
            "test_values": "causal/node_values_test.npy",
            "adjacency": "causal/adj_mat_causal.npy",
        },
        "TEP_SMOOTH10": {
            "values": "node_values_train.npy",
            "test_values": "node_values_test.npy",
            "adjacency": "adj_mat_causal.npy",
        },
        "TEP_SMOOTH20": {
            "values": "node_values_train.npy",
            "test_values": "node_values_test.npy",
            "adjacency": "adj_mat_causal.npy",
        },
        "TEP_SMOOTH60": {
            "values": "node_values_train.npy",
            "test_values": "node_values_test.npy",
            "adjacency": "adj_mat_causal.npy",
        },
    }

    def __init__(
        self,
        dataset: str,
        lag: int = 12,
        horizon: int = 12,
        data_root: Optional[Path] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        target_channel: int = 0,
    ) -> None:
        self.dataset = dataset.upper()
        if self.dataset not in self.DATA_FILES:
            raise ValueError(f"Dataset {self.dataset} not supported.")
        self.lag = lag
        self.horizon = horizon
        self.data_root = Path(data_root) if data_root is not None else Path(__file__).resolve().parent / "data"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.target_channel = target_channel

    def read_data(self) -> TemporalDatasetBundle:
        train_values, test_values, adjacency = self._load_arrays()
        num_nodes = train_values.shape[1]
        num_features = train_values.shape[2]

        # Compute scaler from training data only (avoid data leakage)
        if test_values is not None:
            # TEP datasets: use all of train_values for scaler fitting
            scaler_data = train_values
        else:
            # Other datasets: use first train_ratio portion
            train_end = int(len(train_values) * self.train_ratio)
            scaler_data = train_values[:train_end]

        scaler = StandardScaler(
            mean=scaler_data[..., self.target_channel].mean(),
            std=scaler_data[..., self.target_channel].std()
        )

        # Apply scaler to train data
        train_values[..., self.target_channel] = scaler.transform(
            train_values[..., self.target_channel]
        )

        if test_values is not None:
            # TEP datasets: separate train/val from train file, test from test file
            test_values[..., self.target_channel] = scaler.transform(
                test_values[..., self.target_channel]
            )
            train_data, val_data = self._split_train_val(train_values)
            test_data = test_values
        else:
            # Other datasets: ratio-based split from single file
            train_data, val_data, test_data = self._split(train_values)

        train_dataset = SequenceDataset(train_data, self.lag, self.horizon, self.target_channel)
        val_dataset = SequenceDataset(val_data, self.lag, self.horizon, self.target_channel)
        test_dataset = SequenceDataset(test_data, self.lag, self.horizon, self.target_channel)

        adjacency_tensor = torch.from_numpy(adjacency).float()

        return TemporalDatasetBundle(
            train=train_dataset,
            val=val_dataset,
            test=test_dataset,
            adjacency=adjacency_tensor,
            scaler=scaler,
            num_nodes=num_nodes,
            num_features=num_features,
        )

    def _load_arrays(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Load data arrays. Returns (train_values, test_values, adjacency).

        test_values is None for datasets without separate test files.
        """
        files = self.DATA_FILES[self.dataset]
        data_dir = self.data_root / self.dataset
        values_path = data_dir / files["values"]
        adjacency_path = data_dir / files["adjacency"]

        if not values_path.exists() or not adjacency_path.exists():
            raise FileNotFoundError(f"Missing dataset files in {data_dir}")

        train_values = np.load(values_path)
        adjacency = np.load(adjacency_path)

        # Load separate test file if available (TEP datasets)
        test_values = None
        if "test_values" in files:
            test_path = data_dir / files["test_values"]
            if test_path.exists():
                test_values = np.load(test_path)

        return train_values, test_values, adjacency

    def _split(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/val/test sets without overlap to prevent data leakage.

        Each split is contiguous and non-overlapping. The SequenceDataset class
        handles windowing internally, so we don't need to include extra lag/horizon
        padding between splits.
        """
        total = values.shape[0]
        train_end = int(total * self.train_ratio)
        val_end = min(train_end + int(total * self.val_ratio), total)

        # Non-overlapping splits to prevent data leakage
        train_data = values[:train_end]
        val_data = values[train_end:val_end]
        test_data = values[val_end:]

        return train_data, val_data, test_data

    def _split_train_val(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split values into train and validation sets (for datasets with separate test files).

        Uses 80/20 split: 80% train, 20% val from the train file.
        Non-overlapping to prevent data leakage.
        """
        total = values.shape[0]
        train_end = int(total * 0.8)  # Fixed 80% train, 20% val

        # Non-overlapping splits to prevent data leakage
        train_data = values[:train_end]
        val_data = values[train_end:]

        return train_data, val_data


def compute_node_statistics(
    dataset: str,
    data_root: Optional[Path] = None,
    train_ratio: float = 0.7,
    target_channel: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-node mean and std from training data (original scale).

    Args:
        dataset: Dataset name (e.g., "TEP", "TEP_SMOOTH60").
        data_root: Root directory containing dataset folders.
        train_ratio: Fraction of data used for training.
        target_channel: Feature channel to compute statistics for.

    Returns:
        means: Per-node mean values of shape (num_nodes,).
        stds: Per-node standard deviation values of shape (num_nodes,).
    """
    dataset_upper = dataset.upper()
    if dataset_upper not in DataReader.DATA_FILES:
        raise ValueError(f"Dataset {dataset} not supported.")

    base_path = Path(data_root) if data_root is not None else Path(__file__).resolve().parent / "data"
    files = DataReader.DATA_FILES[dataset_upper]
    data_dir = base_path / dataset_upper
    values_path = data_dir / files["values"]

    if not values_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {values_path}")

    values = np.load(values_path)  # (time, nodes, features)

    # Use only training portion
    train_end = int(len(values) * train_ratio)
    train_values = values[:train_end, :, target_channel]  # (train_time, nodes)

    # Compute per-node statistics
    means = train_values.mean(axis=0)  # (num_nodes,)
    stds = train_values.std(axis=0)  # (num_nodes,)

    return means, stds


def load_dataset(
    dataset: str,
    lag: int = 12,
    horizon: int = 12,
    data_root: Optional[Path] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    target_channel: int = 0,
) -> TemporalDatasetBundle:
    reader = DataReader(
        dataset=dataset,
        lag=lag,
        horizon=horizon,
        data_root=data_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        target_channel=target_channel,
    )
    return reader.read_data()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and summarize a temporal dataset.")
    parser.add_argument("--dataset", type=str, default="METRLA", help="Dataset name (METRLA or PEMSBAY)")
    parser.add_argument("--lag", type=int, default=12, help="Number of historical steps")
    parser.add_argument("--horizon", type=int, default=12, help="Prediction horizon")

    args = parser.parse_args()
    bundle = load_dataset(dataset=args.dataset, lag=args.lag, horizon=args.horizon)
    print(f"Loaded dataset {args.dataset}")
    print(f"Train samples: {len(bundle.train)}, Val samples: {len(bundle.val)}, Test samples: {len(bundle.test)}")
