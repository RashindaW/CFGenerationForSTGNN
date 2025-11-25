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
        "PEMSBAY": {"values": "pems_node_values.npy", "adjacency": "pems_adj_mat.npy"},
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
        values, adjacency = self._load_arrays()
        num_nodes = values.shape[1]
        num_features = values.shape[2]

        scaler = StandardScaler(mean=values[..., self.target_channel].mean(), std=values[..., self.target_channel].std())
        values[..., self.target_channel] = scaler.transform(values[..., self.target_channel])

        train_data, val_data, test_data = self._split(values)

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

    def _load_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        files = self.DATA_FILES[self.dataset]
        data_dir = self.data_root / self.dataset
        values_path = data_dir / files["values"]
        adjacency_path = data_dir / files["adjacency"]
        if not values_path.exists() or not adjacency_path.exists():
            raise FileNotFoundError(f"Missing dataset files in {data_dir}")
        values = np.load(values_path)
        adjacency = np.load(adjacency_path)
        return values, adjacency

    def _split(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = values.shape[0]
        train_end = int(total * self.train_ratio)
        val_end = min(train_end + int(total * self.val_ratio), total)

        train_data = values[:train_end]

        val_start = max(train_end - self.lag, 0)
        val_stop = min(val_end + self.horizon, total)
        val_data = values[val_start:val_stop]

        test_start = max(val_end - self.lag, 0)
        test_data = values[test_start:]

        return train_data, val_data, test_data


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
