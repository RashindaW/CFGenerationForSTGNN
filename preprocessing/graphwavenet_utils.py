from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import linalg


class GWNSampleLoader:
    def __init__(self, xs: np.ndarray, ys: np.ndarray, batch_size: int, pad_with_last_sample: bool = True) -> None:
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            if num_padding:
                x_padding = np.repeat(xs[-1:], num_padding, axis=0)
                y_padding = np.repeat(ys[-1:], num_padding, axis=0)
                xs = np.concatenate([xs, x_padding], axis=0)
                ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self) -> None:
        permutation = np.random.permutation(self.size)
        self.xs = self.xs[permutation]
        self.ys = self.ys[permutation]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std if std != 0 else 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


def _sym_adj(adj: np.ndarray) -> np.ndarray:
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def _asym_adj(adj: np.ndarray) -> np.ndarray:
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def _calculate_normalized_laplacian(adj: np.ndarray) -> sp.coo_matrix:
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def _calculate_scaled_laplacian(adj_mx: np.ndarray, lambda_max: float = 2, undirected: bool = True) -> np.ndarray:
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = _calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    identity = sp.identity(M, format="csr", dtype=L.dtype)
    L = (2 / lambda_max * L) - identity
    return L.astype(np.float32).todense()


def _load_pickle(pickle_file: Path) -> Tuple[Any, Any, np.ndarray]:
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    return pickle_data


def load_graph_wavenet_adj(pkl_filename: Path, adjtype: str):
    sensor_ids, sensor_id_to_ind, adj_mx = _load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [_calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [_calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [_sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [_asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [_asym_adj(adj_mx), _asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        raise ValueError(f"Unsupported adjacency type {adjtype}")
    return sensor_ids, sensor_id_to_ind, adj


def load_graph_wavenet_dataset(
    dataset_dir: Path,
    batch_size: int,
    valid_batch_size: int | None = None,
    test_batch_size: int | None = None,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for category in ["train", "val", "test"]:
        cat_data = np.load(dataset_dir / f"{category}.npz")
        data[f"x_{category}"] = cat_data["x"]
        data[f"y_{category}"] = cat_data["y"]
    scaler = StandardScaler(mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std())
    for category in ["train", "val", "test"]:
        data[f"x_{category}"][..., 0] = scaler.transform(data[f"x_{category}"][..., 0])
    data["train_loader"] = GWNSampleLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = GWNSampleLoader(data["x_val"], data["y_val"], valid_batch_size or batch_size)
    data["test_loader"] = GWNSampleLoader(data["x_test"], data["y_test"], test_batch_size or batch_size)
    data["scaler"] = scaler
    return data


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred: torch.Tensor, real: torch.Tensor) -> Tuple[float, float, float]:
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
