from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class RandomWalkConfig:
    num_walks: int = 20000
    walk_length: int = 40
    restart_prob: float = 0.15
    top_k: int = 3
    seed: int | None = None


def _normalize_transition(adjacency: np.ndarray) -> np.ndarray:
    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return adjacency / row_sums


def random_walk_subgraph(
    adjacency: np.ndarray,
    start: int,
    config: RandomWalkConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (nodes, weights) reached from start via weighted random walks."""

    num_nodes = adjacency.shape[0]
    probs = _normalize_transition(adjacency)
    visits = np.zeros(num_nodes, dtype=np.float64)
    rng = np.random.default_rng(config.seed)

    for _ in range(config.num_walks):
        node = start
        for _ in range(config.walk_length):
            if rng.random() < config.restart_prob:
                node = start
            neighbors = np.nonzero(probs[node] > 0)[0]
            if neighbors.size == 0:
                break
            p = probs[node, neighbors]
            p = p / p.sum()
            node = rng.choice(neighbors, p=p)
            visits[node] += 1.0

    visits[start] = 0.0  # avoid double-counting the anchor
    if config.top_k is not None and config.top_k > 0:
        order = np.argsort(visits)[::-1][: config.top_k]
    else:
        order = np.argsort(visits)[::-1]
    weights = visits[order].astype(np.float32)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    return order.astype(np.int64), weights


def build_random_walk_subgraphs(adjacency: np.ndarray, config: RandomWalkConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [random_walk_subgraph(adjacency, node, config) for node in range(adjacency.shape[0])]


def _structured_array(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    arr = np.zeros(nodes.shape[0], dtype=[("node", np.int64), ("weight", np.float32)])
    arr["node"] = nodes.astype(np.int64)
    arr["weight"] = weights.astype(np.float32)
    return arr


def save_subgraphs(
    subgraphs: Sequence[Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    config: RandomWalkConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (nodes, weights) in enumerate(subgraphs):
        arr = _structured_array(nodes, weights)
        np.save(output_dir / f"node_{idx}.npy", arr)
    meta = {
        "num_nodes": len(subgraphs),
        "num_walks": config.num_walks,
        "walk_length": config.walk_length,
        "restart_prob": config.restart_prob,
        "top_k": config.top_k,
        "seed": config.seed,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_subgraph(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    nodes = arr["node"].astype(np.int64)
    weights = arr["weight"].astype(np.float32)
    return nodes, weights


def ensure_subgraph_cache(
    adjacency: np.ndarray,
    output_dir: Path,
    config: RandomWalkConfig,
) -> None:
    expected = adjacency.shape[0]
    files_exist = all((output_dir / f"node_{i}.npy").exists() for i in range(expected))
    if files_exist:
        return
    subgraphs = build_random_walk_subgraphs(adjacency, config)
    save_subgraphs(subgraphs, output_dir, config)


def _normalize_weights_or_uniform(weights: np.ndarray, size: int) -> np.ndarray:
    if weights.size == 0:
        return np.ones(size, dtype=np.float32) / max(size, 1)
    total = weights.sum()
    if total <= 0:
        return np.ones_like(weights, dtype=np.float32) / max(len(weights), 1)
    return (weights / total).astype(np.float32)


def build_node_weight_vector(
    num_nodes: int,
    target_node: int,
    target_share: float,
    subgraph_nodes: Iterable[int],
    subgraph_weights: Iterable[float],
    spillover_fraction: float = 0.0,
) -> torch.Tensor:
    """Distribute loss across the target node, its subgraph, and optional spillover."""

    target_share = float(max(0.0, min(target_share, 1.0)))
    remaining = max(0.0, 1.0 - target_share)
    spillover = max(0.0, min(spillover_fraction, 1.0)) * remaining
    focus_share = max(0.0, remaining - spillover)

    weights = np.zeros(num_nodes, dtype=np.float32)
    sub_nodes = np.fromiter(subgraph_nodes, dtype=np.int64)
    sub_weights = np.fromiter(subgraph_weights, dtype=np.float32)
    if sub_nodes.size > 0 and focus_share > 0:
        w = _normalize_weights_or_uniform(sub_weights, sub_nodes.size)
        weights[sub_nodes] += focus_share * w

    if spillover > 0:
        mask = np.ones(num_nodes, dtype=bool)
        mask[target_node] = False
        if sub_nodes.size > 0:
            mask[sub_nodes] = False
        count = int(mask.sum())
        if count > 0:
            weights[mask] += spillover / float(count)

    weights[target_node] = target_share
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones(num_nodes, dtype=np.float32) / max(num_nodes, 1)
    else:
        weights = weights / total
    return torch.from_numpy(weights.astype(np.float32))
