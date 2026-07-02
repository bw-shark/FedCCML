import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def dataset_dir(data_root: str, dataset: str) -> Path:
    return Path(data_root).expanduser().resolve() / dataset


def load_config(data_root: str, dataset: str) -> Dict:
    config_path = dataset_dir(data_root, dataset) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing dataset config: {config_path}. "
            "Run `python -m fedccml.prepare_data` first."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def read_client_data(
    data_root: str,
    dataset: str,
    client_id: int,
    split: str,
    few_shot: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    path = dataset_dir(data_root, dataset) / split / f"{client_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing client data file: {path}")

    with path.open("rb") as f:
        data = np.load(f, allow_pickle=True)["data"].tolist()

    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.int64)
    items = [(xi, yi) for xi, yi in zip(x, y)]

    if split == "train" and few_shot > 0:
        counts: Dict[int, int] = {}
        reduced = []
        for xi, yi in items:
            label = int(yi.item())
            if counts.get(label, 0) < few_shot:
                reduced.append((xi, yi))
                counts[label] = counts.get(label, 0) + 1
        items = reduced

    return items


def make_loader(
    data_root: str,
    dataset: str,
    client_id: int,
    split: str,
    batch_size: int,
    few_shot: int = 0,
    drop_last: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    data = read_client_data(data_root, dataset, client_id, split, few_shot=few_shot)
    return DataLoader(data, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)


def split_train_test(
    x_parts: Sequence[np.ndarray],
    y_parts: Sequence[np.ndarray],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    rng = np.random.RandomState(seed)
    train_data: List[Dict[str, np.ndarray]] = []
    test_data: List[Dict[str, np.ndarray]] = []

    for x, y in zip(x_parts, y_parts):
        indices = np.arange(len(y))
        rng.shuffle(indices)
        split = int(len(indices) * train_ratio)
        train_idx = indices[:split]
        test_idx = indices[split:]
        train_data.append({"x": x[train_idx], "y": y[train_idx]})
        test_data.append({"x": x[test_idx], "y": y[test_idx]})

    return train_data, test_data


def save_partitions(
    output_root: str,
    dataset: str,
    train_data: Sequence[Dict[str, np.ndarray]],
    test_data: Sequence[Dict[str, np.ndarray]],
    num_classes: int,
    statistic: Sequence[Sequence[Tuple[int, int]]],
    config_extra: Dict,
) -> None:
    root = dataset_dir(output_root, dataset)
    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(train_data):
        with (train_dir / f"{idx}.npz").open("wb") as f:
            np.savez_compressed(f, data=item)
    for idx, item in enumerate(test_data):
        with (test_dir / f"{idx}.npz").open("wb") as f:
            np.savez_compressed(f, data=item)

    config = {
        "num_clients": len(train_data),
        "num_classes": num_classes,
        "train_samples": [int(len(item["y"])) for item in train_data],
        "test_samples": [int(len(item["y"])) for item in test_data],
        "label_statistics": [[list(pair) for pair in client] for client in statistic],
    }
    config.update(config_extra)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def normalize_labels(labels: Iterable[int]) -> Tuple[np.ndarray, Dict[int, int]]:
    raw = np.asarray(list(labels), dtype=np.int64)
    values = sorted(int(v) for v in np.unique(raw))
    mapping = {value: idx for idx, value in enumerate(values)}
    normalized = np.asarray([mapping[int(v)] for v in raw], dtype=np.int64)
    return normalized, mapping


def client_statistics(y_parts: Sequence[np.ndarray]) -> List[List[Tuple[int, int]]]:
    stats: List[List[Tuple[int, int]]] = []
    for labels in y_parts:
        client_stats = []
        for label in np.unique(labels):
            client_stats.append((int(label), int(np.sum(labels == label))))
        stats.append(client_stats)
    return stats


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    num_classes: int,
    alpha: float,
    train_ratio: float,
    seed: int,
    balance: bool = True,
) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    min_required = int(min(32 / (1 - train_ratio), len(labels) / num_clients / 2))
    min_size = 0
    idx_batch: List[List[int]] = [[] for _ in range(num_clients)]

    while min_size < min_required:
        idx_batch = [[] for _ in range(num_clients)]
        for cls in range(num_classes):
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            if balance:
                proportions = np.asarray(
                    [
                        p * (len(client_indices) < len(labels) / num_clients)
                        for p, client_indices in zip(proportions, idx_batch)
                    ]
                )
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            for client_indices, part in zip(idx_batch, np.split(cls_indices, split_points)):
                client_indices.extend(part.tolist())
        min_size = min(len(client_indices) for client_indices in idx_batch)

    for client_indices in idx_batch:
        rng.shuffle(client_indices)
    return [np.asarray(client_indices, dtype=np.int64) for client_indices in idx_batch]
