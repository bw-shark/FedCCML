import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import (
    client_statistics,
    dirichlet_partition,
    normalize_labels,
    save_partitions,
    split_train_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for FedCCML.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Cifar10", "Cifar100", "Flowers102", "PACS", "OfficeHome"],
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Optional raw-data root for PACS or OfficeHome.",
    )
    return parser.parse_args()


def default_seed(dataset: str) -> int:
    if dataset == "Cifar10":
        return 18
    return 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_dataset_to_numpy(dataset, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for x, y in loader:
        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def prepare_label_skew(args: argparse.Namespace) -> None:
    num_clients = args.num_clients or 20
    root = Path(args.data_root) / args.dataset / "rawdata"
    root.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"Cifar10", "Cifar100"}:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset_cls = torchvision.datasets.CIFAR10
        if args.dataset == "Cifar100":
            dataset_cls = torchvision.datasets.CIFAR100
        train_set = dataset_cls(root=str(root), train=True, download=True, transform=transform)
        test_set = dataset_cls(root=str(root), train=False, download=True, transform=transform)
        x_train, y_train = tensor_dataset_to_numpy(train_set, batch_size=len(train_set))
        x_test, y_test = tensor_dataset_to_numpy(test_set, batch_size=len(test_set))
        x_all = np.concatenate([x_train, x_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        splits = ["train", "val", "test"]
        xs = []
        ys = []
        for split in splits:
            ds = torchvision.datasets.Flowers102(
                root=str(root), split=split, download=True, transform=transform
            )
            x_part, y_part = tensor_dataset_to_numpy(ds, batch_size=len(ds))
            xs.append(x_part)
            ys.append(y_part)
        x_all = np.concatenate(xs, axis=0)
        y_all = np.concatenate(ys, axis=0)

    y_all, _ = normalize_labels(y_all)
    num_classes = int(len(np.unique(y_all)))
    partitions = dirichlet_partition(
        y_all,
        num_clients=num_clients,
        num_classes=num_classes,
        alpha=args.alpha,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    x_parts = [x_all[idx] for idx in partitions]
    y_parts = [y_all[idx] for idx in partitions]
    train_data, test_data = split_train_test(
        x_parts, y_parts, train_ratio=args.train_ratio, seed=args.seed
    )
    save_partitions(
        args.data_root,
        args.dataset,
        train_data,
        test_data,
        num_classes,
        client_statistics(y_parts),
        {
            "partition": "dirichlet",
            "alpha": args.alpha,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
        },
    )


def image_files(root: Path) -> List[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return [path for path in root.rglob("*") if path.suffix.lower() in suffixes]


def find_domain_root(raw_root: Path, domain_aliases: Dict[str, Tuple[str, ...]]) -> Path:
    candidates = [raw_root] + [p for p in raw_root.rglob("*") if p.is_dir()]
    for candidate in candidates:
        names = {p.name.lower().replace("-", "_").replace(" ", "_") for p in candidate.iterdir() if p.is_dir()}
        hits = 0
        for aliases in domain_aliases.values():
            if any(alias in names for alias in aliases):
                hits += 1
        if hits == len(domain_aliases):
            return candidate
    raise FileNotFoundError(
        f"Could not find domain folders under {raw_root}. "
        "Place the extracted dataset there and rerun this command."
    )


def resolve_domains(root: Path, domain_aliases: Dict[str, Tuple[str, ...]]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower().replace("-", "_").replace(" ", "_")
        for canonical, aliases in domain_aliases.items():
            if name in aliases:
                result[canonical] = child
    missing = set(domain_aliases) - set(result)
    if missing:
        raise FileNotFoundError(f"Missing domain folders: {sorted(missing)}")
    return result


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path], labels: np.ndarray, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])


def collect_domain_data(domain_path: Path, class_to_id: Dict[str, int], transform):
    paths: List[Path] = []
    labels: List[int] = []
    for class_dir in sorted(p for p in domain_path.iterdir() if p.is_dir()):
        if class_dir.name not in class_to_id:
            continue
        class_paths = image_files(class_dir)
        paths.extend(class_paths)
        labels.extend([class_to_id[class_dir.name]] * len(class_paths))

    if not paths:
        raise RuntimeError(f"No images found in {domain_path}")

    labels_np = np.asarray(labels, dtype=np.int64)
    dataset = ImagePathDataset(paths, labels_np, transform)
    return tensor_dataset_to_numpy(dataset, batch_size=len(dataset))


def prepare_domain_skew(args: argparse.Namespace) -> None:
    if args.dataset == "PACS":
        order = ["photo", "art", "cartoon", "sketch"]
        aliases = {
            "photo": ("photo",),
            "art": ("art", "art_painting", "art_paintings"),
            "cartoon": ("cartoon", "cartoons"),
            "sketch": ("sketch", "sketches"),
        }
    else:
        order = ["art", "clipart", "product", "real"]
        aliases = {
            "art": ("art",),
            "clipart": ("clipart",),
            "product": ("product",),
            "real": ("real_world", "realworld", "real"),
        }

    raw_root = Path(args.raw_root) if args.raw_root else Path(args.data_root) / args.dataset / "rawdata"
    domain_root = find_domain_root(raw_root, aliases)
    domain_paths = resolve_domains(domain_root, aliases)

    class_names = sorted(
        {
            child.name
            for domain_path in domain_paths.values()
            for child in domain_path.iterdir()
            if child.is_dir()
        }
    )
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    x_parts = []
    y_parts = []
    for domain in order:
        x, y = collect_domain_data(domain_paths[domain], class_to_id, transform)
        x_parts.append(x)
        y_parts.append(y)

    train_data, test_data = split_train_test(
        x_parts, y_parts, train_ratio=args.train_ratio, seed=args.seed
    )
    save_partitions(
        args.data_root,
        args.dataset,
        train_data,
        test_data,
        len(class_names),
        client_statistics(y_parts),
        {
            "partition": "domain",
            "domains": order,
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "raw_root": str(raw_root),
        },
    )


def main() -> None:
    args = parse_args()
    if args.seed is None:
        args.seed = default_seed(args.dataset)
    set_seed(args.seed)
    if args.dataset in {"Cifar10", "Cifar100", "Flowers102"}:
        prepare_label_skew(args)
    else:
        prepare_domain_skew(args)
    print(f"Prepared {args.dataset} under {Path(args.data_root).resolve()}.")


if __name__ == "__main__":
    main()
