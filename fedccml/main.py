import argparse
import os
import random

import numpy as np
import torch

from .models import build_model
from .server import FedCCMLServer, config_from_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedCCML experiments.")

    paths = parser.add_argument_group("paths")
    paths.add_argument("--data-root", type=str, default="data")
    paths.add_argument("--output-root", type=str, default="results")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    runtime.add_argument("--device-id", type=str, default="0")
    runtime.add_argument("--times", type=int, default=1)
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--goal", type=str, default="test")

    paper = parser.add_argument_group("paper defaults")
    paper.add_argument("--dataset", type=str, default="Cifar100")
    paper.add_argument("--model", type=str, default="CNN", choices=["CNN", "ResNet18"])
    paper.add_argument("--batch-size", type=int, default=32)
    paper.add_argument("--local-learning-rate", type=float, default=0.005)
    paper.add_argument("--server-learning-rate", type=float, default=0.005)
    paper.add_argument("--local-epochs", type=int, default=5)
    paper.add_argument("--alpha", type=float, default=20.0)
    paper.add_argument("--beta", type=float, default=1.0)

    tuning = parser.add_argument_group("optional tuning")
    tuning.add_argument("--num-classes", type=int, default=None)
    tuning.add_argument("--num-clients", type=int, default=None)
    tuning.add_argument("--global-rounds", type=int, default=None)
    tuning.add_argument("--eval-gap", type=int, default=10)
    tuning.add_argument("--feature-dim", type=int, default=512)
    tuning.add_argument("--few-shot", type=int, default=0)
    tuning.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    for run_id in range(args.times):
        set_seed(args.seed + run_id)
        cfg = config_from_dataset(args, device=device, run_id=run_id)
        model = build_model(
            args.model,
            args.dataset,
            cfg.num_classes,
            device,
            pretrained=not args.no_pretrained,
        )
        print("=" * 50)
        for key, value in sorted(vars(args).items()):
            print(f"{key} = {value}")
        print(f"run_id = {run_id}")
        print(f"resolved_num_clients = {cfg.num_clients}")
        print(f"resolved_num_classes = {cfg.num_classes}")
        print(f"resolved_global_rounds = {cfg.global_rounds}")
        print("=" * 50)

        server = FedCCMLServer(cfg, model)
        server.train()


if __name__ == "__main__":
    main()
