from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class BaseHeadSplit(nn.Module):
    def __init__(self, base: nn.Module, head: nn.Module):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        return self.head(features)


class FedAvgCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, flat_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc1 = nn.Sequential(nn.Linear(flat_dim, 512), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        return self.fc(out)


def cnn_flat_dim(dataset: str) -> int:
    if dataset.lower() in {"cifar10", "cifar100"}:
        return 1600
    return 10816


def build_model(
    name: str,
    dataset: str,
    num_classes: int,
    device: torch.device,
    pretrained: bool = True,
) -> BaseHeadSplit:
    if name == "CNN":
        model = FedAvgCNN(
            in_channels=3,
            num_classes=num_classes,
            flat_dim=cnn_flat_dim(dataset),
        )
        head = model.fc
        model.fc = nn.Identity()
        return BaseHeadSplit(model, head).to(device)

    if name == "ResNet18":
        weights: Optional[ResNet18_Weights]
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        head = nn.Linear(in_features, num_classes)
        model.fc = nn.Identity()
        return BaseHeadSplit(model, head).to(device)

    raise ValueError(f"Unsupported model: {name}")
