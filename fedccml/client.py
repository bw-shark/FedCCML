import copy
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import make_loader


class ClientCCML:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        data_root: str,
        dataset: str,
        train_samples: int,
        test_samples: int,
        num_classes: int,
        batch_size: int,
        local_epochs: int,
        local_lr: float,
        alpha: float,
        device: torch.device,
        few_shot: int = 0,
        train_slow: bool = False,
    ):
        self.id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.data_root = data_root
        self.dataset = dataset
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.alpha = alpha
        self.device = device
        self.few_shot = few_shot
        self.train_slow = train_slow

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_lr)
        self.global_head = copy.deepcopy(self.model.head).to(device)
        for parameter in self.global_head.parameters():
            parameter.requires_grad = False

        self.protos: Dict[int, torch.Tensor] = {}

    def train_loader(self):
        return make_loader(
            self.data_root,
            self.dataset,
            self.id,
            "train",
            self.batch_size,
            few_shot=self.few_shot,
            drop_last=True,
            shuffle=True,
        )

    def test_loader(self):
        return make_loader(
            self.data_root,
            self.dataset,
            self.id,
            "test",
            self.batch_size,
            few_shot=0,
            drop_last=False,
            shuffle=False,
        )

    def set_base(self, base_model: nn.Module) -> None:
        for source, target in zip(base_model.parameters(), self.model.base.parameters()):
            target.data.copy_(source.data)

    def set_head(self, head: nn.Module) -> None:
        for source, target in zip(head.parameters(), self.model.head.parameters()):
            target.data.copy_(source.data)

    def set_global_head(self, head: nn.Module) -> None:
        for source, target in zip(head.parameters(), self.global_head.parameters()):
            target.data.copy_(source.data)
            target.requires_grad = False

    def fine_tune_head(self) -> None:
        for parameter in self.model.base.parameters():
            parameter.requires_grad = False
        for parameter in self.model.head.parameters():
            parameter.requires_grad = True

        self.model.train()
        for _ in range(self.local_epochs):
            for x, y in self.train_loader():
                x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * float(np.random.rand()))
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

    def train(self) -> None:
        self.model.train()
        self.global_head.eval()

        for parameter in self.model.base.parameters():
            parameter.requires_grad = True
        for parameter in self.model.head.parameters():
            parameter.requires_grad = False

        for _ in range(self.local_epochs):
            for x, y in self.train_loader():
                x = x.to(self.device)
                y = y.to(self.device)

                features = self.model.base(x)
                logits = self.model.head(features)
                ce_loss = self.loss_fn(logits, y)

                features_norm = F.normalize(features, p=2, dim=1)
                local_weight = F.normalize(self.model.head.weight.detach(), p=2, dim=1)
                global_weight = F.normalize(self.global_head.weight.detach(), p=2, dim=1)

                local_pos = local_weight.index_select(0, y)
                global_pos = global_weight.index_select(0, y)
                local_align = (1.0 - (features_norm * local_pos).sum(dim=1)).mean()
                global_align = (1.0 - (features_norm * global_pos).sum(dim=1)).mean()

                loss = ce_loss + self.alpha * (local_align + global_align)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()

                if self.train_slow:
                    time.sleep(0.1 * float(np.random.rand()))

    def collect_prototypes(self) -> Dict[int, torch.Tensor]:
        self.model.eval()
        protos = defaultdict(list)
        with torch.no_grad():
            for x, y in self.train_loader():
                x = x.to(self.device)
                y = y.to(self.device)
                features = self.model.base(x)
                for row, label in zip(features, y):
                    protos[int(label.item())].append(row.detach().cpu())

        self.protos = {
            label: torch.stack(items, dim=0).mean(dim=0)
            for label, items in protos.items()
            if items
        }
        self.model.train()
        return self.protos

    def test_metrics(self) -> Tuple[int, int]:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.test_loader():
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                correct += int((torch.argmax(logits, dim=1) == y).sum().item())
                total += int(y.shape[0])
        return correct, total

    def train_metrics(self) -> Tuple[float, int]:
        self.model.eval()
        total_loss = 0.0
        total = 0
        with torch.no_grad():
            for x, y in self.train_loader():
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                total_loss += float(loss.item()) * int(y.shape[0])
                total += int(y.shape[0])
        return total_loss, total
