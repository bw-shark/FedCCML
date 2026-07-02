import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from .client import ClientCCML
from .data import load_config, read_client_data


@dataclass
class ServerConfig:
    data_root: str
    dataset: str
    output_root: str
    num_clients: int
    num_classes: int
    feature_dim: int
    batch_size: int
    local_epochs: int
    local_lr: float
    server_lr: float
    global_rounds: int
    eval_gap: int
    alpha: float
    beta: float
    few_shot: int
    device: torch.device
    goal: str
    seed: int
    run_id: int


class AttentionHyperNet(nn.Module):
    def __init__(
        self,
        num_clients: int,
        num_classes: int,
        feat_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        mlp_dim: int = 1024,
    ):
        super().__init__()
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.output_proj = nn.Linear(hidden_dim, num_classes * (feat_dim + 1))

    def forward(self, proto_mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = proto_mat.unsqueeze(0)
        h = self.input_proj(x)
        attn_out, _ = self.attn(h, h, h)
        h = h + attn_out
        h = h + self.mlp(h)
        out = self.output_proj(h).squeeze(0)
        out = out.view(self.num_clients, self.num_classes, self.feat_dim + 1)
        weights = out[:, :, : self.feat_dim]
        biases = out[:, :, self.feat_dim]
        return weights, biases


class FedCCMLServer:
    def __init__(self, cfg: ServerConfig, model: nn.Module):
        self.cfg = cfg
        self.model_template = model
        self.global_base = copy.deepcopy(model.base).to(cfg.device)
        self.proxy_head = copy.deepcopy(model.head).to(cfg.device)
        self.ce_loss = nn.CrossEntropyLoss()

        self.clients = self._create_clients()
        self.selected_clients: List[ClientCCML] = []

        self.uploaded_weights: List[float] = []
        self.uploaded_bases: List[nn.Module] = []
        self.uploaded_protos: List[Tuple[int, torch.Tensor, torch.Tensor]] = []

        self.hyper_net = AttentionHyperNet(
            num_clients=cfg.num_clients,
            num_classes=cfg.num_classes,
            feat_dim=cfg.feature_dim,
        ).to(cfg.device)
        self.hyper_optimizer = optim.SGD(self.hyper_net.parameters(), lr=cfg.server_lr)
        self.client_proto_mat = torch.zeros(
            cfg.num_clients, cfg.feature_dim, dtype=torch.float32
        )
        self.personalized_heads: Dict[int, nn.Module] = {}

        self.rs_test_acc: List[float] = []
        self.rs_test_auc: List[float] = []
        self.rs_train_loss: List[float] = []
        self.round_times: List[float] = []

    def _create_clients(self) -> List[ClientCCML]:
        clients = []
        for client_id in range(self.cfg.num_clients):
            train_samples = len(
                read_client_data(
                    self.cfg.data_root,
                    self.cfg.dataset,
                    client_id,
                    "train",
                    few_shot=self.cfg.few_shot,
                )
            )
            test_samples = len(
                read_client_data(self.cfg.data_root, self.cfg.dataset, client_id, "test")
            )
            clients.append(
                ClientCCML(
                    client_id=client_id,
                    model=self.model_template,
                    data_root=self.cfg.data_root,
                    dataset=self.cfg.dataset,
                    train_samples=train_samples,
                    test_samples=test_samples,
                    num_classes=self.cfg.num_classes,
                    batch_size=self.cfg.batch_size,
                    local_epochs=self.cfg.local_epochs,
                    local_lr=self.cfg.local_lr,
                    alpha=self.cfg.alpha,
                    device=self.cfg.device,
                    few_shot=self.cfg.few_shot,
                )
            )
        return clients

    def send_models(self) -> None:
        for client in self.clients:
            client.set_base(self.global_base)
            if client.id in self.personalized_heads:
                client.set_head(self.personalized_heads[client.id])
            client.fine_tune_head()
            client.set_global_head(self.proxy_head)

    def receive_models(self) -> None:
        total_samples = sum(client.train_samples for client in self.selected_clients)
        self.uploaded_weights = [
            client.train_samples / total_samples for client in self.selected_clients
        ]
        self.uploaded_bases = [client.model.base for client in self.selected_clients]

    def receive_protos(self) -> None:
        uploaded = []
        for client in self.selected_clients:
            for class_id, feature in client.protos.items():
                label = torch.tensor(class_id, dtype=torch.long, device=self.cfg.device)
                uploaded.append((client.id, feature.detach().to(self.cfg.device), label))
        self.uploaded_protos = uploaded

    def aggregate_base(self) -> None:
        if not self.uploaded_bases:
            return
        self.global_base = copy.deepcopy(self.uploaded_bases[0]).to(self.cfg.device)
        for parameter in self.global_base.parameters():
            parameter.data.zero_()
        for weight, base in zip(self.uploaded_weights, self.uploaded_bases):
            for server_param, client_param in zip(
                self.global_base.parameters(), base.parameters()
            ):
                server_param.data += client_param.data.clone() * weight

    def _build_proto_matrix(self) -> torch.Tensor:
        sum_proto = torch.zeros(
            self.cfg.num_clients,
            self.cfg.feature_dim,
            dtype=torch.float32,
            device=self.cfg.device,
        )
        count_proto = torch.zeros(
            self.cfg.num_clients, 1, dtype=torch.float32, device=self.cfg.device
        )

        for client_id, feature, _ in self.uploaded_protos:
            sum_proto[int(client_id)] += feature.to(self.cfg.device)
            count_proto[int(client_id)] += 1.0

        previous = self.client_proto_mat.to(self.cfg.device)
        count_safe = count_proto.clone()
        count_safe[count_safe == 0] = 1.0
        proto_mat = sum_proto / count_safe
        missing = count_proto.squeeze(1) == 0
        if missing.any():
            proto_mat[missing] = previous[missing]
        self.client_proto_mat = proto_mat.detach().cpu()
        return proto_mat

    def train_hyperhead(self) -> None:
        if not self.uploaded_protos:
            return

        self.hyper_net.train()
        proto_mat = self._build_proto_matrix()
        loader = DataLoader(
            self.uploaded_protos,
            batch_size=self.cfg.batch_size,
            drop_last=False,
            shuffle=True,
        )

        for client_ids, features, labels in loader:
            self.hyper_optimizer.zero_grad(set_to_none=True)
            weights_all, biases_all = self.hyper_net(proto_mat)

            ce_sum = torch.tensor(0.0, device=self.cfg.device)
            norm_sum = torch.tensor(0.0, device=self.cfg.device)
            valid = 0

            for client_id, feature, label in zip(client_ids, features, labels):
                cid = int(client_id.item()) if torch.is_tensor(client_id) else int(client_id)
                feature = feature.to(self.cfg.device)
                label = label.to(self.cfg.device).view(1)
                weight = weights_all[cid]
                bias = biases_all[cid]

                logits = F.linear(feature.view(1, -1), weight, bias)
                ce_sum = ce_sum + self.ce_loss(logits, label)
                class_weight = weight[int(label.item())]
                norm_sum = norm_sum + (class_weight.norm(p=2) - feature.norm(p=2)).pow(2)
                valid += 1

            if valid == 0:
                continue

            loss = ce_sum / valid + self.cfg.beta * (norm_sum / valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.hyper_net.parameters(), 10.0)
            self.hyper_optimizer.step()

        with torch.no_grad():
            weights_all, biases_all = self.hyper_net(proto_mat)
            self._set_generated_heads(weights_all.detach(), biases_all.detach())

    def _set_generated_heads(self, weights_all: torch.Tensor, biases_all: torch.Tensor) -> None:
        self.personalized_heads = {}
        for client_id in range(self.cfg.num_clients):
            head = copy.deepcopy(self.proxy_head).to(self.cfg.device)
            head.weight.data.copy_(weights_all[client_id].to(self.cfg.device))
            head.bias.data.copy_(biases_all[client_id].to(self.cfg.device))
            self.personalized_heads[client_id] = head

        self.proxy_head.weight.data.copy_(weights_all.mean(dim=0).to(self.cfg.device))
        self.proxy_head.bias.data.copy_(biases_all.mean(dim=0).to(self.cfg.device))

    def evaluate(self, round_id: int) -> None:
        test_correct = []
        test_total = []
        train_losses = []
        train_total = []
        for client in self.clients:
            correct, total = client.test_metrics()
            loss, train_count = client.train_metrics()
            test_correct.append(correct)
            test_total.append(total)
            train_losses.append(loss)
            train_total.append(train_count)

        total_acc = float(sum(test_correct) / max(1, sum(test_total)))
        total_loss = float(sum(train_losses) / max(1, sum(train_total)))
        client_acc = [
            correct / max(1, total) for correct, total in zip(test_correct, test_total)
        ]
        self.rs_test_acc.append(total_acc)
        self.rs_test_auc.append(0.0)
        self.rs_train_loss.append(total_loss)

        print(f"\n------------- Round {round_id} -------------")
        print(f"Averaged train loss: {total_loss:.4f}")
        print(f"Averaged test accuracy: {total_acc:.4f}")
        print(f"Std test accuracy: {np.std(client_acc):.4f}")

    def train(self) -> None:
        print(f"Total clients: {self.cfg.num_clients}")
        print("Participation: full")
        print("Finished creating FedCCML server and clients.")
        self.evaluate(round_id=0)

        train_rounds = self.cfg.global_rounds

        for round_id in range(1, train_rounds + 1):
            start = time.time()
            self.selected_clients = list(self.clients)
            self.send_models()

            for client in self.selected_clients:
                client.train()
                client.collect_prototypes()

            if round_id % self.cfg.eval_gap == 0:
                self.evaluate(round_id=round_id)

            self.receive_models()
            self.receive_protos()
            self.train_hyperhead()
            self.aggregate_base()

            elapsed = time.time() - start
            self.round_times.append(elapsed)
            print(f"Round {round_id} time cost: {elapsed:.2f}s")

        print("\nBest accuracy.")
        print(max(self.rs_test_acc) if self.rs_test_acc else 0.0)
        if self.round_times:
            print("\nAverage time cost per round.")
            print(sum(self.round_times) / len(self.round_times))
        self.save_results()

    def save_results(self) -> None:
        result_dir = Path(self.cfg.output_root).expanduser().resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self.cfg.dataset}_FedCCML_{self.cfg.goal}_{self.cfg.run_id}.h5"
        path = result_dir / name
        with h5py.File(path, "w") as hf:
            hf.create_dataset("rs_test_acc", data=np.asarray(self.rs_test_acc))
            hf.create_dataset("rs_test_auc", data=np.asarray(self.rs_test_auc))
            hf.create_dataset("rs_train_loss", data=np.asarray(self.rs_train_loss))
        print(f"Saved results to: {path}")


PAPER_ROUNDS = {
    "Cifar10": 100,
    "Cifar100": 300,
    "Flowers102": 300,
    "PACS": 50,
    "OfficeHome": 50,
}


def config_from_dataset(args, device: torch.device, run_id: int) -> ServerConfig:
    dataset_config = load_config(args.data_root, args.dataset)
    num_clients = args.num_clients or int(dataset_config["num_clients"])
    num_classes = args.num_classes or int(dataset_config["num_classes"])
    global_rounds = args.global_rounds
    if global_rounds is None:
        global_rounds = PAPER_ROUNDS.get(args.dataset, 300)
    return ServerConfig(
        data_root=args.data_root,
        dataset=args.dataset,
        output_root=args.output_root,
        num_clients=num_clients,
        num_classes=num_classes,
        feature_dim=args.feature_dim,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        local_lr=args.local_learning_rate,
        server_lr=args.server_learning_rate,
        global_rounds=global_rounds,
        eval_gap=args.eval_gap,
        alpha=args.alpha,
        beta=args.beta,
        few_shot=args.few_shot,
        device=device,
        goal=args.goal,
        seed=args.seed + run_id,
        run_id=run_id,
    )
