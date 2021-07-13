# This file is part of Scalable, Axiomatic Explanations of Deep
# Alzheimer's Diagnosis from Heterogeneous Data (SVEHNN).
#
# SVEHNN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SVEHNN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SVEHNN. If not, see <https://www.gnu.org/licenses/>.
from typing import Dict, Optional, Tuple

import torch
from torch import nn, optim
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def weight_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        init.kaiming_normal_(m.weight.data, nonlinearity="relu")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        lr: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self._optimizer = optim.Adam(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )

    def get_device(self) -> torch.device:
        device = self.device
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return device

    def train_one_epoch(self) -> Dict[str, float]:
        device = self.get_device()

        model = self.model
        model.train()

        train_loss = 0.0
        correct = 0
        total = 0
        for points, label in self.dataloader:
            points, label = points.to(device), label.to(device)

            prediction = model(points)
            is_multiclass = prediction.shape[1] > 1
            if is_multiclass:
                criterion = nn.CrossEntropyLoss()
                _, pred_class = torch.max(prediction.data, 1)
            else:
                criterion = nn.BCEWithLogitsLoss()
                label = torch.unsqueeze(label, dim=1).type(torch.FloatTensor).to(device)
                prediction_prob = torch.sigmoid(prediction)
                pred_class = (
                    (prediction_prob.data > 0.5).type(torch.FloatTensor).to(device)
                )
            loss = criterion(prediction, label)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            train_loss += loss.detach().cpu().numpy()
            total += label.size(0)
            correct += (pred_class == label).sum().detach().cpu().numpy()

        acc = correct / total
        return {"train_loss": train_loss, "train_accuracy": acc}

    def train(self, num_epochs: int) -> None:
        pbar = tqdm(
            range(num_epochs),
            total=num_epochs,
            desc="Training",
        )
        # do init on CPU for reproducibility
        self.model.apply(weight_init)
        self.model = self.model.to(self.get_device())
        for _ in pbar:
            stats = self.train_one_epoch()
            pbar.set_postfix(stats)
