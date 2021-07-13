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
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class PointNetfeat(nn.Module):
    def __init__(self, global_feat: bool = True, num_outputs: int = 1024) -> None:
        super(PointNetfeat, self).__init__()
        self.num_outputs = num_outputs
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, num_outputs, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(num_outputs)
        self.global_feat = global_feat

    def forward(self, x):
        n_pts = x.size()[2]
        trans = None
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.num_outputs)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, self.num_outputs, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class WidePointNetClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        num_global: int = 1024,
        num_deep: int = 64,
    ) -> None:
        super().__init__()
        assert num_features > 0, "{} <= 0".format(num_features)
        self.feat = PointNetfeat(
            global_feat=True,
            num_outputs=num_global,
        )
        units = [num_global]
        for _ in range(2):
            units.append(units[-1] // 2)
        self.fc1 = nn.Linear(units[0], units[1])
        self.fc2 = nn.Linear(units[1], units[2])
        self.fc3 = nn.Linear(units[2], num_deep)
        self.fc4 = nn.Linear(num_deep + num_features, num_outputs)
        self.bn1 = nn.BatchNorm1d(units[1])
        self.bn2 = nn.BatchNorm1d(units[2])
        self.bn3 = nn.BatchNorm1d(num_deep)

    def forward(self, x, feats):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = torch.cat((x, feats), dim=1)
        x = self.fc4(x)

        return x, trans, trans_feat


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model = model.to(device).eval()

    predictions = {"logits": [], "class_idx": [], "prob": []}
    for batch in loader:
        pc = batch[0].to(device)
        pc.transpose_(2, 1)
        feat = batch[1].to(device)

        logits = model(pc, feat)[0]
        assert logits.dim() == 2 and logits.size()[1] == 1
        prob = torch.sigmoid(logits)
        probs = torch.cat((1 - prob, prob), dim=1)
        class_idx = torch.argmax(probs, dim=1)

        for k, v in zip(("logits", "class_idx", "prob"), (logits, class_idx, prob)):
            predictions[k].append(v.detach().cpu().numpy())

    for k, v in predictions.items():
        if v[0].ndim == 1:
            v = np.concatenate(v)
        else:
            v = np.row_stack(v)
        predictions[k] = v
    return predictions
