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
from dataclasses import dataclass

import numpy as np
from torch import Tensor, nn


@dataclass
class PointNetConfig:
    num_points: int
    num_units_mlp: int
    num_units_fc_1: int
    num_units_fc_2: int
    num_units_deep: int
    num_outputs: int


class Flatten(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        out = np.prod(inputs.size()[1:])
        return inputs.view(-1, out)


class ThePointNet(nn.Sequential):
    def __init__(self, config: PointNetConfig) -> None:
        super().__init__(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, config.num_units_mlp, 1, bias=False),
            nn.BatchNorm1d(config.num_units_mlp),
            nn.MaxPool1d([config.num_points], stride=1),
            Flatten(),
            nn.Linear(config.num_units_mlp, config.num_units_fc_1, bias=False),
            nn.BatchNorm1d(config.num_units_fc_1),
            nn.ReLU(),
            nn.Linear(config.num_units_fc_1, config.num_units_fc_2, bias=False),
            nn.BatchNorm1d(config.num_units_fc_2),
            nn.ReLU(),
            nn.Linear(config.num_units_fc_2, config.num_units_deep, bias=False),
            nn.BatchNorm1d(config.num_units_deep),
            nn.ReLU(),
            nn.Linear(config.num_units_deep, config.num_outputs),
        )
