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
import torch.nn.functional as F
from torch import nn

from ..lp_layers import LPBatchNorm1d, LPConv1D, LPLinear, LPMaxLayer, LPReLU, ProbConv1DInput
from ..lp_layers.utils import square


class LPPointNetfeat(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        coalition_size: int = 3,
    ) -> None:
        super().__init__()

        self.conv1 = ProbConv1DInput(3, 64, 1, coalition_size=coalition_size)
        self.conv2 = LPConv1D(64, 128, 1)
        self.conv3 = LPConv1D(128, num_outputs, 1)
        self.bn1 = LPBatchNorm1d(64)
        self.bn2 = LPBatchNorm1d(128)
        self.bn3 = LPBatchNorm1d(num_outputs)
        self.max_pool = LPMaxLayer()
        self.relu = LPReLU()

    def forward(self, inputs):
        x_wo_i, x_w_i = self.conv1(inputs)

        outputs = []
        for x in (x_wo_i, x_w_i,):
            x = self.relu(self.bn1(x))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x = self.max_pool(x)
#             x = x.view(-1, self.num_outputs)
            outputs.append(x)
        return tuple(outputs)


class LPConcatLinear(nn.Linear):
    def __init__(
        self, in_pc: int, in_tabular: int, out_features: int, bias: bool = True,
    ) -> None:
        super().__init__(
            in_features=in_pc + in_tabular,
            out_features=out_features,
            bias=bias,
        )
        self.in_pc = in_pc
        self.in_tabular = in_tabular

    @property
    def weight_pc(self):
        weight_pc = self.weight[:, :self.in_pc]
        return weight_pc

    @property
    def weight_tabular(self):
        weight_tabular = self.weight[:, self.in_pc:]
        return weight_tabular

    def forward(self, inputs_pc, inputs_tabular):
        pc_mean, pc_var = inputs_pc

        m_pc = F.linear(pc_mean, self.weight_pc, self.bias)
        v_pc = F.linear(pc_var, square(self.weight_pc))

        m_tabular = F.linear(inputs_tabular, self.weight_tabular, self.bias)

        m = m_pc + m_tabular
        v = v_pc

        return m, v


class LPWidePointNetClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        num_global: int,
        num_deep: int,
        coalition_size: int = 3,
    ) -> None:
        super().__init__()

        self.feat = LPPointNetfeat(
            num_outputs=num_global,
            coalition_size=coalition_size,
        )

        units = [num_global]
        for _ in range(2):
            units.append(units[-1] // 2)

        self.fc1 = LPLinear(units[0], units[1])
        self.fc2 = LPLinear(units[1], units[2])
        self.fc3 = LPLinear(units[2], num_deep)
        self.fc4 = LPConcatLinear(num_deep, num_features, num_outputs)
        self.bn1 = LPBatchNorm1d(units[1])
        self.bn2 = LPBatchNorm1d(units[2])
        self.bn3 = LPBatchNorm1d(num_deep)
        self.relu = LPReLU()

    def forward(self, pointcloud, feats):
        mvs = self.feat(pointcloud)
        outputs = []
        for x in mvs:
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.relu(self.bn3(self.fc3(x)))
            x = self.fc4(x, feats)
            outputs.append(x)

        return tuple(outputs)
