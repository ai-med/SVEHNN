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
from torch import nn

from ..lp_layers import LPBatchNorm1d, LPConv1D, LPLinear, LPMaxLayer, LPReLU, ProbConv1DInput
from ..lp_layers.base import InputLayerInputs, InputLayerOutputs
from .model import PointNetConfig


class TheLPPointnet(nn.Module):
    def __init__(self, config: PointNetConfig, coalition_size: int = 3) -> None:
        super().__init__()
        layers = (
            ProbConv1DInput(
                3, 64, 1, bias=False,
                coalition_size=coalition_size
            ),
            LPBatchNorm1d(64),
            LPReLU(),
            LPConv1D(64, 128, 1, bias=False),
            LPBatchNorm1d(128),
            LPReLU(),
            LPConv1D(128, config.num_units_mlp, 1, bias=False),
            LPBatchNorm1d(config.num_units_mlp),
            LPMaxLayer(),
            LPLinear(config.num_units_mlp, config.num_units_fc_1, bias=False),
            LPBatchNorm1d(config.num_units_fc_1),
            LPReLU(),
            LPLinear(config.num_units_fc_1, config.num_units_fc_2, bias=False),
            LPBatchNorm1d(config.num_units_fc_2),
            LPReLU(),
            LPLinear(config.num_units_fc_2, config.num_units_deep, bias=False),
            LPBatchNorm1d(config.num_units_deep),
            LPReLU(),
            LPLinear(config.num_units_deep, config.num_outputs),
        )

        offset = 0
        for key, mod in enumerate(layers):
            self.add_module(str(key + offset), mod)
            # account for Flatten
            if isinstance(mod, LPMaxLayer):
                offset = 1

    def forward(self, inputs: InputLayerInputs) -> InputLayerOutputs:
        input_wo_i, input_w_i = self._modules["0"](inputs)

        for key in filter(lambda x: x != "0", self._modules.keys()):
            layer = self._modules[key]
            input_w_i = layer(input_w_i)
            input_wo_i = layer(input_wo_i)

        return input_wo_i, input_w_i
