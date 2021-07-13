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
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d

from .base import InputLayerInputs, InputLayerOutputs
from .utils import square

__all__ = ["LPConv1D", "ProbConv1DInput"]


class LPConv1D(Conv1d):
    """
    Propagate distributions over a probabilistic Conv1D layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(LPConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        mean, variance = inputs
        m = F.conv1d(
            mean,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        v = F.conv1d(
            variance,
            weight=square(self.weight),
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return m, v


class ProbConv1DInput(Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        coalition_size: int,
    ) -> None:
        super(ProbConv1DInput, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.coalition_size = coalition_size
        self.epsilon = 1e-7

    def _conv1d(self, inputs, weight):
        return F.conv1d(
            inputs,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(
        self,
        inputs: InputLayerInputs,
    ) -> InputLayerOutputs:
        input_, mask, baseline, k = inputs
        assert input_.dim() == mask.dim(), "Inputs and mask must have same rank."
        size_coalition = self.coalition_size

        mask_in = mask.expand_as(input_)
        baseline = baseline.expand_as(input_).to(input_.device)

        # get n_players from shape without batch dimension
        n_players = np.prod(input_.size()[1:])
        # account for size of coalition
        n_players = torch.tensor(
            n_players / size_coalition,
            dtype=torch.float, device=input_.device)

        k = torch.unsqueeze(k, -1)
        assert mask_in.dim() == input_.dim(), \
            "Inputs must have the same number of dimensions."
        one = torch.as_tensor([1.0], dtype=torch.float, device=input_.device)

        cond = mask_in == 0
        in_ghost = cond.to(dtype=torch.float)
        inputs_i = torch.where(cond, input_, baseline)

        conv_m = self._conv1d(input_, self.weight)
        conv_m_i = self._conv1d(inputs_i, self.weight)
        conv_count = self._conv1d(
            in_ghost, weight=torch.ones_like(self.weight)
        )
        # If we mask complete points, conv_count will be zero for the removed point,
        # leading to division by zero later on
        # assert torch.gt(conv_count, 0).all().item() != 0
        conv_v_i = self._conv1d(square(inputs_i), square(self.weight))

        kn = torch.div(k, n_players)
        # here using k' = conv_count * k / n_players is not necessary
        # because conv_count cancels out when computing mean
        m_wo_i = torch.mul(conv_m_i, kn)
        # account for i-th point, which is non-random
        # m_w_i = m_wo_i + (conv_m - conv_m_i)
        mask_out = mask.expand_as(m_wo_i)
        m_w_i = torch.where(mask_out == 1, conv_m, m_wo_i)

        # Compensate for number of players in the coalition
        k = torch.mul(conv_count, kn)
        v_wo_i = torch.div(conv_v_i, conv_count) - square(torch.div(conv_m_i, conv_count))
        v_wo_i = v_wo_i * k * (one - (k - one) / (conv_count - one))

        # find the index of point i (masked out)
        # note: current approach only works if i is masked out as complete point (- all coordinates)
        # so we can find out which index was masked by looking at dim 2 only
        idx = torch.nonzero(mask_in[0, 0, :] == 1).squeeze(dim=1)
        # the point i is non-random -> set all variances for point i to 0
        v_wo_i[:, :, idx] = 0.0
        v_wo_i.clamp_(min=self.epsilon)
        v_w_i = v_wo_i

        if isinstance(self.bias, torch.nn.Parameter):
            b = self.bias.view(-1, 1)
            m_wo_i.add_(b)
            m_w_i.add_(b)

        return (m_wo_i, v_wo_i), (m_w_i, v_w_i)
