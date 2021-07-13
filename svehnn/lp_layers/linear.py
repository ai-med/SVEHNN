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
import torch.nn
from torch.nn import functional as F

from .base import LPLayerInput
from .utils import square

__all__ = ["LPLinear"]


class LPLinear(torch.nn.Linear):
    """
    Lightweight probabilistic linear layer, with inputs being normally distributed.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(LPLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def forward(self, inputs: LPLayerInput) -> LPLayerInput:
        input_mean, input_var = inputs
        m = F.linear(input_mean, self.weight, self.bias)
        v = F.linear(input_var, square(self.weight))

        return m, v
