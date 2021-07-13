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
from torch.distributions import Normal

from .base import LPLayerInput
from .utils import square

__all__ = ["LPReLU"]


class LPReLU(torch.nn.Module):
    def __init__(self):
        super(LPReLU, self).__init__()
        self.epsilon = 1e-7

    def forward(self, mv: LPLayerInput) -> LPLayerInput:
        input_mean, input_var = mv
        normal = Normal(
            torch.tensor([0.0], dtype=torch.float32, device=input_mean.device),
            torch.tensor([1.0], dtype=torch.float32, device=input_mean.device),
        )

        v = torch.clamp(input_var, min=self.epsilon)
        s = torch.sqrt(v)
        m_div_s = input_mean / s
        prob = torch.exp(normal.log_prob(m_div_s))
        m_out = input_mean * normal.cdf(m_div_s) + s * prob
        v_out = (
            (square(input_mean) + v) * normal.cdf(m_div_s)
            + (input_mean * s) * prob
            - square(m_out)
        )
        return m_out, v_out
