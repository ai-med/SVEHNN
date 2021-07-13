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
import torch
import torch.nn
from torch.distributions.normal import Normal

from .base import LPLayerInput
from .utils import square

__all__ = ["LPMaxLayer"]


def _ab_max_pooling(
    mv1: LPLayerInput, mv2: LPLayerInput, normal: Normal, eps: float = 1e-10,
) -> LPLayerInput:
    """
    Takes the probabilistic maximum of two input distributions.

    Args:
        mv1 (LPLayerInput): Mean and variance of the first input distribution.
        mv2 (LPLayerInput): Mean and variance of the other input distribution.

    Returns:
        Mean and variance with the maximum of the two distributions.
    """
    mu_a, va = mv1
    mu_b, vb = mv2
    vavb = torch.sqrt(torch.clamp(va + vb, min=eps))
    assert torch.isfinite(vavb).all().item() != 0

    muamub = mu_a - mu_b
    muamub_p = mu_a + mu_b
    alpha = muamub / vavb

    mu_c = vavb * torch.exp(normal.log_prob(alpha)) + muamub * normal.cdf(alpha) + mu_b
    vc = muamub_p * vavb * torch.exp(normal.log_prob(alpha))
    vc += (
        (square(mu_a) + va) * normal.cdf(alpha)
        + (square(mu_b) + vb) * (1.0 - normal.cdf(alpha))
        - square(mu_c)
    )

    return mu_c, vc


class LPMaxLayer(torch.nn.Module):
    """Lightweight Probabilistic Max operation"""

    def __init__(self):
        super(LPMaxLayer, self).__init__()
        self.normal = Normal(loc=0.0, scale=1.0)

    def forward(self, mv: LPLayerInput) -> LPLayerInput:
        m, v = mv

        # unpack along time dimension
        m_chunks = torch.split(m, 1, dim=2)
        v_chunks = torch.split(v, 1, dim=2)
        # initialize with values of first time point
        m_a = m_chunks[0].squeeze(dim=2)
        v_a = v_chunks[0].squeeze(dim=2)
        for m_i, v_i in zip(m_chunks[1:], v_chunks[1:]):
            m_i.squeeze_(dim=2)
            v_i.squeeze_(dim=2)
            m_a, v_a = _ab_max_pooling((m_a, v_a), (m_i, v_i), self.normal)

        return m_a, v_a
