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
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def spaced_elements(array: np.ndarray, num_elems: int, **kwargs) -> torch.Tensor:
    """
    Selects equally spaced coalition sizes from all possible sizes.
    Args:
        array (): tensor containing the range from 0 to the number of players
        num_elems (): number of sizes to be selected

    Returns: a tensor of size num_elems containing valid coalition sizes (ks)
    """
    return torch.tensor([x[len(x) // 2] for x in np.array_split(array, num_elems)], **kwargs)


@torch.no_grad()
def explain_svehnn(
    lp_model: nn.Module,
    n_players_pc: int,
    n_players_tabular: int,
    explain_loader: DataLoader,
    device: torch.device,
    n_steps: int = 150,
) -> np.ndarray:
    max_batch_size = explain_loader.batch_size

    model = lp_model.to(device).eval()

    pbar = tqdm(total=n_players_pc * len(explain_loader))

    attributions = np.zeros(
        (max_batch_size * len(explain_loader), n_players_pc + n_players_tabular, 1), dtype=float, order="F"
    )
    for batch_idx, batch in enumerate(explain_loader):
        pc = batch[0].to(device)
        pc.transpose_(2, 1)
        feat = batch[1].to(device)
        pc_baseline = batch[2].to(device)
        pc_baseline.transpose_(2, 1)

        batch_start = batch_idx * max_batch_size
        batch_size = pc.size()[0]
        batch_end = batch_start + batch_size

        tabular_weights = model.fc4.weight_tabular
        for i in range(tabular_weights.size()[0]):  # iterate over outputs of last layer
            attr_tabular = feat * tabular_weights[i].unsqueeze(dim=0)
            attributions[batch_start:batch_end, n_players_pc:, i] = attr_tabular.detach().cpu().numpy()

        ks = spaced_elements(np.arange(n_players_pc), n_steps, dtype=torch.float, device=device)
        ks = ks.repeat(batch_size).unsqueeze(dim=1)

        pc_in = pc.repeat_interleave(n_steps, dim=0)
        pc_baseline_in = pc_baseline.repeat_interleave(n_steps, dim=0)
        feat_in = feat.repeat_interleave(n_steps, dim=0)

        for i in range(n_players_pc):
            mask = torch.zeros(1, 1, n_players_pc, device=device)
            mask[..., i] = 1.0
            mask_in = mask.expand(n_steps * batch_size, -1, -1)

            mv_wo_i, mv_with_i = model((pc_in, mask_in, pc_baseline_in, ks), feat_in)

            means_with_i = torch.chunk(mv_with_i[0], batch_size)
            means_wo_i = torch.chunk(mv_wo_i[0], batch_size)

            for n, m_w_i, m_wo_i in zip(range(batch_start, batch_end), means_with_i, means_wo_i):
                # take mean over all ks
                diff = torch.mean(m_w_i - m_wo_i, dim=0).detach().cpu().numpy()
                attributions[n, i] = diff

            del mv_wo_i, mv_with_i, means_with_i, means_wo_i, mask, mask_in

            pbar.update()

    return attributions
