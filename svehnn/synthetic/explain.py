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
from more_itertools import powerset
from scipy.special import factorial
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.no_grad()
def get_baseline_diff(
    model: nn.Module,
    n_players: int,
    explain_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model = model.eval().to(device)

    diffs = []
    baseline = torch.zeros(1, 3, n_players, device=device)

    for batch in explain_loader:
        pc = batch[0].to(device)

        ref_diff = (model(pc) - model(baseline)).detach().cpu().numpy()
        diffs.append(ref_diff)

    return np.row_stack(diffs)


def coef(q: int, s: int) -> float:
    return factorial(q, exact=True) * factorial(s - q - 1, exact=True) / factorial(s, exact=True)


@torch.no_grad()
def explain_exact(
    model: nn.Module,
    n_players: int,
    explain_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model = model.eval().to(device)

    max_batch_size = explain_loader.batch_size

    attributions = np.zeros((len(explain_loader) * max_batch_size, n_players, 1), dtype=float)

    pbar = tqdm(total=n_players * len(explain_loader), desc="Exact SV")
    baseline = torch.zeros(1, 3, n_players, device=device).expand(
        max_batch_size, -1, -1)

    for batch_idx, batch in enumerate(explain_loader):
        pc = batch[0].to(device)
        indices = np.arange(n_players)
        batch_start = batch_idx * max_batch_size
        batch_size = pc.size()[0]
        batch_end = batch_start + batch_size

        for i in range(n_players):
            indices_wo_i = np.delete(indices, i)
            for subset in powerset(indices_wo_i):
                pc_indices = torch.as_tensor(subset + (i,), dtype=torch.long, device=device)
                copy_indices = torch.arange(len(subset) + 1, dtype=torch.long, device=device)
                pc_in = baseline.index_copy(
                    2,
                    copy_indices,
                    pc[..., pc_indices]).repeat(2, 1, 1)
                pc_in[batch_size:, :, len(subset)] = baseline[..., i]

                pred = model(pc_in)
                pred_w_i, pred_wo_i = torch.chunk(pred, 2)
                delta = pred_w_i - pred_wo_i

                cc = coef(len(subset), n_players)
                attributions[batch_start:batch_end, i] += cc * delta.detach().cpu().numpy()

            pbar.update()

    return attributions


@torch.no_grad()
def explain_sampling(
    model: nn.Module,
    n_players: int,
    explain_loader: DataLoader,
    device: torch.device,
    n_steps: int = 2000,
) -> np.ndarray:
    max_batch_size = explain_loader.batch_size
    model = model.eval().to(device)

    pbar = tqdm(total=n_steps * n_players * len(explain_loader), desc="Sampling SV")

    baseline = torch.zeros(1, 3, n_players, device=device).expand(
        max_batch_size, -1, -1)

    attributions = np.zeros(
        (max_batch_size * len(explain_loader), n_players, 1), dtype=float
    )
    for batch_idx, (pc, _) in enumerate(explain_loader):
        pc = pc.to(device)
        batch_start = batch_idx * max_batch_size
        batch_size = pc.size()[0]
        batch_end = batch_start + batch_size

        for _ in range(n_steps):
            perm = torch.randperm(n_players, device=device)
            pc_in = baseline.repeat(2, 1, 1)
            for end in range(n_players):
                i = perm[end]
                # pc_in already includes real values that come before `j` in `perm`
                # only need to add current point to first half
                pc_in[:batch_size, :, i] = pc[..., i]

                pred = model(pc_in)
                pred_w_i, pred_wo_i = torch.chunk(pred, 2)
                delta = pred_w_i - pred_wo_i

                attributions[batch_start:batch_end, i] += delta.detach().cpu().numpy()

                # add current point to second half, such that both parts are equal
                pc_in[batch_size:, :, i] = pc[..., i]

                pbar.update()

    attributions /= n_steps

    return attributions


@torch.no_grad()
def explain_svehnn(
    lp_model: nn.Module, n_players: int, explain_loader: DataLoader, device: torch.device,
) -> np.ndarray:
    batch_size = explain_loader.batch_size

    model = lp_model.eval().to(device)

    pbar = tqdm(total=n_players * len(explain_loader), desc="SVEHNN")

    attributions = np.zeros(
        (batch_size * len(explain_loader), n_players, 1), dtype=float, order="F"
    )
    baseline = torch.zeros(1, 3, n_players, device=device)
    for batch_idx, (pc, _) in enumerate(explain_loader):
        pc = pc.to(device)
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size

        ks = torch.arange(n_players, dtype=torch.float, device=device)
        ks = ks.repeat(batch_size).unsqueeze(dim=1)

        for i in range(n_players):
            mask = torch.zeros(1, 1, n_players, device=device)
            mask[..., i] = 1.0
            mask_in = mask.expand(n_players * batch_size, -1, -1)

            pc_in = pc.repeat_interleave(n_players, dim=0)

            mv_wo_i, mv_with_i = model((pc_in, mask_in, baseline, ks))

            means_with_i = torch.chunk(mv_with_i[0], batch_size)
            means_wo_i = torch.chunk(mv_wo_i[0], batch_size)

            for n, m_w_i, m_wo_i in zip(range(batch_start, batch_end), means_with_i, means_wo_i):
                # take mean over all ks
                diff = torch.mean(m_w_i - m_wo_i, dim=0).detach().cpu().numpy()
                attributions[n, i] = diff

            pbar.update()

    return attributions


@torch.no_grad()
def explain_occlusion(
    model: nn.Module, n_players: int, explain_loader: DataLoader, device: torch.device,
) -> np.ndarray:
    model = model.eval().to(device)

    max_batch_size = explain_loader.batch_size

    pbar = tqdm(total=n_players * len(explain_loader), desc="Occlusion")

    attributions = np.zeros(
        (max_batch_size * len(explain_loader), n_players, 1), dtype=float
    )
    for batch_idx, (pc, _) in enumerate(explain_loader):
        pc = pc.to(device)
        batch_start = batch_idx * max_batch_size
        batch_size = pc.size()[0]
        batch_end = batch_start + batch_size

        for i in range(n_players):
            mask = torch.ones(1, 1, n_players, device=device)
            mask[..., i] = 0.0
            mask_in = mask.expand(batch_size, -1, -1)

            pc_masked = pc * mask_in
            pc_in = torch.cat((pc, pc_masked), dim=0)

            pred = model(pc_in)
            pred_w_i, pred_wo_i = torch.chunk(pred, 2)
            delta = pred_w_i - pred_wo_i

            attributions[batch_start:batch_end, i] = delta.detach().cpu().numpy()

            pbar.update()

    return attributions
