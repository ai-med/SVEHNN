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
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import classification_report, confusion_matrix, ndcg_score
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluation_report(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    model = model.eval().to(device)

    y_true = []
    y_pred = []
    for pc, label in loader:
        pc = pc.to(device)

        logits = model(pc)
        assert logits.dim() == 2 and logits.size()[1] == 1
        probs = torch.sigmoid(logits)
        predictions = torch.cat((1 - probs, probs), dim=1)
        class_idx = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        y_pred.append(class_idx)
        y_true.append(label.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    report = classification_report(
        y_true=y_true, y_pred=y_pred)
    print(report)

    mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("Confusion matrix:")
    print("-----------------")
    print(mat)


@torch.no_grad()
def check_all_predictions_close(
    model: nn.Module, lp_model: nn.Module, loader: DataLoader, device: torch.device,
) -> None:
    model = model.eval().to(device)
    lp_model = lp_model.eval().to(device)

    y_model = []
    y_lp = []
    logits_model = []
    logits_lp = []

    for batch in loader:
        pc = batch[0].to(device)

        logits = model(pc)
        assert logits.dim() == 2 and logits.size()[1] == 1
        probs = torch.sigmoid(logits)
        predictions = torch.cat((1 - probs, probs), dim=1)
        class_idx = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        logits_model.append(logits.detach().cpu().numpy())
        y_model.append(class_idx)

        in_shape = pc.size()
        mask = torch.zeros((in_shape[0], 1, in_shape[2]), device=device)
        mask[..., 0] = 1
        baseline = torch.zeros_like(pc)

        set_size = torch.tensor(
            [in_shape[2] - 1] * in_shape[0],
            dtype=torch.float,
            device=device
        ).unsqueeze(dim=-1)
        mv_wo, mv_with = lp_model((pc, mask, baseline, set_size))

        logits = mv_with[0]
        assert logits.dim() == 2 and logits.size()[1] == 1
        probs = torch.sigmoid(logits)
        predictions = torch.cat((1 - probs, probs), dim=1)
        class_idx = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        logits_lp.append(logits.detach().cpu().numpy())
        y_lp.append(class_idx)

    y_model = np.concatenate(y_model)
    y_lp = np.concatenate(y_lp)

    logits_model = np.row_stack(logits_model)
    logits_lp = np.row_stack(logits_lp)

    print("Same class prediction:", np.mean(y_model == y_lp))
    print("Max logits difference:", np.max(np.absolute(logits_model - logits_lp)))


def mse(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("shape mismatch")

    return np.mean(np.square(a - b))


def rmse(a: Sequence[float], b: Sequence[float]) -> float:
    return np.sqrt(mse(a, b))


def ndcg(true_sv: Sequence[float], approx_sv: Sequence[float], k: Optional[int] = None) -> float:
    true_sv = np.asarray(true_sv)
    approx_sv = np.asarray(approx_sv)
    y_true = np.absolute(true_sv.squeeze(2))
    y_score = np.absolute(approx_sv.squeeze(2))
    return ndcg_score(y_true, y_score, k=k)


def rank_correlation(true_sv: Sequence[float], approx_sv: Sequence[float]) -> float:
    true_sv = np.asarray(true_sv)
    approx_sv = np.asarray(approx_sv)
    correlations = []
    for a, b in zip(true_sv.squeeze(2), approx_sv.squeeze(2)):
        corr, _pval = spearmanr(a, b, nan_policy="raise")
        # if values are constant, the correlation coefficent
        # is not defined, which results in nan
        if np.isfinite(corr):
            correlations.append(corr)
    return np.mean(correlations)
