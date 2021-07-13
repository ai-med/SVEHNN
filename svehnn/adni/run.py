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
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import DataModule
from .explain import explain_svehnn
from .lp_model import LPWidePointNetClassifier
from .model import WidePointNetClassifier, predict


def load_model(checkpoint_path: str, config: Dict[str, Any]) -> nn.Module:
    model = WidePointNetClassifier(
        num_features=config["num_features"],
        num_outputs=config["num_outputs"],
        num_global=config["num_global"],
        num_deep=config["num_deep"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)

    model = model.eval()
    return model


def get_differences(
    model: nn.Module, ref_loader: DataLoader, other_loader: DataLoader, device: torch.device,
) -> np.ndarray:
    pred_ref = predict(model, ref_loader, device)
    pred_other = predict(model, other_loader, device)

    diff = pred_ref["logits"] - pred_other["logits"]
    return diff


def make_zero_explain_loader(pc_loader: DataLoader) -> DataLoader:
    new_dataset = []
    for dp in pc_loader.dataset:
        dp = list(dp)
        dp.insert(2, torch.zeros(dp[0].size()[0], 1, dtype=dp[0].dtype))
        new_dataset.append(tuple(dp))
    return DataLoader(new_dataset, batch_size=pc_loader.batch_size)


def make_hull_explain_loader(pc_loader: DataLoader, hull_loader: DataLoader) -> DataLoader:
    new_dataset = []
    for dp_pc, dp_hull in zip(pc_loader.dataset, hull_loader.dataset):
        dp = list(dp_pc)
        dp.insert(2, dp_hull[0])
        new_dataset.append(tuple(dp))
    return DataLoader(new_dataset, batch_size=pc_loader.batch_size)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["ad", "control"], required=True)
    parser.add_argument("--baseline", choices=["zero", "hull"], required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    checkpoint = args.checkpoint

    config = {
        "num_outputs": 1,
        "num_global": 600,
        "num_deep": 100,
        "num_features": 31,
        "num_points": 1024,
    }

    model = load_model(checkpoint, config)

    device = torch.device("cuda")

    data_mod = DataModule()
    test_loader = data_mod.test_loader()
    test_true = data_mod.test_labels()
    test_pred = predict(model, test_loader, device)

    auc = roc_auc_score(y_true=test_true, y_score=test_pred["prob"])
    print(f"Test AUROC: {auc:.3f}")
    del auc

    bacc = balanced_accuracy_score(y_true=test_true, y_pred=test_pred["class_idx"])
    print(f"Test bACC: {bacc:.3f}")
    del bacc

    out_name = f"adni-svehnn-{args.subset}-{args.baseline}"

    idx_range = None
    if args.start is not None:
        idx_range = (args.start, args.end)
        out_name = f"{out_name}-{args.start}-{args.end}"

    out_dir = args.output_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=False)

    if args.subset == "ad":
        pc_loader, hull_loader = data_mod.true_ad_loader(arange=idx_range)
    elif args.subset == "control":
        pc_loader, hull_loader = data_mod.true_control_loader(arange=idx_range)
    else:
        raise ValueError(f"{args.subset} is unsupported")

    # maps i-th entry of pc_loader to j-th entry of splits["test"]
    matched_ids = data_mod.find_pc(pc_loader.dataset)
    del matched_ids

    if idx_range is None:
        idx_start, idx_end = 0, len(pc_loader.dataset)
    else:
        idx_start, idx_end = idx_range

    zeros_loader = DataLoader(
        [(torch.zeros(config["num_points"], 3), torch.zeros(config["num_features"]), 0)], batch_size=1
    )
    ref_diff_zero = get_differences(model, pc_loader, zeros_loader, device)
    print('# ZERO')
    print(ref_diff_zero)

    ref_diff_hull = get_differences(model, pc_loader, hull_loader, device)
    print('\n# HULL')
    print(ref_diff_hull)

    diff = pd.DataFrame.from_dict(
        {"ZERO": ref_diff_zero.squeeze(1), "HULL": ref_diff_hull.squeeze(1)}
    )
    diff.index = list(range(idx_start, idx_end))
    diff.to_csv(
        out_dir / "baseline-differences.csv"
    )
    del diff, ref_diff_zero, ref_diff_hull, zeros_loader

    lp_model = LPWidePointNetClassifier(
        num_features=config["num_features"],
        num_outputs=config["num_outputs"],
        num_global=config["num_global"],
        num_deep=config["num_deep"],
    )

    lp_model.load_state_dict(model.state_dict())
    print(lp_model)

    if args.baseline == "zero":
        explain_loader = make_zero_explain_loader(pc_loader)
    elif args.baseline == "hull":
        explain_loader = make_hull_explain_loader(pc_loader, hull_loader)
    else:
        raise ValueError(f"{args.baseline} is unsupported")

    sv_svehnn = explain_svehnn(
        lp_model, config["num_points"], config["num_features"], explain_loader, device
    )

    with open(out_dir / "sv_svehnn.pkl", "wb") as fout:
        pickle.dump(dict(zip(range(idx_start, idx_end), sv_svehnn)), fout, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
