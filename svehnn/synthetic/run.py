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
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import DataModule
from .evaluate import check_all_predictions_close, evaluation_report, mse, ndcg, rank_correlation, rmse
from .explain import explain_exact, explain_occlusion, explain_sampling, explain_svehnn, get_baseline_diff
from .lp_model import TheLPPointnet
from .model import PointNetConfig, ThePointNet
from .train import Trainer

device = None


def fit_model_and_save(experiment_dir, train_loader, config):
    pointnet = ThePointNet(config)
    print(pointnet)

    num_params = sum(map(lambda x: x.numel(), pointnet.parameters()))
    print("PARAMETERS:", num_params)

    trainer = Trainer(
        pointnet,
        train_loader,
        lr=0.001,
        weight_decay=0.001,
        device=device,
    )
    trainer.train(num_epochs=300)

    torch.save(pointnet.state_dict(), str(experiment_dir / "pointnet.pth"))
    return pointnet


def run_sv_exact(pointnet, explain_loader, num_points, experiment_dir):
    sv_exact = explain_exact(pointnet, num_points, explain_loader, device)

    # print(sv_exact)

    print("Saving Exact SV")
    with open(experiment_dir / "sv_exact.pkl", "wb") as fout:
        pickle.dump(sv_exact, fout, protocol=pickle.HIGHEST_PROTOCOL)


def run_sv_sampling(pointnet, explain_loader, num_points, experiment_dir, n_steps):
    sv_sampling = explain_sampling(
        pointnet, num_points, explain_loader, device, n_steps=n_steps,
    )

    # print(sv_sampling)

    print("Saving Sampled SV")
    with open(experiment_dir / f"sv_sampling_{n_steps}.pkl", "wb") as fout:
        pickle.dump(sv_sampling, fout, protocol=pickle.HIGHEST_PROTOCOL)


def make_lp_pointnet(pointnet, config, test_loader):
    lp_pointnet = TheLPPointnet(config)
    lp_pointnet = lp_pointnet.to(device)
    lp_pointnet.load_state_dict(pointnet.state_dict())
    print(lp_pointnet)

    check_all_predictions_close(pointnet, lp_pointnet, test_loader, device)
    return lp_pointnet


def run_sv_svehnn(lp_pointnet, explain_loader, num_points, experiment_dir):
    sv_svehnn = explain_svehnn(lp_pointnet, num_points, explain_loader, device)

    with open(experiment_dir / "sv_svehnn.pkl", "wb") as fout:
        pickle.dump(sv_svehnn, fout, protocol=pickle.HIGHEST_PROTOCOL)


def run_sv_occlusion(pointnet, explain_loader, num_points, experiment_dir):
    sv_occlusion = explain_occlusion(pointnet, num_points, explain_loader, device)

    with open(experiment_dir / "sv_occlusion.pkl", "wb") as fout:
        pickle.dump(sv_occlusion, fout, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate_all(experiment_dir: Path, ref_diff: np.ndarray):
    paths = (
        experiment_dir / "sv_exact.pkl",
        experiment_dir / "sv_sampling_2000.pkl",
        experiment_dir / "sv_sampling_32.pkl",
        experiment_dir / "sv_occlusion.pkl",
        experiment_dir / "sv_svehnn.pkl",
    )
    sv = {}
    for path in paths:
        with open(path, "rb") as fin:
            sv[path.stem] = pickle.load(fin)

    completness = {
        key: np.absolute(val.sum(1) - ref_diff).squeeze(1)
        for key, val in sv.items()
    }
    pd.DataFrame.from_dict(completness).to_csv(
        experiment_dir / "completness-delta.csv"
    )

    reference = "sv_exact"
    sv_ref = sv.pop(reference)

    results_rmse = pd.Series(
        {
            key: rmse(sv_ref, val) for key, val in sv.items()
        },
        name="RMSE",
    )
    results_mse = pd.Series(
        {
            key: mse(sv_ref, val) for key, val in sv.items()
        },
        name="MSE",
    )
    results_ndcg = pd.Series(
        {
            key: ndcg(sv_ref, val) for key, val in sv.items()
        },
        name="NDCG"
    )
    results_corr = pd.Series(
        {
            key: rank_correlation(sv_ref, val) for key, val in sv.items()
        },
        name="SRC"
    )
    results = pd.concat((results_rmse, results_mse, results_corr, results_ndcg), axis=1)
    print(results)

    results.to_csv(experiment_dir / "metrics.csv")


def main(seed: int = 868662447):
    global device

    torch.manual_seed(seed)
    random.seed(seed)

    experiment_dir = Path(f"outputs-synthetic-seed{seed}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    data_mod = DataModule()
    data_mod.save(experiment_dir / "data.pkl")
    train_loader = data_mod.train_loader()
    test_loader = data_mod.test_loader()
    explain_loader = data_mod.explain_loader()

    config = PointNetConfig(
        num_points=16,
        num_units_deep=8,
        num_units_mlp=10,
        num_units_fc_1=5,
        num_units_fc_2=2,
        num_outputs=1,
    )
    device = torch.device("cuda:0")
    pointnet = fit_model_and_save(experiment_dir, train_loader, config)

    evaluation_report(pointnet, test_loader, device)

    ref_diff = get_baseline_diff(pointnet, config.num_points, explain_loader, device)

    print("Difference we are going to explain:")
    print(ref_diff)

    run_sv_exact(pointnet, explain_loader, config.num_points, experiment_dir)
    run_sv_sampling(pointnet, explain_loader, config.num_points, experiment_dir, n_steps=2000)
    run_sv_sampling(pointnet, explain_loader, config.num_points, experiment_dir, n_steps=32)
    run_sv_occlusion(pointnet, explain_loader, config.num_points, experiment_dir)

    lp_pointnet = make_lp_pointnet(pointnet, config, test_loader)
    run_sv_svehnn(lp_pointnet, explain_loader, config.num_points, experiment_dir)

    evaluate_all(experiment_dir, ref_diff)


if __name__ == "__main__":
    main()
