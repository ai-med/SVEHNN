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
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .transform import _random_angle, add_linear_noise, rotate_point_cloud_x, rotate_point_cloud_y, rotate_point_cloud_z

DataLabelPair = Tuple[np.ndarray, int]
DataItemList = List[DataLabelPair]


def make_x() -> Tensor:
    """Creates a point cloud tensor template of an X shape."""
    return torch.tensor(
        [
            [
                0.25,
                0.33,
                0.3,
                0.43,
                0.4,
                0.58,
                0.62,
                0.73,
                0.7,
                0.15,
                0.27,
                0.27,
                0.39,
                0.65,
                0.8,
                0.82,
            ],
            [
                0.2,
                0.2,
                0.38,
                0.4,
                0.58,
                0.6,
                0.8,
                0.8,
                0.9,
                0.88,
                0.88,
                0.7,
                0.75,
                0.4,
                0.25,
                0.36,
            ],
            [
                0.1,
                0.2,
                0.0,
                0.08,
                0.03,
                0.11,
                0.14,
                0.02,
                0.0,
                0.15,
                0.0,
                0.06,
                0.02,
                0.12,
                0.0,
                0.2,
            ],
        ]
    )


def make_line() -> Tensor:
    """Creates a point cloud tensor template of a straight line."""
    line = torch.tensor(
        [
            [
                0.5,
                0.53,
                0.55,
                0.42,
                0.5,
                0.44,
                0.52,
                0.43,
                0.47,
                0.52,
                0.49,
                0.47,
                0.49,
                0.48,
                0.5,
                0.51,
            ],
            [
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.84,
                0.88,
                0.9,
            ],
            [
                0.1,
                0.2,
                0.0,
                0.08,
                0.03,
                0.11,
                0.14,
                0.02,
                0.0,
                0.15,
                0.0,
                0.06,
                0.02,
                0.12,
                0.0,
                0.2,
            ],
        ]
    )
    return line


@torch.no_grad()
def add_noise_and_rotation(pc_template: Tensor) -> np.ndarray:
    """Adds random noise and rotation to a template point cloud.

    Args:
        pc_template (Tensor): point cloud template which should be perturbed.

    Returns:
        point_cloud (np.ndarray): modified point cloud.
    """
    pc = add_linear_noise(pc_template)
    if bool(random.getrandbits(1)):
        pc = rotate_point_cloud_x(pc, _random_angle(-1, +1))
    if bool(random.getrandbits(1)):
        pc = rotate_point_cloud_y(pc, _random_angle(-1, +1))
    if bool(random.getrandbits(1)):
        pc = rotate_point_cloud_z(pc, _random_angle(-1, +1))
    # convert to numpy for reproducibility
    return pc.numpy()


def generate_dataset(
    class0_template: Tensor, class1_template: Tensor, num_examples: int
) -> List[DataLabelPair]:
    """
    Creates a synthetic dataset based on single example templates for each class.
    Template tensors are perturbed by random to get many examples of the same class.

    Args:
        class0_template (Tensor): Template tensor as representative for class 0.
        class1_template (Tensor): Template tensor as representative for class 1.
        num_examples (int): Number of examples to generate per class.

    Returns:
        list: A List with length `2 * num_examples` containing a Tuple
        of a point cloud and the corresponding label.
    """
    x_dataset = [
        (add_noise_and_rotation(class0_template), 0) for _ in range(num_examples)
    ]
    line_dataset = [
        (add_noise_and_rotation(class1_template), 1) for _ in range(num_examples)
    ]
    return x_dataset + line_dataset


class PointCloudDataset(Dataset):
    def __init__(
        self, data: DataItemList, transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        self.data = data
        self.transform = transform

    def __getitem__(self, index: int) -> DataLabelPair:
        data = self.data[index]
        if self.transform is not None:
            data = (self.transform(data[0]),) + data[1:]
        return data

    def __len__(self) -> int:
        return len(self.data)


class DataModule:
    """
    Build a synthetic point dataset for PointNet.
    Created are Xand I (line) shapes which should be distinguished by the network.
    """

    def __init__(self) -> None:
        self.train_data = self.generate_data(500)
        self.test_data = self.generate_data(50)
        self._pc_transform = transforms.Lambda(lambda x: x.transpose(1, 0))

    def generate_data(self, num_examples: int) -> List[DataLabelPair]:
        x_template = make_x()
        line_template = make_line()

        ds = generate_dataset(x_template, line_template, num_examples)
        return ds

    def train_loader(self) -> DataLoader:
        train_loader = DataLoader(
            PointCloudDataset(self.train_data, transform=self._pc_transform),
            shuffle=True,
            drop_last=True,
            batch_size=500,
        )
        return train_loader

    def test_loader(self) -> DataLoader:
        test_loader = DataLoader(
            PointCloudDataset(self.test_data, transform=self._pc_transform),
            batch_size=50,
        )
        return test_loader

    def explain_loader(self) -> DataLoader:
        data = self.test_data
        explain_loader = DataLoader(
            PointCloudDataset(data, transform=self._pc_transform),
            batch_size=len(data),
            pin_memory=True,
        )
        return explain_loader

    def save(self, filename: Union[Path, str]) -> None:
        data = {"train_data": self.train_data, "test_data": self.test_data}
        with open(filename, "wb") as fout:
            pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)
