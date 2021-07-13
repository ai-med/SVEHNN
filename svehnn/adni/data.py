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
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader


class BaseDataModule(metaclass=ABCMeta):
    @abstractmethod
    def test_loader(self) -> DataLoader:
        """Return a :class:`DataLoader` to evaluate the model on (test data)."""

    @abstractmethod
    def test_labels(self) -> np.ndarray:
        """"Return an array of the labels from the test data."""

    @abstractmethod
    def true_ad_loader(
        self, batch_size: int = 3, arange: Optional[Tuple[int, int]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """"Return two :class:`DataLoader` with correctly classified AD patients.

        Args:
            batch_size (int): The batch size to use.
            arange (tuple|None): Return DataLoaders that are restricted to
            samples with indices in the interval `start` to `end` (excluding).

        Returns:
            pc_feature_loader (DataLoader): A :class:`DataLoader` that returns
            a tuple consisting of
                1. the actual hippocampus point cloud, shape = (N, 3, NUM_POINTS).
                2. the tabular data, shape = (N, NUM_FEATURES).
                3. the class label, shape = (N, 1).
            baseline_loader (DataLoader):  A :class:`DataLoader` that returns
            a tuple consisting of
                1. the non-informative baseline shape for the
               corresponding entry in `pc_feature_loader`, shape = (N, 3, NUM_POINTS).
        """

    @abstractmethod
    def true_control_loader(
        self, batch_size: int = 3, arange: Optional[Tuple[int, int]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """"Return two :class:`DataLoader` with correctly classified healthy control patients.

        Args:
            batch_size (int): The batch size to use.
            arange (tuple|None): Return DataLoaders that are restricted to
            samples with indices in the interval `start` to `end` (excluding).

        Returns:
            pc_feature_loader (DataLoader): A :class:`DataLoader` that returns
            a tuple consisting of
                1. the actual hippocampus point cloud, shape = (N, 3, NUM_POINTS).
                2. the tabular data, shape = (N, NUM_FEATURES).
                3. the class label, shape = (N, 1).
            baseline_loader (DataLoader):  A :class:`DataLoader` that returns
            a tuple consisting of
                1. the non-informative baseline shape for the
               corresponding entry in `pc_feature_loader`, shape = (N, 3, NUM_POINTS).
        """


class DataModule(BaseDataModule):
    def __init__(self):
        raise NotImplementedError("You need to implement svehnn.data.BaseDataModule")

    def test_loader(self) -> DataLoader:
        raise NotImplementedError()

    def test_labels(self) -> np.ndarray:
        raise NotImplementedError()

    def true_ad_loader(
        self, batch_size: int = 3, arange: Optional[Tuple[int, int]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError()

    def true_control_loader(
        self, batch_size: int = 3, arange: Optional[Tuple[int, int]] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError()
