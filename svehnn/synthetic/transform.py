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
"""
Library of utils to perturb and transform data.
"""
from typing import Union

import numpy as np
import torch

Angle = Union[torch.Tensor, float]


def add_linear_noise(pc: torch.Tensor) -> torch.Tensor:
    noise = torch.randn((3, 16)) * 0.03
    return (pc + noise).transpose(1, 0)


def _check_point_cloud(point_cloud: torch.Tensor) -> None:
    shape = point_cloud.shape
    if len(shape) != 2:
        raise ValueError(
            "Rank mismatch: Rank of point_cloud (received %d)"
            "should be 2." % len(shape)
        )
    pc_shape = list(shape)
    if pc_shape[-1] != 3:
        raise ValueError(
            "Dimension mismatch: last dimension of point_cloud (received %d)"
            "should to be 3." % pc_shape[-1]
        )


def _random_angle(minval: float, maxval: float) -> torch.Tensor:
    return torch.distributions.uniform.Uniform(minval, maxval).sample()


def rotate_point_cloud_x(
    point_cloud: torch.Tensor, rotation_angle: Angle
) -> torch.Tensor:
    """Rotate the point cloud about the X-axis."""
    cosval = torch.cos(rotation_angle)
    sinval = torch.sin(rotation_angle)
    rotation_matrix = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cosval, -sinval], [0.0, sinval, cosval]]
    )
    rotated_data = torch.matmul(point_cloud, rotation_matrix)
    return rotated_data


def rotate_point_cloud_y(
    point_cloud: torch.Tensor, rotation_angle: Angle
) -> torch.Tensor:
    """Rotate the point cloud about the Y-axis."""
    _check_point_cloud(point_cloud)
    cosval = torch.cos(rotation_angle)
    sinval = torch.sin(rotation_angle)
    rotation_matrix = torch.tensor(
        [[cosval, 0.0, sinval], [0.0, 1.0, 0.0], [-sinval, 0.0, cosval]]
    )
    rotated_data = torch.matmul(point_cloud, rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(
    point_cloud: torch.Tensor, rotation_angle: Angle
) -> torch.Tensor:
    """Rotate the point cloud about the Z-axis."""
    cosval = torch.cos(rotation_angle)
    sinval = torch.sin(rotation_angle)
    rotation_matrix = torch.tensor(
        [[cosval, -sinval, 0.0], [sinval, cosval, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_data = torch.matmul(point_cloud, rotation_matrix)
    return rotated_data


def random_rotate_point_cloud_x(
    point_cloud: torch.Tensor, minval: float = 0.0, maxval: float = 2.0 * np.pi
) -> torch.Tensor:
    """Randomly rotate the point cloud about the X-axis."""
    rotation_angle = _random_angle(minval, maxval)
    return rotate_point_cloud_x(point_cloud, rotation_angle)


def random_rotate_point_cloud_y(
    point_cloud: torch.Tensor, minval: float = 0.0, maxval: float = 2.0 * np.pi
) -> torch.Tensor:
    """Randomly rotate the point cloud about the Y-axis."""
    rotation_angle = _random_angle(minval, maxval)
    return rotate_point_cloud_y(point_cloud, rotation_angle)


def random_rotate_point_cloud_z(
    point_cloud: torch.Tensor, minval: int = 0, maxval: int = 2.0 * np.pi
) -> torch.Tensor:
    """Randomly rotate the point cloud about the Z-axis."""
    rotation_angle = _random_angle(minval, maxval)
    return rotate_point_cloud_z(point_cloud, rotation_angle)


def center_and_scale(point_cloud: torch.Tensor) -> torch.Tensor:
    point_cloud = point_cloud - torch.mean(point_cloud, dim=0, keepdim=True)  # center
    max_dist = torch.max(torch.norm(point_cloud, p="fro", dim=1))
    normed = torch.div(point_cloud, max_dist)  # scale
    return normed
