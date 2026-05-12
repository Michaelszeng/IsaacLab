# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_to_peg_distance(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
) -> torch.Tensor:
    """Negative reward proportional to the distance between EEF and peg center.

    Encourages the robot to approach and grasp the peg.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    peg: RigidObject = env.scene[peg_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    peg_pos = peg.data.root_pos_w
    return -torch.linalg.norm(ee_pos - peg_pos, dim=-1)


def peg_to_socket_distance(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
    socket_tip_offset: float = 0.1435,
) -> torch.Tensor:
    """Negative reward proportional to the XY distance between peg and socket opening.

    socket_tip_offset is the height from the socket root to the socket tip (opening).
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    socket: Articulation = env.scene[socket_cfg.name]

    peg_pos = peg.data.root_pos_w
    socket_pos = socket.data.root_pos_w

    # XY alignment distance
    xy_dist = torch.linalg.norm(peg_pos[:, :2] - socket_pos[:, :2], dim=-1)
    # Z proximity: reward being close to the socket tip height
    socket_tip_z = socket_pos[:, 2] + socket_tip_offset
    z_dist = torch.abs(peg_pos[:, 2] - socket_tip_z)

    return -(xy_dist + z_dist)


def insertion_success_bonus(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
    xy_threshold: float = 0.003,
    z_threshold: float = 0.005,
    socket_tip_offset: float = 0.1435,
) -> torch.Tensor:
    """Binary bonus reward when the peg is aligned over and lowered into the socket opening."""
    peg: RigidObject = env.scene[peg_cfg.name]
    socket: Articulation = env.scene[socket_cfg.name]

    peg_pos = peg.data.root_pos_w
    socket_pos = socket.data.root_pos_w

    xy_dist = torch.linalg.norm(peg_pos[:, :2] - socket_pos[:, :2], dim=-1)
    socket_tip_z = socket_pos[:, 2] + socket_tip_offset
    z_below_tip = peg_pos[:, 2] < (socket_tip_z + z_threshold)

    inserted = (xy_dist < xy_threshold) & z_below_tip
    return inserted.float()
