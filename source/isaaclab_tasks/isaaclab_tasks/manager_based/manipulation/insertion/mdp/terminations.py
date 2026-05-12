# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def peg_inserted(
    env: ManagerBasedRLEnv,
    peg_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
    socket_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
    xy_threshold: float = 0.003,
    z_threshold: float = 0.005,
    socket_tip_offset: float = 0.1435,
    insertion_depth: float = 0.04,
) -> torch.Tensor:
    """Returns True when the peg center is within xy_threshold of the socket axis and
    near the socket opening (within insertion_depth below the socket tip).

    insertion_depth prevents false positives: without it, any peg position below the
    socket tip (including table height) would satisfy the Z condition.
    """
    peg: RigidObject = env.scene[peg_cfg.name]
    socket: Articulation = env.scene[socket_cfg.name]

    peg_pos = peg.data.root_pos_w
    socket_pos = socket.data.root_pos_w

    xy_dist = torch.linalg.norm(peg_pos[:, :2] - socket_pos[:, :2], dim=-1)
    socket_tip_z = socket_pos[:, 2] + socket_tip_offset
    # Peg must be near the socket opening: within insertion_depth below the tip,
    # not just anywhere below it (which would fire even at table height).
    z_near_tip = peg_pos[:, 2] > (socket_tip_z - insertion_depth)
    z_below_tip = peg_pos[:, 2] < (socket_tip_z + z_threshold)

    return (xy_dist < xy_threshold) & z_near_tip & z_below_tip
