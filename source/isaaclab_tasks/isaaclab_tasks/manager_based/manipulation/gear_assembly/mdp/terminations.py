# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gear_on_shaft(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
    shaft_local_offset: tuple[float, float, float] = (-0.045375, 0.0, 0.014),
    bore_local_offset: tuple[float, float, float] = (-0.045375, 0.0, 0.0),
    xy_threshold: float = 0.005,
    z_threshold: float = 0.003,
    insertion_depth: float = 0.04,
) -> torch.Tensor:
    """Returns True when the gear's bore is aligned with the shaft.

    Both positions are computed in world frame, applying each object's quaternion
    to its local-frame offset — this is essential because `randomize_gear_pose`
    rotates the gear with a random yaw every episode:

        shaft_world = gear_base_pos + R(gear_base_quat) @ shaft_local_offset
        bore_world  = gear_pos      + R(gear_quat)      @ bore_local_offset
    """
    gear: RigidObject = env.scene[gear_cfg.name]
    gear_base: RigidObject = env.scene[gear_base_cfg.name]

    shaft_off = torch.tensor(shaft_local_offset, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    shaft_pos = gear_base.data.root_pos_w + math_utils.quat_apply(gear_base.data.root_quat_w, shaft_off)

    bore_off = torch.tensor(bore_local_offset, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    bore_pos = gear.data.root_pos_w + math_utils.quat_apply(gear.data.root_quat_w, bore_off)

    xy_dist = torch.linalg.norm(bore_pos[:, :2] - shaft_pos[:, :2], dim=-1)
    z_near = bore_pos[:, 2] > (shaft_pos[:, 2] - insertion_depth)
    z_at_shaft = torch.abs(bore_pos[:, 2] - shaft_pos[:, 2]) < z_threshold

    return (xy_dist < xy_threshold) & z_near & z_at_shaft
