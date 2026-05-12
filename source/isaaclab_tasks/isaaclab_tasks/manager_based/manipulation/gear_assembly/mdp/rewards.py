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
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_to_gear_distance(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    gear_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
) -> torch.Tensor:
    """Negative reward proportional to EEF-to-gear distance. Encourages grasping."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    gear: RigidObject = env.scene[gear_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    gear_pos = gear.data.root_pos_w
    return -torch.linalg.norm(ee_pos - gear_pos, dim=-1)


def gear_to_shaft_distance(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
    shaft_local_offset: tuple[float, float, float] = (-0.045375, 0.0, 0.0),
    bore_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Negative reward = XY misalignment + Z distance between gear bore and shaft.

    Both endpoints rotate with their parent body's quaternion so the reward is
    yaw-invariant under random gear placement.
    """
    gear: RigidObject = env.scene[gear_cfg.name]
    gear_base: RigidObject = env.scene[gear_base_cfg.name]

    shaft_off = torch.tensor(shaft_local_offset, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    shaft_pos = gear_base.data.root_pos_w + math_utils.quat_apply(gear_base.data.root_quat_w, shaft_off)

    bore_off = torch.tensor(bore_local_offset, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    bore_pos = gear.data.root_pos_w + math_utils.quat_apply(gear.data.root_quat_w, bore_off)

    xy_dist = torch.linalg.norm(bore_pos[:, :2] - shaft_pos[:, :2], dim=-1)
    z_dist = torch.abs(bore_pos[:, 2] - shaft_pos[:, 2])
    return -(xy_dist + z_dist)


def gear_on_shaft_bonus(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
    shaft_local_offset: tuple[float, float, float] = (-0.045375, 0.0, 0.0),
    bore_local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    xy_threshold: float = 0.005,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """Binary bonus when the gear's bore is aligned over the shaft."""
    gear: RigidObject = env.scene[gear_cfg.name]
    gear_base: RigidObject = env.scene[gear_base_cfg.name]

    shaft_off = torch.tensor(shaft_local_offset, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    shaft_pos = gear_base.data.root_pos_w + math_utils.quat_apply(gear_base.data.root_quat_w, shaft_off)

    bore_off = torch.tensor(bore_local_offset, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    bore_pos = gear.data.root_pos_w + math_utils.quat_apply(gear.data.root_quat_w, bore_off)

    xy_dist = torch.linalg.norm(bore_pos[:, :2] - shaft_pos[:, :2], dim=-1)
    z_near = torch.abs(bore_pos[:, 2] - shaft_pos[:, 2]) < z_threshold

    return ((xy_dist < xy_threshold) & z_near).float()
