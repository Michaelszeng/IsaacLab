# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gear_pos_w(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
) -> torch.Tensor:
    """Active gear position in the environment frame."""
    gear: RigidObject = env.scene[gear_cfg.name]
    return gear.data.root_pos_w - env.scene.env_origins


def gear_quat_w(
    env: ManagerBasedRLEnv,
    gear_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
) -> torch.Tensor:
    """Active gear orientation (w, x, y, z)."""
    gear: RigidObject = env.scene[gear_cfg.name]
    return gear.data.root_quat_w


def shaft_pos_w(
    env: ManagerBasedRLEnv,
    gear_base_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
    shaft_local_offset: tuple[float, float, float] = (0.076125, 0.0, 0.0),
) -> torch.Tensor:
    """Target shaft position in the environment frame.

    Computed by transforming shaft_local_offset from the gear base's local frame
    into world frame, then subtracting env origin so the policy sees a consistent
    frame regardless of multi-env tiling.
    """
    gear_base: RigidObject = env.scene[gear_base_cfg.name]
    offset = torch.tensor(shaft_local_offset, device=env.device).unsqueeze(0).expand(env.num_envs, -1)
    shaft_pos = gear_base.data.root_pos_w + math_utils.quat_apply(gear_base.data.root_quat_w, offset)
    return shaft_pos - env.scene.env_origins


def gear_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    gear_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Binary signal: 1 when the robot is holding the gear, 0 otherwise.

    Checks both proximity (gear near EEF) and gripper closure.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    gear: RigidObject = env.scene[gear_cfg.name]

    gear_pos = gear.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(gear_pos - ee_pos, dim=1)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    assert len(gripper_joint_ids) == 2

    finger_not_open = torch.logical_and(
        torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]] - env.cfg.gripper_open_val)
        > env.cfg.gripper_threshold,
        torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]] - env.cfg.gripper_open_val)
        > env.cfg.gripper_threshold,
    )
    return torch.logical_and(pose_diff < diff_threshold, finger_not_open)
