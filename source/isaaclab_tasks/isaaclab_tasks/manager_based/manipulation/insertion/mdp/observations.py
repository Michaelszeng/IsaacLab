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


def peg_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
) -> torch.Tensor:
    """Peg position relative to the env origin."""
    peg: RigidObject = env.scene[asset_cfg.name]
    return peg.data.root_pos_w - env.scene.env_origins


def peg_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
) -> torch.Tensor:
    """Peg orientation in world frame (w, x, y, z)."""
    peg: RigidObject = env.scene[asset_cfg.name]
    return peg.data.root_quat_w


def socket_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
) -> torch.Tensor:
    """Socket position relative to the env origin."""
    socket: Articulation = env.scene[asset_cfg.name]
    return socket.data.root_pos_w - env.scene.env_origins


def socket_quat_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("fixed_asset"),
) -> torch.Tensor:
    """Socket orientation in world frame (w, x, y, z)."""
    socket: Articulation = env.scene[asset_cfg.name]
    return socket.data.root_quat_w


def peg_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    peg_cfg: SceneEntityCfg = SceneEntityCfg("held_asset"),
    proximity_threshold: float = 0.06,
) -> torch.Tensor:
    """Returns 1.0 if the peg is near the EEF and the gripper is closed, else 0.0.

    Shape: (num_envs, 1)
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    peg: RigidObject = env.scene[peg_cfg.name]

    # EEF proximity to peg
    peg_pos = peg.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    dist = torch.linalg.norm(peg_pos - ee_pos, dim=-1)
    near_peg = dist < proximity_threshold

    # Gripper closed check
    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    finger_pos = robot.data.joint_pos[:, gripper_joint_ids[0]]
    gripper_closed = finger_pos < (env.cfg.gripper_open_val - env.cfg.gripper_threshold)

    return (near_peg & gripper_closed).float().unsqueeze(-1)
