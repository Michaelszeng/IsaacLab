# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.insertion.insertion_env_cfg import InsertionEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack import mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaInsertionIKRelEnvCfg(InsertionEnvCfg):
    """Franka Panda insertion env with IK-relative EEF control."""

    def __post_init__(self):
        super().__post_init__()

        # Stiffer PD gains for better IK tracking during contact.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Override init pose: bent elbow, EEF above table workspace.
        # data.default_joint_pos is initialized from init_state.joint_pos and then
        # written to PhysX by reset_joints_by_offset at each episode reset.
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.4,
            "panda_joint3": 0.0,
            "panda_joint4": -1.8,
            "panda_joint5": 0.0,
            "panda_joint6": 1.8,
            "panda_joint7": 0.785,
            "panda_finger_joint.*": 0.04,
        }

        # EEF frame: offset from panda_hand to fingertip midpoint.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.107]),
                ),
            ],
        )

        # IK-relative delta pose control.
        # scale=0.5 means action in [-1,1] maps to delta in [-0.5, 0.5] m/rad.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger_joint.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        self.gripper_joint_names = ["panda_finger_joint.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005
