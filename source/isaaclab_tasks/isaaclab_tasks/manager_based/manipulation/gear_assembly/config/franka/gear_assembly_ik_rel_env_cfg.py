# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.gear_assembly.gear_assembly_env_cfg import GearAssemblyEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack import mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip

# Ready pose: bent elbow, EEF above table workspace.
_READY_POSE = [0.0, -0.4, 0.0, -1.8, 0.0, 1.8, 0.785]


@configclass
class FrankaGearAssemblyIKRelEnvCfg(GearAssemblyEnvCfg):
    """Franka gear assembly with IK-relative EEF delta-pose control.

Scene at reset:
  - gear_small and gear_medium (non-kinematic) parked at the gear base position
  - held_asset (factory_gear_large) randomised on the table (negative-Y side)
  Robot must pick up held_asset and place it on the large shaft.
"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # init_state.joint_pos → data.default_joint_pos → written to PhysX by
        # reset_joints_by_offset at each episode reset.
        self.scene.robot.init_state.joint_pos = {
            "panda_joint1": _READY_POSE[0],
            "panda_joint2": _READY_POSE[1],
            "panda_joint3": _READY_POSE[2],
            "panda_joint4": _READY_POSE[3],
            "panda_joint5": _READY_POSE[4],
            "panda_joint6": _READY_POSE[5],
            "panda_joint7": _READY_POSE[6],
            "panda_finger_joint.*": 0.04,
        }

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
