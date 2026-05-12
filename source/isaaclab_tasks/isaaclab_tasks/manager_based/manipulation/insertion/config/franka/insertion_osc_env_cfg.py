# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.insertion.insertion_env_cfg import InsertionEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack import mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# ---------------------------------------------------------------------------
# Franka in torque-control mode: zero arm stiffness/damping so OSC torques
# are applied directly without a PD controller fighting them.
# disable_gravity=True means the sim applies no gravity to the robot links;
# the OSC must NOT add gravity_compensation in this case (that would push
# the arm in the wrong direction).
# ---------------------------------------------------------------------------
FRANKA_TORQUE_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_TORQUE_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_TORQUE_CFG.actuators["panda_shoulder"].stiffness = 0.0
FRANKA_TORQUE_CFG.actuators["panda_shoulder"].damping = 0.0
FRANKA_TORQUE_CFG.actuators["panda_forearm"].stiffness = 0.0
FRANKA_TORQUE_CFG.actuators["panda_forearm"].damping = 0.0


@configclass
class FrankaInsertionOSCEnvCfg(InsertionEnvCfg):
    """Franka insertion env with Operational Space Control (OSC).

    OSC tracks absolute EEF pose targets using joint torques (impedance
    control). This gives:
      - All joints contribute to motion (whole-body, natural feel)
      - Orientation is actively held when rotation is not commanded
      - No arc-sweep drift like DifferentialIK
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_TORQUE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Full starting pose: bent elbow, EEF over the table workspace, and
        # panda_joint5 = 0.5 to stay clear of the wrist singularity at joint5 = 0.
        # nullspace_joint_pos_target="default" will drive toward this configuration.
        _READY_POSE = [0.0, -0.4, 0.0, -1.8, 0.5, 1.8, 0.785]
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

        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
            # "default" drives null space toward the robot's default_joint_pos,
            # which keeps the arm near its initial configuration.
            # Required when nullspace_control="position" (cannot be "none").
            nullspace_joint_pos_target="default",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                # Gains from AutoMate: Kp_pos=100, Kp_rot=30.
                motion_stiffness_task=(400.0, 400.0, 400.0, 80.0, 80.0, 80.0),
                motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                nullspace_control="position",
                nullspace_stiffness=10.0,
                nullspace_damping_ratio=1.0,
                # True requires computing Lambda = (J * M^-1 * J^T)^-1.
                # When J is rank-deficient (e.g., near panda_joint5 = 0 wrist
                # singularity), Lambda is ill-conditioned and torques blow up —
                # concentrating entirely on the shoulder. Use False for stability.
                inertial_dynamics_decoupling=False,
                gravity_compensation=True,
            ),
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
