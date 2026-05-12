# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import ArticulationRootPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import (
    randomize_object_pose,
)

from . import mdp

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/AutoMate/00015"

# Height from socket root to the socket opening (tip). Measured from AutoMate assembly config.
SOCKET_TIP_OFFSET = 0.1435
# Peg half-height for placing on table surface.
PEG_HALF_HEIGHT = 0.025


@configclass
class InsertionSceneCfg(InteractiveSceneCfg):
    """Scene definition: robot, peg (held_asset), socket (fixed_asset), table, plane, light."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # Peg: dynamic rigid body, starts on the table, robot must grasp it.
    held_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.25, PEG_HALF_HEIGHT)),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_DIR}/plug.usd",
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=5.0,
                max_contact_impulse=1e32,
            ),
            mass_props=MassPropertiesCfg(mass=0.019),
        ),
    )

    # Socket: fixed articulation (kinematic), stays on the table.
    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.05),
            # Explicitly empty: overrides the default {".*": 0.0} which would
            # call find_joints([".*"]) on a joint-less articulation and crash.
            joint_pos={},
            joint_vel={},
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_DIR}/socket.usd",
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
            ),
            mass_props=MassPropertiesCfg(mass=0.05),
        ),
        actuators={},
    )

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        peg_pos = ObsTerm(func=mdp.peg_pos_w)
        peg_quat = ObsTerm(func=mdp.peg_quat_w)
        socket_pos = ObsTerm(func=mdp.socket_pos_w)
        socket_quat = ObsTerm(func=mdp.socket_quat_w)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Binary signals used for MimicGen subtask annotation."""

        grasp = ObsTerm(
            func=mdp.peg_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "peg_cfg": SceneEntityCfg("held_asset"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class RewardsCfg:
    # Phase 1: approach and grasp the peg.
    ee_to_peg = RewTerm(func=mdp.ee_to_peg_distance, weight=0.5)
    # Phase 2: bring peg to socket.
    peg_to_socket = RewTerm(
        func=mdp.peg_to_socket_distance,
        weight=1.0,
        params={"socket_tip_offset": SOCKET_TIP_OFFSET},
    )
    # Success bonus.
    success = RewTerm(
        func=mdp.insertion_success_bonus,
        weight=5.0,
        params={"socket_tip_offset": SOCKET_TIP_OFFSET},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Named `success` so record_demos.py extracts it (and stops the env from
    # auto-resetting during teleop — the script drives the save+reset cycle).
    # During RL training, this term stays active and ends the episode on success.
    success = DoneTerm(
        func=mdp.peg_inserted,
        params={"socket_tip_offset": SOCKET_TIP_OFFSET},
    )


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_peg_pose = EventTerm(
        func=randomize_object_pose,
        mode="reset",
        params={
            # Keep the peg on the table surface, randomize XY and yaw.
            "pose_range": {
                "x": (0.35, 0.55),
                "y": (0.10, 0.30),
                "z": (PEG_HALF_HEIGHT, PEG_HALF_HEIGHT),
                "yaw": (-3.14159, 3.14159),
            },
            "min_separation": 0.05,
            "asset_cfgs": [SceneEntityCfg("held_asset")],
        },
    )


@configclass
class InsertionEnvCfg(ManagerBasedRLEnvCfg):
    """Base config for the peg-in-hole insertion task."""

    scene: InsertionSceneCfg = InsertionSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Gripper config used by observation and event functions.
    gripper_joint_names: list = ["panda_finger_joint.*"]
    gripper_open_val: float = 0.04
    gripper_threshold: float = 0.005

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.decimation = 4  # 30 Hz control at 120 Hz physics
        self.viewer.eye = (2.23, 1.0, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.3)
        self.sim = SimulationCfg(
            dt=1 / 120,
            physx=PhysxCfg(
                solver_type=1,
                max_position_iteration_count=192,
                max_velocity_iteration_count=1,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.00625,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**23,
                # Single partition is critical for stable contact-rich simulation.
                gpu_max_num_partitions=1,
            ),
        )
