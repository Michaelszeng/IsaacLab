# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import numpy as np

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
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import randomize_object_pose

from . import mdp


# ---------------------------------------------------------------------------
# Camera helper: convert (eye, lookat) into a quaternion (w,x,y,z) suitable for
# CameraCfg.OffsetCfg(convention="ros").  IsaacLab's "ros" convention is the
# camera OPTICAL frame:  +X right, +Y down, +Z forward (i.e. look direction is
# the camera's +Z axis).  NOT to be confused with ROS body/vehicle convention
# (+X forward, +Y left, +Z up) — they're named the same in different contexts.
# ---------------------------------------------------------------------------
def _lookat_quat_ros(eye, lookat, world_up=(0.0, 0.0, 1.0)):
    eye, lookat = np.array(eye, float), np.array(lookat, float)
    world_up = np.array(world_up, float)
    fwd = lookat - eye
    fwd /= np.linalg.norm(fwd)
    # Top-down singularity: forward parallel to up → swap reference axis.
    if abs(float(np.dot(fwd, world_up))) > 0.999:
        world_up = np.array([1.0, 0.0, 0.0])
    rgt = np.cross(fwd, world_up)
    rgt /= np.linalg.norm(rgt)
    down = np.cross(fwd, rgt)
    # Camera optical frame columns: X=right, Y=down, Z=forward.
    R = np.stack([rgt, down, fwd], axis=1)
    t = R.trace()
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


# Scene camera positions match _PRESET_VIEWS[0] and [1] in record_demos.py.
#   View 1: front close-up — near-head-on, slightly above workspace
#   View 2: low rear-left — looking forward/up toward the workspace
_FRONT_EYE = (1.429, -0.082, 0.76)
_FRONT_LOOKAT = (0.158, -0.091, 0.206)
_REAR_LEFT_EYE = (0.232, -0.279, 0.001)
_REAR_LEFT_LOOKAT = (1.367, 0.769, 0.466)

# ---------------------------------------------------------------------------
# Shaft layout (gear base LOCAL frame, 90° Z rotation → local X maps to world Y).
#
#   small  shaft: local [+0.076125, 0, 0]  → world offset from base [0, +0.076125, 0]
#   medium shaft: local [+0.030375, 0, 0]  → world offset from base [0, +0.030375, 0]
#   large  shaft: local [-0.045375, 0, 0]  → world offset from base [0, -0.045375, 0]
#
# Task design:
#   - gear_small and gear_medium park at the gear base position (non-kinematic).
#   - gear_large (held_asset) is randomised on the table — the robot picks it up
#     and places it on the large shaft to complete the assembly.
#   - The gear base is kinematic (fixed in place on the table).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Two-offset success geometry (must apply BOTH rotations to be yaw-invariant):
#
#   shaft_world = gear_base_pos + R(gear_base_quat) @ SHAFT_LOCAL_OFFSET
#   bore_world  = gear_pos      + R(gear_quat)      @ BORE_LOCAL_OFFSET
# ---------------------------------------------------------------------------

# Large-gear shaft offset in the gear base's local frame.
# X component from the original Factory deploy task's gear_offsets dict.
# Z component pushes the target up to where the seated gear actually rests
# (teleop traces show bore_z ≈ 0.058–0.071 when seated, base_z = 0.05).
SHAFT_LOCAL_OFFSET = (-0.045375, 0.0, 0.0)

# Large-gear bore offset in the gear's local frame.
# Verified from teleop data: the gear's root prim is placed at the gear base
# origin, so the bore is offset by the same vector that locates the large
# shaft on the base.  Spinning the gear on the shaft traces a circle of
# radius |this offset| ≈ 0.0454 m around the shaft — confirms the value.
BORE_LOCAL_OFFSET = (-0.045375, 0.0, 0.0)

# Approximate gear half-height — tune after viewing the assets in the simulator.
GEAR_HALF_HEIGHT = 0.02

_GEAR_BASE_POS = (0.5, 0.0, 0.05)
_GEAR_BASE_ROT = (0.70711, 0.0, 0.0, 0.70711)  # 90° Z rotation


def _gear_rigid_props(kinematic: bool = False) -> RigidBodyPropertiesCfg:
    return RigidBodyPropertiesCfg(
        disable_gravity=False,
        kinematic_enabled=kinematic,
        solver_position_iteration_count=64,
        solver_velocity_iteration_count=1,
        max_depenetration_velocity=5.0,
        max_contact_impulse=1e32,
    )


@configclass
class GearAssemblySceneCfg(InteractiveSceneCfg):
    """Scene: Franka, large gear (held_asset), small+medium parked at gear base, gear base (fixed_asset)."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # ---- Active gear: large — picked up from the table, placed on the large shaft ----
    held_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HeldAsset",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.1, GEAR_HALF_HEIGHT)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_large/factory_gear_large.usd",
            activate_contact_sensors=True,
            rigid_props=_gear_rigid_props(kinematic=False),
        ),
    )

    # ---- Non-active gears: non-kinematic, parked at the gear base position ----
    # Matches original Factory RL task: non-active gears initialise at the same
    # position as the gear base (they are NOT on their shafts; they just pile up
    # at the base origin and can be knocked around freely).
    gear_small: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GearSmall",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_GEAR_BASE_POS,
            rot=_GEAR_BASE_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_small/factory_gear_small.usd",
            activate_contact_sensors=False,
            rigid_props=_gear_rigid_props(kinematic=False),
        ),
    )

    gear_medium: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GearMedium",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_GEAR_BASE_POS,
            rot=_GEAR_BASE_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_medium/factory_gear_medium.usd",
            activate_contact_sensors=False,
            rigid_props=_gear_rigid_props(kinematic=False),
        ),
    )

    # ---- Gear base: kinematic (fixed on table, provides the 3 shafts) ----------
    fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/FixedAsset",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_GEAR_BASE_POS,
            rot=_GEAR_BASE_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/factory_gear_base/factory_gear_base.usd",
            activate_contact_sensors=True,
            rigid_props=_gear_rigid_props(kinematic=True),
        ),
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

    # ---- Cameras for visuomotor data recording -------------------------------
    # Wrist camera: parented to panda_hand, follows the EEF.
    wrist_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Mounted 12 cm out on +X (clear of the panda_hand body), 2 cm
            # behind the hand origin, tilted ~10° toward the fingertip so the
            # EEF appears closer to the centre of the frame.
            pos=(0.05, 0.0, -0.01),
            rot=(0.70442, -0.06163, -0.06163, 0.70442),
            convention="ros",
        ),
    )

    # Scene camera 1: front close-up (= _PRESET_VIEWS[0]).
    scene_cam_front: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/scene_cam_front",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_FRONT_EYE,
            rot=_lookat_quat_ros(_FRONT_EYE, _FRONT_LOOKAT),
            convention="ros",
        ),
    )

    # Scene camera 2: low rear-left (= _PRESET_VIEWS[1]).
    scene_cam_rear_left: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/scene_cam_rear_left",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_REAR_LEFT_EYE,
            rot=_lookat_quat_ros(_REAR_LEFT_EYE, _REAR_LEFT_LOOKAT),
            convention="ros",
        ),
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
        gear_pos = ObsTerm(func=mdp.gear_pos_w)
        gear_quat = ObsTerm(func=mdp.gear_quat_w)
        shaft_pos = ObsTerm(
            func=mdp.shaft_pos_w,
            params={"shaft_local_offset": SHAFT_LOCAL_OFFSET},
        )
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        actions = ObsTerm(func=mdp.last_action)

        # Camera image observations — recorded into the HDF5 dataset.
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        scene_cam_front = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("scene_cam_front"), "data_type": "rgb", "normalize": False},
        )
        scene_cam_rear_left = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("scene_cam_rear_left"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp = ObsTerm(
            func=mdp.gear_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "gear_cfg": SceneEntityCfg("held_asset"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class RewardsCfg:
    ee_to_gear = RewTerm(func=mdp.ee_to_gear_distance, weight=0.5)
    gear_to_shaft = RewTerm(
        func=mdp.gear_to_shaft_distance,
        weight=1.0,
        params={
            "shaft_local_offset": SHAFT_LOCAL_OFFSET,
            "bore_local_offset": BORE_LOCAL_OFFSET,
        },
    )
    success = RewTerm(
        func=mdp.gear_on_shaft_bonus,
        weight=5.0,
        params={
            "shaft_local_offset": SHAFT_LOCAL_OFFSET,
            "bore_local_offset": BORE_LOCAL_OFFSET,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Named `success` so record_demos.py extracts it (and stops the env from
    # auto-resetting during teleop — the script drives the save+reset cycle).
    # During RL training, this term stays active and ends the episode on success.
    success = DoneTerm(
        func=mdp.gear_on_shaft,
        params={
            "shaft_local_offset": SHAFT_LOCAL_OFFSET,
            "bore_local_offset": BORE_LOCAL_OFFSET,
        },
    )


@configclass
class EventCfg:
    # Resets all objects (including non-active gears) to their init_state positions.
    # Non-active gears reset to gear base position — matching the original task behaviour.
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Randomise only the active (large) gear on the table surface.
    randomize_gear_pose = EventTerm(
        func=randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.25, 0.45),
                "y": (-0.30, -0.05),  # negative-Y side, away from gear base cluster
                "z": (GEAR_HALF_HEIGHT, GEAR_HALF_HEIGHT),
                "yaw": (-3.14159, 3.14159),
            },
            "min_separation": 0.05,
            "asset_cfgs": [SceneEntityCfg("held_asset")],
        },
    )


@configclass
class GearAssemblyEnvCfg(ManagerBasedRLEnvCfg):
    """Franka gear mesh assembly (MimicGen-compatible).

    Mirrors original Factory task structure:
      - All 3 gears are non-kinematic (free to rotate/fall).
      - Non-active gears (small, medium) park at the gear base position on reset.
      - Active gear (large) is randomised on the table for the human to pick up.
      - Gear base is kinematic (fixed on table).
    """

    scene: GearAssemblySceneCfg = GearAssemblySceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    gripper_joint_names: list = ["panda_finger_joint.*"]
    gripper_open_val: float = 0.04
    gripper_threshold: float = 0.005

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.decimation = 4
        # Head-on view with 30° azimuth: camera at 2 m horizontal radius, 1.5 m
        # height, rotated 30° from directly in front of the workspace.
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
                gpu_max_num_partitions=1,
            ),
        )
        # Re-render a few frames after reset so the cameras settle (matches
        # the visuomotor stack env's approach).
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        # List used by visuomotor consumers (e.g., MimicGen image generation).
        self.image_obs_list = ["wrist_cam", "scene_cam_front", "scene_cam_rear_left"]
