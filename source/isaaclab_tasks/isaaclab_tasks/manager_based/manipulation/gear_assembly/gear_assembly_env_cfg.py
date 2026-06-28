# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import random
from dataclasses import MISSING

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
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
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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
        solver_velocity_iteration_count=2,
        max_depenetration_velocity=10.0,
        max_contact_impulse=1e32,
        # Body-level viscous damping. Independent of contact direction, so it
        # damps tangential/rotational gear motion (sliding or yaw oscillation
        # in the gripper's grasp) that compliant-contact damping — which only
        # acts in the contact normal direction — can't touch. ``angular_damping``
        # is set higher because the observed oscillation is primarily rotational.
        # Both are velocity-proportional drag, so a stationary held gear is
        # unaffected.
        linear_damping=0.5,
        angular_damping=5.0,
    )


# Larger-than-default contact offsets on the gear shells so that any
# approaching shape (gripper finger, another gear, the gear-base structure)
# generates a contact constraint *before* the bodies overlap. This gives the
# solver more substeps to ramp up the contact force, which prevents the
# gripper from punching through the gear during a forceful insertion press
# without raising ``max_depenetration_velocity`` (which would re-introduce
# the gear-on-gear jitter we just fixed).
#
# - contact_offset = 5 mm  → contacts are generated up to 5 mm before overlap.
#   For Factory gears whose teeth are ~3-4 mm thick this is generous but
#   leaves no visible air gap because rest_offset stays small.
# - rest_offset    = 0.5 mm → bodies sit ~0.5 mm apart "at rest". Small enough
#   to be visually invisible; large enough that the gripper-gear contact
#   stays in the well-conditioned regime where PhysX is stable.
# - PhysX requires contact_offset >= 2 * rest_offset.
def _gear_collision_props() -> CollisionPropertiesCfg:
    return CollisionPropertiesCfg(
        contact_offset=0.005,
        rest_offset=0.0005,
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
            collision_props=_gear_collision_props(),
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
            collision_props=_gear_collision_props(),
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
            collision_props=_gear_collision_props(),
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
            collision_props=_gear_collision_props(),
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


# Per-call counter used purely for the "[event N]" label below. Not behaviour
# critical — just makes it easy to follow which reset a given log block came
# from when scrolling back through a long log.
_gear_event_counter = {"n": 0}


def randomize_held_gear_about_center(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    pose_range: dict,
    bore_local_offset: tuple = BORE_LOCAL_OFFSET,
):
    """Randomize the held gear so ``pose_range`` is interpreted in the gear's GEOMETRIC center.

    The factory_gear_large USD has its root prim on the gear's rim — offset
    from the bore (= visible center) by ``bore_local_offset`` in the gear's
    local frame (default ~4.5 cm along local -X). The vanilla
    ``randomize_object_pose`` writes the sampled position to the root, so any
    yaw randomization swings the visible center along a circle of radius
    ||bore_local_offset|| ≈ 4.5 cm — looks like the gear teleports.

    This version samples a target *center* pose (x, y, z, yaw) and back-solves
    the root translation that places the geometric center on the sample::

        center_world = root_world + R(orient) @ bore_local_offset
        =>  root_world = center_world − R(orient) @ bore_local_offset
    """
    if env_ids is None:
        return
    device = env.device
    asset = env.scene[asset_cfg.name]

    _gear_event_counter["n"] += 1
    event_idx = _gear_event_counter["n"]
    print(
        f"[event {event_idx}] randomize_held_gear_about_center -> samples GEOMETRIC-CENTER pose"
        f" from {pose_range}; compensating for bore offset {bore_local_offset}",
        flush=True,
    )

    range_list = [pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    bore_offset_t = torch.tensor(bore_local_offset, device=device, dtype=torch.float32).unsqueeze(0)  # (1, 3)

    for cur_env in env_ids.tolist():
        # Sample target center pose (env-relative).
        sample_list = [random.uniform(r[0], r[1]) for r in range_list]
        sample = torch.tensor([sample_list], device=device, dtype=torch.float32)
        center_pos_env = sample[:, 0:3]  # (1, 3)
        orient = math_utils.quat_from_euler_xyz(sample[:, 3], sample[:, 4], sample[:, 5])  # (1, 4)

        cx, cy, cz, roll, pitch, yaw = sample_list
        print(
            f"    -> [env {cur_env}] sampled CENTER pose:"
            f" xyz=({cx:.3f}, {cy:.3f}, {cz:.3f})"
            f"  rpy=({roll:.3f}, {pitch:.3f}, {yaw:.3f})"
            f"  (yaw {math.degrees(yaw):.1f}°)",
            flush=True,
        )

        rotated_offset = math_utils.quat_apply(orient, bore_offset_t)  # (1, 3)
        root_pos_env = center_pos_env - rotated_offset
        root_pos_world = root_pos_env + env.scene.env_origins[cur_env : cur_env + 1, 0:3]

        cur_env_t = torch.tensor([cur_env], device=device)
        asset.write_root_pose_to_sim(torch.cat([root_pos_world, orient], dim=-1), env_ids=cur_env_t)
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=cur_env_t)

    # Read back the actual gear root pose after the write so the printed value
    # matches what the rest of the env observes (env-relative).
    pos = (asset.data.root_pos_w - env.scene.env_origins)[0]
    print(
        f"    -> gear pos after randomize_held_gear_about_center: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
        flush=True,
    )


def set_gear_physics_properties(
    env,
    env_ids,
    asset_names: list,
    static_friction: float = 0.05,
    dynamic_friction: float = 0.05,
    restitution: float = 0.1,
    mass_scale: dict | None = None,
    compensate_finger_friction: bool = True,
    robot_name: str = "robot",
    finger_body_regex: str = "panda_.*finger.*",
    default_scene_friction: float = 0.5,
):
    """Set per-shape friction/restitution/mass on the gears via the PhysX tensor view, bypassing USD entirely.

    USD ``bind_physics_material`` cannot reach the collision geometry of
    ``factory_gear_*`` / ``panda_instanceable`` assets: their collision meshes
    are marked ``instanceable`` in the source USD, so the only prims the USD
    importer exposes for them are read-only instance proxies — there is no
    per-instance prim to attach a material *relationship* to. This is why
    every prior run logged "WARNING: Could not perform 'bind_physics_material'
    on any prims...": the bind always failed, silently, and every body in the
    scene (gears, fingers, table) ran on PhysX's scene-default material the
    whole time (static=0.5, dynamic=0.5, restitution=0.0, combine="average").

    This function writes friction/restitution/mass straight into the PhysX
    tensor buffers instead, via ``root_physx_view.set_material_properties`` /
    ``set_masses`` / ``set_inertias`` — the same mechanism used by IsaacLab's
    own ``randomize_rigid_body_material`` / ``randomize_rigid_body_mass``
    event terms (see ``isaaclab.envs.mdp.events``). This is unaffected by
    instancing, and the values can be read back at any time with
    ``get_material_properties`` / ``get_masses`` to verify what was applied.

    .. note::
        This sets absolute friction values but not the PhysX "combine mode"
        (e.g. the previous design's "max") — that is a material-*prim*
        attribute only reachable through the same broken bind path. Under the
        engine's default "average" combine: gear-on-gear contact is exactly
        ``static_friction``/``dynamic_friction`` (averaging two equal values
        is a no-op), but gear-vs-other contact dilutes towards
        ``default_scene_friction``. If ``compensate_finger_friction`` is True,
        the Franka finger links' friction is also raised via the same tensor
        mechanism so that, after averaging with the gear's low friction, the
        *effective* gear-finger friction is unchanged — keeping the grasp
        unaffected. The table is a purely static collider (no
        ``RigidBodyAPI``), so it has no PhysX tensor view at all; gear-vs-table
        friction can't be compensated this way and will be
        ``(static_friction + default_scene_friction) / 2``.

    .. note::
        Compliant contact (stiffness/damping) is *not* exposed by the PhysX
        tensor API (confirmed via isaac-sim/IsaacLab issues #4706 and #2281)
        — it can only be set by binding a USD material, so it remains
        unavailable here until ``instanceable`` is removed from the gear/robot
        collision geometry (requires editing local copies of the USD files).
    """
    num_envs = env.scene.num_envs
    all_env_ids = torch.arange(num_envs, device="cpu")

    for asset_name in asset_names:
        asset = env.scene[asset_name]
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = static_friction
        materials[..., 1] = dynamic_friction
        materials[..., 2] = restitution
        asset.root_physx_view.set_material_properties(materials, all_env_ids)
        print(
            f"[startup] set {asset_name} friction: static={static_friction}, dynamic={dynamic_friction},"
            f" restitution={restitution}",
            flush=True,
        )

        if mass_scale is not None and asset_name in mass_scale:
            scale = mass_scale[asset_name]
            masses = asset.root_physx_view.get_masses()
            inertias = asset.root_physx_view.get_inertias()
            default_mass = masses[0].item()
            masses *= scale
            inertias *= scale
            asset.root_physx_view.set_masses(masses, all_env_ids)
            asset.root_physx_view.set_inertias(inertias, all_env_ids)
            print(
                f"[startup] scaled {asset_name} mass by {scale}x: {default_mass:.4f} kg -> "
                f"{default_mass * scale:.4f} kg",
                flush=True,
            )

    if compensate_finger_friction:
        robot = env.scene[robot_name]
        body_ids, body_names = robot.find_bodies(finger_body_regex)
        if not body_ids:
            raise ValueError(
                f"set_gear_physics_properties: no robot bodies matched '{finger_body_regex}' for"
                " finger-friction compensation."
            )
        # Map body index -> shape index range, mirroring the workaround in
        # isaaclab.envs.mdp.events.randomize_rigid_body_material (Articulation
        # does not expose a direct body->shape mapping).
        num_shapes_per_body = []
        for link_path in robot.root_physx_view.link_paths[0]:
            link_view = robot._physics_sim_view.create_rigid_body_view(link_path)
            num_shapes_per_body.append(link_view.max_shapes)

        compensated_static = 2 * default_scene_friction - static_friction
        compensated_dynamic = 2 * default_scene_friction - dynamic_friction
        materials = robot.root_physx_view.get_material_properties()
        for body_id in body_ids:
            start_idx = sum(num_shapes_per_body[:body_id])
            end_idx = start_idx + num_shapes_per_body[body_id]
            materials[:, start_idx:end_idx, 0] = compensated_static
            materials[:, start_idx:end_idx, 1] = compensated_dynamic
        robot.root_physx_view.set_material_properties(materials, all_env_ids)
        print(
            f"[startup] compensated finger friction ({body_names}) to static={compensated_static:.3f},"
            f" dynamic={compensated_dynamic:.3f} so gear contact averages back to {default_scene_friction}",
            flush=True,
        )


@configclass
class EventCfg:
    # One-shot: set low friction directly via the PhysX tensor view on all gear
    # bodies (and the gear base) so gear-on-gear contact is slippery, while
    # compensating the Franka finger friction so gear-on-gripper contact is
    # unaffected. See ``set_gear_physics_properties`` for why this bypasses
    # USD material binding (which silently fails on these instanced assets)
    # and for the gear-on-table caveat. Scale per-asset mass relative to its
    # default (USD-baked) value via ``mass_scale``, e.g. {"held_asset": 2.0}
    # doubles the large gear's mass — default below doubles it.
    apply_gear_physics_properties = EventTerm(
        func=set_gear_physics_properties,
        mode="startup",
        params={
            "asset_names": ["held_asset", "gear_small", "gear_medium", "fixed_asset"],
            "static_friction": 0.075,
            "dynamic_friction": 0.075,
            "restitution": 0.1,
            "mass_scale": {"held_asset": 1.0},
        },
    )

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

    # Randomise only the active (large) gear on the table surface. ``pose_range``
    # here describes the gear's GEOMETRIC center (= bore axis), not the root prim
    # — see ``randomize_held_gear_about_center`` for the offset compensation.
    randomize_gear_pose = EventTerm(
        func=randomize_held_gear_about_center,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "pose_range": {
                "x": (0.28, 0.36),
                "y": (-0.275, -0.205),  # negative-Y side, away from gear base cluster
                "z": (GEAR_HALF_HEIGHT, GEAR_HALF_HEIGHT),
                "yaw": (-3.14159, 3.14159),
            },
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
        self.decimation = 8
        # Head-on view with 30° azimuth: camera at 2 m horizontal radius, 1.5 m
        # height, rotated 30° from directly in front of the workspace.
        self.viewer.eye = (2.23, 1.0, 1.5)
        self.viewer.lookat = (0.5, 0.0, 0.3)
        self.sim = SimulationCfg(
            dt=1 / 240,
            physx=PhysxCfg(
                solver_type=1,
                max_position_iteration_count=192,
                max_velocity_iteration_count=2,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.01,
                friction_correlation_distance=0.00625,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**23,
                gpu_max_num_partitions=1,
                # Scene-level continuous collision detection. Adds a sweep pass
                # in the broad-phase that prevents fast-moving thin contacts
                # from tunneling in a single sub-step (e.g. a closing finger
                # passing through a gear tooth). Per-body opt-in is not set
                # here — if visual tunneling persists, add a startup event
                # that sets PhysxRigidBodyAPI:enableCCD on the gears.
                enable_ccd=True,
            ),
        )
        # Re-render a few frames after reset so the cameras settle (matches
        # the visuomotor stack env's approach).
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        # List used by visuomotor consumers (e.g., MimicGen image generation).
        self.image_obs_list = ["wrist_cam", "scene_cam_front", "scene_cam_rear_left"]
