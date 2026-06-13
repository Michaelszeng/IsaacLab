"""
Scripted-expert data collection for the Franka gear assembly task.

Runs a Markovian finite-state machine in which every FSM state is uniquely
determined by the current environment observation (no latched bits, no memory
of the previous step). The FSM has three high-level buckets — ``grasping``,
``transporting``, ``released`` — each with a few leaf sub-states. Each leaf
emits a target EEF pose plus a binary gripper command; the IK-rel controller
turns the (target − current) delta into a smooth motion over multiple steps.

The HDF5 output schema matches ``record_demos.py``.

Usage:
    # Headless, batched (recommended for production).
    python scripts/tools/collect_demos_scripted.py \
        --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
        --dataset_file ./datasets/gear_assembly_scripted.hdf5 \
        --num_envs 32 --num_demos 200 \
        --enable_cameras --headless

    # GUI mode for visual debugging (num_envs forced to 1).
    python scripts/tools/collect_demos_scripted.py \
        --task Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0 \
        --dataset_file ./datasets/gear_assembly_scripted.hdf5 \
        --enable_cameras
"""

import argparse
import os

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI parsing must happen before the AppLauncher boots Isaac Sim.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Scripted-expert data collection for gear assembly.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0",
    help="Isaac Lab task id.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/gear_assembly_scripted.hdf5",
    help="Output HDF5 path. Appends if it already exists (matches record_demos.py behaviour).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=32,
    help="Parallel envs. Forced to 1 when running with the GUI (no --headless).",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Target *total* number of successful demos in the dataset file. 0 = run indefinitely.",
)
parser.add_argument(
    "--max_steps_per_demo",
    type=int,
    default=300,
    help="Episode timeout in env steps. Set lower for faster failure cycles when tuning.",
)
parser.add_argument(
    "--step_hz",
    type=int,
    default=None,
    help="Rate limit. Default: 30 Hz in GUI mode, unlimited when --headless.",
)
parser.add_argument(
    "--n-video-trials",
    type=int,
    default=20,
    help="Number of initial scripted policy attempts to record as video.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force num_envs=1 in GUI mode so the human can actually watch one rollout.
if not args_cli.headless and args_cli.num_envs != 1:
    print(f"[scripted_expert] GUI mode detected — overriding --num_envs {args_cli.num_envs} → 1.")
    args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Remaining imports.
# ---------------------------------------------------------------------------
import math
import time

import cv2
import gymnasium as gym
import h5py
import imageio
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_mimic.envs  # noqa: F401 — registers Isaac-*-Mimic-v0

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# ---------------------------------------------------------------------------
# Geometry pulled from gear_assembly_env_cfg.py.
# ---------------------------------------------------------------------------
# Gear's bore (the hole that drops onto the shaft) sits at this offset in the
# gear's local frame relative to its root prim. Used to align both the grasp
# (gripper centred on the bore) and the insertion (bore over the shaft).
BORE_LOCAL_OFFSET = (-0.045375, 0.0, 0.0)


# =========================================================================== #
# Tunables — edit these to retune the expert. All distances in meters.        #
# =========================================================================== #

# Z heights (env frame). The table surface is roughly z = 0.005; the gear's
# half-height is 0.02 so a resting gear sits with gear_pos.z ≈ 0.02.
Z_PREGRASP = 0.12  # hover over the gear before descending
Z_GRASP = 0.023  # Z position of gripper when grasping the gear
Z_TRANSPORT = 0.15  # lift height after grasp and during retreat
Z_INSERT_DEPTH = 0.06  # fingertip Z at end of insertion
Z_TABLE = 0.02
Z_TABLE_TOL = 0.020  # "gear on table" if gear_pos.z < Z_TABLE + Z_TABLE_TOL

# Position tolerances for advancing to the next FSM sub-state.
XY_TOL = 0.008
XY_TOL_INSERT = 0.002
Z_TOL = 0.006

# Gripper detection + binary commands.
GRIPPER_OPEN_VAL = 0.04  # joint pos at "fully open"
GRIPPER_CLOSED_THRESHOLD = 0.052  # min(fingers) < this  →  closed
GRIPPER_OPEN_CMD = +1.0  # BinaryJointPositionActionCfg convention: >0 → open
GRIPPER_CLOSE_CMD = -1.0  # ≤0 → close

# Per-step delta bounds (action units). The clamp is applied to the *magnitude*
# (L2 norm) of the 3-vector, not per-component, so commanded motion is purely
# along the line from current → target. ``MIN`` boosts very small deltas up to
# at least this magnitude so the robot doesn't crawl when close to the target;
# ``MAX`` scales large deltas down so single-step commands stay feasible for
# the IK-rel controller. Exactly-zero deltas pass through unchanged (so "hold
# pose" states do not jitter).
# The IK-rel controller scales by 0.5 internally — i.e. a commanded magnitude
# of 0.05 corresponds to ≈2.5 cm of actual motion per env step.
MIN_DELTA_POS = 0.02
MAX_DELTA_POS = 0.18
MIN_DELTA_ROT = 0.02
MAX_DELTA_ROT = 0.10

# Top-down ("vertical") grasp orientation in the world frame, as a (w, x, y, z)
# quaternion. (0, 1, 0, 0) is a 180° rotation about world X, which points the
# panda hand's approach axis (body +Z) straight down. Used by the MOVE_TO_GEAR_Z
# branch so the gripper is vertical before descending onto the gear.
GRASP_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)

# Number of teeth on the large (held) gear. Sets the gear's rotational
# symmetry: the gear is indistinguishable under rotations of
# (360° / LARGE_GEAR_NUM_TEETH), so the expert wraps the commanded yaw into
# the nearest equivalent within ±(half tooth period) of the current gear yaw,
# producing the minimum rotation that achieves the desired mesh.
LARGE_GEAR_NUM_TEETH = 60

# Desired large-gear yaw *relative to the medium gear's yaw* during the
# transport→insert phase, in degrees. The expert reads the medium gear's actual
# orientation from the scene and rotates the gripper so that
# ``yaw(large_gear) ≡ yaw(medium_gear) + RELATIVE_YAW_OFFSET_DEG``
# (modulo the gear's tooth-symmetry period, see LARGE_GEAR_NUM_TEETH). Mesh is
# invariant under that symmetry, so the useful tuning range is just
# ±(360° / N_teeth / 2); the rest is symmetry.
RELATIVE_YAW_OFFSET_DEG = 0.7

# Target-pose noise — added to (target_x, target_y, target_z) each step
# before delta computation. Provides trajectory diversity for downstream IL.
NOISE_STD_XY = 0.006
NOISE_STD_Z = 0.005

# Target-rotation noise — a small random rotation applied to ``target_quat``
# each step before delta computation, for states that actively command an
# orientation. Expressed as a per-axis std (radians) on the rotation vector
# (axis-angle), so the perturbation is isotropic in SO(3) for small angles.
NOISE_STD_ROT = 0.025

# Action noise — added directly to the (delta_pos, delta_rot) action
# components after magnitude clamping, just before the action is sent to
# the controller. Unlike the target-pose noise above (which perturbs the
# *goal* and produces correlated multi-step drift toward a shifted target),
# action noise perturbs the *commanded motion* independently each step,
# providing additional state-distribution coverage for downstream IL.
# Expressed as per-axis std on the 3D delta vectors (m for pos, rad for rot
# axis-angle). Keep these small relative to MIN_DELTA_*/MAX_DELTA_* so the
# noisy action stays well within the controller's operating range without
# needing to re-clamp.
ACTION_NOISE_STD_POS = 0.0025
ACTION_NOISE_STD_ROT = 0.02

# =========================================================================== #


INITIAL_FSM_STATE = "MOVE_TO_GEAR_XY"

ALL_FSM_STATES = (
    # "grasping" bucket
    "MOVE_TO_GEAR_XY",
    "MOVE_TO_GEAR_Z",
    "GRASP_GEAR",
    # "transporting" bucket
    "RAISE_GEAR_Z",
    "MOVE_TO_PEG_XY",
    "INSERT",
    "RELEASE",
    # "released" bucket
    "RETREAT_UP",
)


def _bore_world(gear_pos: torch.Tensor, gear_quat: torch.Tensor) -> torch.Tensor:
    """``gear_bore_world = gear_pos + R(gear_quat) @ BORE_LOCAL_OFFSET``. Single env: (3,)."""
    offset = torch.tensor(BORE_LOCAL_OFFSET, dtype=gear_pos.dtype, device=gear_pos.device)
    return gear_pos + math_utils.quat_apply(gear_quat.unsqueeze(0), offset.unsqueeze(0))[0]


def _topdown_quat_with_yaw(yaw: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Top-down panda orientation with an extra rotation of ``yaw`` (rad) about world Z.

    Composition is ``R_z(yaw) · R_x(π)``, which in (w, x, y, z) is
    ``(0, cos(yaw/2), sin(yaw/2), 0)``.
    """
    half = yaw / 2.0
    return torch.tensor([0.0, math.cos(half), math.sin(half), 0.0], dtype=dtype, device=device)


def _yaw_from_quat(quat: torch.Tensor) -> float:
    """Extract the yaw component (rotation about world +Z) from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _clamp_magnitude(vec: torch.Tensor, min_mag: float, max_mag: float) -> torch.Tensor:
    """Clamp the L2 norm of ``vec`` into ``[min_mag, max_mag]``, preserving direction.

    - If the input is essentially zero (norm < 1e-9), it is returned unchanged so
      that "hold pose" states (target == current) don't produce spurious motion.
    - Otherwise the vector is rescaled to ``norm ∈ [min_mag, max_mag]``.
    """
    norm = torch.linalg.vector_norm(vec)
    if float(norm) < 1e-9:
        return vec
    target_norm = torch.clamp(norm, min=min_mag, max=max_mag)
    return vec * (target_norm / norm)


class GearAssemblyScriptedExpert:
    """Markovian FSM scripted expert for the gear-assembly task.

    Maintains one FSM state string per env in ``self.fsm_state``. Both
    ``compute_FSM_state`` and ``compute_action`` operate on a single env at a
    time; the driver loop fans them out across envs.
    """

    def __init__(self, num_envs: int, env=None):
        # The "global" FSM state — one string per env.
        self.fsm_state: list[str] = [INITIAL_FSM_STATE] * num_envs
        # Env reference is used to read the medium-gear quat from the scene
        # (it isn't exposed via the policy obs). Stays optional so the class
        # is still constructible without an env for unit testing.
        self.env = env

    def reset_env(self, env_idx: int) -> None:
        """Reset the FSM for a single env (called when that env terminates)."""
        self.fsm_state[env_idx] = INITIAL_FSM_STATE

    def _mesh_yaw_target_quat(
        self,
        env_idx: int,
        gear_quat: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Top-down quat that meshes the large gear's teeth with the medium gear's.

        Goal::

            yaw(large_gear) ≡ yaw(medium_gear) + RELATIVE_YAW_OFFSET_DEG
                (mod 2π / LARGE_GEAR_NUM_TEETH)

        Because the large gear is rotationally symmetric every tooth period,
        any rotation equivalent to the canonical target produces the same
        mesh. We pick the equivalent that is *closest to the current gear
        yaw* — i.e. wrap the required rotation into ``[-period/2, +period/2)``
        — so the gripper rotates by the minimum amount, never more than a
        half-tooth-period (3° for a 60-tooth gear).

        The gripper holds the gear rigidly, so once we have the minimum gear
        rotation ``Δ``, the gripper rotates by the same ``Δ``:
        ``target_grip_yaw = grip_yaw + Δ``. The result is composed with the
        top-down base orientation, ``R_z(target_grip_yaw) · R_x(π)``.
        """
        if self.env is None:
            # Fallback: ignore the medium gear and just apply the offset as an
            # absolute world-frame yaw. Useful for offline unit tests.
            return _topdown_quat_with_yaw(math.radians(RELATIVE_YAW_OFFSET_DEG), dtype, device)

        medium_gear = self.env.scene["gear_medium"]
        medium_quat = medium_gear.data.root_quat_w[env_idx]

        # All three orientations are read in the world frame; we only care
        # about their yaw component (rotation about world +Z). The gripper
        # quat is fetched on demand from the obs buffer.
        current_gripper_quat = self.env.obs_buf["policy"]["eef_quat"][env_idx]

        medium_yaw = _yaw_from_quat(medium_quat)
        gear_yaw = _yaw_from_quat(gear_quat)
        gripper_yaw = _yaw_from_quat(current_gripper_quat)

        # Canonical (one representative of the equivalence class) target yaw
        # for the large gear.
        canonical_target_gear_yaw = medium_yaw + math.radians(RELATIVE_YAW_OFFSET_DEG)

        # Wrap the required gear rotation into [-period/2, +period/2). Using
        # Python's floor-mod (always non-negative for positive divisor) means
        # the formula below also absorbs the natural 2π wraparound of atan2's
        # output, so adversarial cases like canonical ≈ +π and gear_yaw ≈ −π
        # produce the small in-period delta, not a 2π-sized swing.
        period = 2.0 * math.pi / LARGE_GEAR_NUM_TEETH
        delta = canonical_target_gear_yaw - gear_yaw
        delta_min = ((delta + period / 2.0) % period) - period / 2.0

        target_gripper_yaw = gripper_yaw + delta_min

        return _topdown_quat_with_yaw(target_gripper_yaw, dtype, device)

    def compute_FSM_state(self, obs: dict, env_idx: int) -> str:
        """Inspect the current obs for ``env_idx`` and return the new FSM-state string."""
        pol = obs["policy"]
        eef_pos = pol["eef_pos"][env_idx]  # (3,)
        gear_pos = pol["gear_pos"][env_idx]  # (3,)
        gear_quat = pol["gear_quat"][env_idx]  # (4,)
        shaft_pos = pol["shaft_pos"][env_idx]  # (3,)
        gripper_pos = pol["gripper_pos"][env_idx]  # (2,)
        print(f"    eef_pos: {eef_pos}")
        print(f"    gear_pos: {gear_pos}")
        print(f"    gear_quat: {gear_quat}")
        print(f"    shaft_pos: {shaft_pos}")
        print(f"    gripper_pos: {gripper_pos}")

        bore = _bore_world(gear_pos, gear_quat)

        grasped = math.fabs(gripper_pos[0] - gripper_pos[1]) < GRIPPER_CLOSED_THRESHOLD
        gear_on_table = float(gear_pos[2]) < (Z_TABLE + Z_TABLE_TOL)
        print(f"    grasped: {grasped}")
        print(f"    gear_on_table: {gear_on_table}")

        if gear_on_table and not grasped:
            # "grasping" bucket.
            eef_xy_to_bore = float(torch.linalg.vector_norm(eef_pos[:2] - bore[:2]))
            eef_z_to_grasp_z = float(math.fabs(eef_pos[2] - Z_GRASP))
            if eef_xy_to_bore > XY_TOL:
                return "MOVE_TO_GEAR_XY"
            if eef_z_to_grasp_z > Z_TOL:
                return "MOVE_TO_GEAR_Z"
            return "GRASP_GEAR"

        if grasped:
            # "transporting" bucket.
            bore_xy_to_shaft = float(torch.linalg.vector_norm(bore[:2] - shaft_pos[:2]))
            if float(eef_pos[2]) < Z_TRANSPORT - Z_TOL and bore_xy_to_shaft > XY_TOL_INSERT:
                return "RAISE_GEAR_Z"
            if bore_xy_to_shaft > XY_TOL_INSERT:
                return "MOVE_TO_PEG_XY"
            if float(eef_pos[2]) > Z_INSERT_DEPTH + Z_TOL:
                return "INSERT"
            return "RELEASE"

        # "released" bucket — everything else.
        return "RETREAT_UP"

    def compute_action(self, obs: dict, env_idx: int) -> torch.Tensor:
        """Compute the 7D action for ``env_idx`` given ``self.fsm_state[env_idx]``."""
        state = self.fsm_state[env_idx]
        print(f"STATE: {state}")
        pol = obs["policy"]
        eef_pos = pol["eef_pos"][env_idx]  # (3,)
        gear_pos = pol["gear_pos"][env_idx]
        gear_quat = pol["gear_quat"][env_idx]
        shaft_pos = pol["shaft_pos"][env_idx]
        device, dtype = eef_pos.device, eef_pos.dtype

        bore = _bore_world(gear_pos, gear_quat)

        # Default: hold current pose with gripper open. ``target_quat`` stays
        # ``None`` for states that don't actively command orientation, which
        # produces ``delta_rot = 0`` (= "no rotation change") below.
        target_pos = eef_pos.clone()
        target_quat: torch.Tensor | None = None
        gripper = GRIPPER_OPEN_CMD

        if state == "MOVE_TO_GEAR_XY":
            target_pos[0] = bore[0]
            target_pos[1] = bore[1]
            target_pos[2] = Z_PREGRASP
            # Align the gripper to "vertical" (top-down) before descending so
            # the fingers come down straight onto the gear regardless of any
            # orientation drift accumulated by the IK-rel controller.
            target_quat = torch.tensor(GRASP_QUAT_WXYZ, dtype=dtype, device=device)
            gripper = GRIPPER_OPEN_CMD
        elif state == "MOVE_TO_GEAR_Z":
            target_pos[0] = bore[0]
            target_pos[1] = bore[1]
            target_pos[2] = Z_GRASP
            gripper = GRIPPER_OPEN_CMD
        elif state == "GRASP_GEAR":
            # Hold pose; close gripper.
            gripper = GRIPPER_CLOSE_CMD
        elif state == "RAISE_GEAR_Z":
            target_pos[2] = Z_TRANSPORT
            gripper = GRIPPER_CLOSE_CMD
        elif state == "MOVE_TO_PEG_XY":
            target_pos[0] = eef_pos[0] + (shaft_pos[0] - bore[0])
            target_pos[1] = eef_pos[1] + (shaft_pos[1] - bore[1])
            target_pos[2] = Z_TRANSPORT
            # Yaw the gripper (and therefore the held gear) so the large gear's
            # yaw matches the medium gear's yaw + RELATIVE_YAW_OFFSET_DEG. No
            # jam risk up here at hover height.
            target_quat = self._mesh_yaw_target_quat(env_idx, gear_quat, dtype, device)
            gripper = GRIPPER_CLOSE_CMD
        elif state == "INSERT":
            target_pos[0] = eef_pos[0] + (shaft_pos[0] - bore[0])
            target_pos[1] = eef_pos[1] + (shaft_pos[1] - bore[1])
            target_pos[2] = Z_INSERT_DEPTH
            # Hold the same gear-relative-to-medium yaw while descending.
            target_quat = self._mesh_yaw_target_quat(env_idx, gear_quat, dtype, device)
            gripper = GRIPPER_CLOSE_CMD
        elif state == "RELEASE":
            # Hold pose; open gripper.
            gripper = GRIPPER_OPEN_CMD
        elif state == "RETREAT_UP":
            target_pos[2] = Z_TRANSPORT
            gripper = GRIPPER_OPEN_CMD
        else:
            raise ValueError(f"Unknown FSM state: {state!r}")

        # Small Gaussian noise on the target before computing the delta.
        noise = torch.randn(3, dtype=dtype, device=device) * torch.tensor(
            [NOISE_STD_XY, NOISE_STD_XY, NOISE_STD_Z], dtype=dtype, device=device
        )
        target_pos = target_pos + noise

        delta_pos = _clamp_magnitude(target_pos - eef_pos, MIN_DELTA_POS, MAX_DELTA_POS)

        if target_quat is None:
            # No orientation command this state — keep current EEF rotation.
            delta_rot = torch.zeros(3, dtype=dtype, device=device)
        else:
            # Perturb the target orientation by a small random rotation
            # (axis-angle noise composed in the world frame) for trajectory
            # diversity, mirroring the position noise above.
            rot_noise = torch.randn(3, dtype=dtype, device=device) * NOISE_STD_ROT
            noise_quat = math_utils.quat_from_angle_axis(
                torch.linalg.vector_norm(rot_noise).unsqueeze(0),
                rot_noise.unsqueeze(0),
            )[0]
            target_quat = math_utils.quat_mul(
                noise_quat.unsqueeze(0), target_quat.unsqueeze(0)
            )[0]

            # delta_quat such that ``delta_quat * current_quat = target_quat``
            # (world-frame composition, matches the convention used by the
            # mimic env's target_eef_pose_to_action).
            current_quat = pol["eef_quat"][env_idx]
            delta_quat = math_utils.quat_mul(
                target_quat.unsqueeze(0),
                math_utils.quat_conjugate(current_quat.unsqueeze(0)),
            )[0]
            delta_rot = math_utils.axis_angle_from_quat(delta_quat.unsqueeze(0))[0]
            delta_rot = _clamp_magnitude(delta_rot, MIN_DELTA_ROT, MAX_DELTA_ROT)

        # Per-step action noise on top of the (clamped) deltas. Applied to
        # both components regardless of state — for orientation-holding
        # states this introduces a small random rotation drift, which is
        # self-corrected on entry to the next orientation-commanding state.
        delta_pos = delta_pos + torch.randn(3, dtype=dtype, device=device) * ACTION_NOISE_STD_POS
        delta_rot = delta_rot + torch.randn(3, dtype=dtype, device=device) * ACTION_NOISE_STD_ROT

        gripper_t = torch.tensor([gripper], dtype=dtype, device=device)

        return torch.cat([delta_pos, delta_rot, gripper_t], dim=0)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _make_video_frame(env_obs, env_idx: int, video_cam_keys: list) -> np.ndarray:
    """Build a side-by-side video frame for env_idx by concatenating cameras
    at source resolution. All cameras must share the same H.
    """
    pol = env_obs["policy"] if isinstance(env_obs, dict) and "policy" in env_obs else env_obs
    panes = []
    for k in video_cam_keys:
        if k not in pol:
            continue
        img = pol[k][env_idx].detach().cpu().numpy().astype(np.uint8)
        if img.shape[-1] == 4:
            img = img[..., :3]
        panes.append(img)
    if not panes:
        return np.zeros((240, 320, 3), dtype=np.uint8)
    h = panes[0].shape[0]
    panes = [p if p.shape[0] == h else cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panes]
    return np.concatenate(panes, axis=1)


def _write_mp4(frames, path, fps=10):
    """Write a list of (H, W, 3) uint8 frames to MP4."""
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)


def _count_existing_demos(path: str) -> int:
    """Number of demos already in ``path`` (or 0 if file absent / unreadable)."""
    if not os.path.isfile(path):
        return 0
    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                return 0
            return sum(1 for k in f["data"].keys() if k.startswith("demo_"))
    except Exception as e:
        print(f"[scripted_expert] Could not read existing demo count from '{path}': {e}")
        return 0


def main():
    # --- Output dir ----------------------------------------------------------
    output_dir = os.path.dirname(args_cli.dataset_file) or "."
    output_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # --- Env config ----------------------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task.split(":")[-1]

    # The FSM accesses obs by name, so we need per-key obs (not concatenated).
    env_cfg.observations.policy.concatenate_terms = False
    if hasattr(env_cfg.observations, "subtask_terms"):
        env_cfg.observations.subtask_terms.concatenate_terms = False

    # Map --max_steps_per_demo onto the env's time-out (env steps = sim steps /
    # decimation). This forces stuck rollouts to recycle and not block the
    # batch.
    sim_dt = env_cfg.sim.dt
    decimation = env_cfg.decimation
    env_cfg.episode_length_s = float(args_cli.max_steps_per_demo * decimation * sim_dt)

    # Recorder: write *successful* demos only (env auto-flushes on terminate).
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_filename
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # --- Env + expert --------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device = torch.device(args_cli.device)
    expert = GearAssemblyScriptedExpert(num_envs=env.num_envs, env=env)

    existing_demos = _count_existing_demos(args_cli.dataset_file)
    print(f"[scripted_expert] {existing_demos} existing demos in {args_cli.dataset_file}.")
    if args_cli.num_demos > 0:
        print(f"[scripted_expert] target = {args_cli.num_demos} total demos.")
    print(
        f"[scripted_expert] num_envs={env.num_envs}, "
        f"max_steps_per_demo={args_cli.max_steps_per_demo}, "
        f"headless={args_cli.headless}"
    )

    # --- Rate limit (GUI ~30 Hz so the human can follow; headless: full speed)
    rate_limit = (not args_cli.headless) or (args_cli.step_hz is not None)
    target_hz = args_cli.step_hz if args_cli.step_hz is not None else 30
    last_step_time = time.time()

    # --- Driver loop ---------------------------------------------------------
    obs, _ = env.reset()
    env.recorder_manager.reset()
    last_exported = env.recorder_manager.exported_successful_episode_count
    last_logged_state: list[str | None] = [None] * env.num_envs
    episode_steps = [0] * env.num_envs
    # Running counters for the per-episode success-rate print. ``total_episodes``
    # increments every time an env terminates *or* truncates (i.e. every demo
    # attempt, regardless of outcome). ``total_successes`` counts only the
    # ones that fired the env's success termination.
    total_episodes = 0
    total_successes = 0

    # Video recording setup
    videos_dir = os.path.join(output_dir, "scripted_videos")
    videos_started = 0
    record_env = [False] * env.num_envs
    frame_buffers = [[] for _ in range(env.num_envs)]
    video_cam_keys = []
    if args_cli.n_video_trials > 0:
        os.makedirs(videos_dir, exist_ok=True)
        # Find camera keys in the policy obs dict
        pol = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs
        video_cam_keys = sorted([k for k in pol.keys() if k.startswith("cam_") or k.endswith("_cam")])
        if not video_cam_keys:
            print("[scripted_expert] Warning: --n-video-trials > 0 but no camera obs keys detected.")
        
        for i in range(env.num_envs):
            if videos_started < args_cli.n_video_trials:
                record_env[i] = True
                videos_started += 1
                frame_buffers[i].append(_make_video_frame(obs, i, video_cam_keys))

    while simulation_app.is_running():
        with torch.inference_mode():
            # 1) Update FSM state per env from the current obs.
            for i in range(env.num_envs):
                expert.fsm_state[i] = expert.compute_FSM_state(obs, i)

            # 2) Compute per-env action, then stack into a (num_envs, 7) tensor.
            # The IK-rel arm action is 6D + 1D gripper = 7D total.
            actions = torch.stack([expert.compute_action(obs, i) for i in range(env.num_envs)], dim=0)

        # GUI-mode FSM trace: print only on state transitions to keep it readable.
        if not args_cli.headless and env.num_envs == 1:
            if expert.fsm_state[0] != last_logged_state[0]:
                print(f"[fsm] -> {expert.fsm_state[0]}")
                last_logged_state[0] = expert.fsm_state[0]

        obs, _, terminated, truncated, _ = env.step(actions)

        if args_cli.n_video_trials > 0:
            for i in range(env.num_envs):
                if record_env[i]:
                    frame_buffers[i].append(_make_video_frame(obs, i, video_cam_keys))

        # Count the step that just occurred for every env.
        for i in range(env.num_envs):
            episode_steps[i] += 1

        # Track recorder progress.
        current_exported = env.recorder_manager.exported_successful_episode_count
        if current_exported > last_exported:
            new = current_exported - last_exported
            last_exported = current_exported
            total = existing_demos + current_exported
            tgt_str = f"/{args_cli.num_demos}" if args_cli.num_demos > 0 else ""
            print(f"[scripted_expert] +{new} successful demo(s)  →  total: {total}{tgt_str}")
            if args_cli.num_demos > 0 and total >= args_cli.num_demos:
                print("[scripted_expert] target reached, exiting.")
                break

        # Per-env reset bookkeeping: the env auto-resets envs whose terminated /
        # truncated fired, so we just reset the FSM string for those envs.
        done = (terminated | truncated).cpu()
        for i in range(env.num_envs):
            if bool(done[i]):
                is_success = bool(terminated[i])
                outcome = "SUCCESS" if is_success else "TIMEOUT"
                total_episodes += 1
                if is_success:
                    total_successes += 1
                rate = total_successes / total_episodes
                print(
                    f"[scripted_expert] env {i} episode ended ({outcome}) after"
                    f" {episode_steps[i]} steps. Success rate so far:"
                    f" {total_successes}/{total_episodes} ({rate:.1%})."
                )
                
                if args_cli.n_video_trials > 0 and record_env[i]:
                    video_path = os.path.join(videos_dir, f"trial_{total_episodes:04d}_{outcome}.mp4")
                    _write_mp4(frame_buffers[i], video_path, fps=10)
                    print(f"[scripted_expert] Saved video: {video_path}")
                    frame_buffers[i] = []
                    if videos_started < args_cli.n_video_trials:
                        record_env[i] = True
                        videos_started += 1
                        frame_buffers[i].append(_make_video_frame(obs, i, video_cam_keys))
                    else:
                        record_env[i] = False

                episode_steps[i] = 0
                expert.reset_env(i)
                last_logged_state[i] = None

        if rate_limit:
            elapsed = time.time() - last_step_time
            sleep = max(0.0, 1.0 / target_hz - elapsed)
            if sleep > 0.0:
                time.sleep(sleep)
            last_step_time = time.time()

    final_total = existing_demos + env.recorder_manager.exported_successful_episode_count
    print(f"[scripted_expert] done. Final demo count in {args_cli.dataset_file}: {final_total}.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
