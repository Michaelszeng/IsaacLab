# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

NOTE: this script appends to the output file rather than overwriting it.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Target *total* number of successful demonstrations in the dataset
                              file (existing demos in the file count toward this target since the
                              script appends). Set to 0 to record indefinitely. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful.
                              (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help=(
        "Target *total* number of successful demonstrations in the dataset file. Because this script"
        " appends to --dataset_file, demos already present in the file count toward this target;"
        " only (num_demos - existing) new demos will be recorded this run. Set to 0 for infinite."
    ),
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Validate required arguments
if args_cli.task is None:
    parser.error("--task is required")

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version
    # installed by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


# Third-party imports
import logging
import os
import time

import gymnasium as gym
import torch

import omni.kit.app
import omni.ui as ui

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from collections.abc import Callable

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# import logger
logger = logging.getLogger(__name__)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def count_existing_demos(dataset_file: str) -> int:
    """Return the number of demos already stored in ``dataset_file``.

    The dataset file is expected to follow the HDF5 layout written by
    :class:`~isaaclab.utils.datasets.HDF5DatasetFileHandler`, where each
    successful demo lives under ``/data/demo_<idx>``. Returns 0 if the file
    doesn't exist, is unreadable, or has no demos.
    """
    if not os.path.isfile(dataset_file):
        return 0
    try:
        import h5py

        with h5py.File(dataset_file, "r") as f:
            if "data" not in f:
                return 0
            data_group = f["data"]
            if not isinstance(data_group, h5py.Group):
                return 0
            return sum(1 for k in data_group.keys() if k.startswith("demo_"))
    except Exception as e:
        logger.warning(f"Could not read existing demo count from '{dataset_file}': {e}")
        return 0


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object]]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    if args_cli.xr:
        # If cameras are not enabled and XR is enabled, remove camera configs
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)


def setup_teleop_device(callbacks: dict[str, Callable]) -> object:
    """Set up the teleoperation device based on configuration.

    Attempts to create a teleoperation device based on the environment configuration.
    Falls back to default devices if the specified device is not found in the configuration.

    Args:
        callbacks: Dictionary mapping callback keys to functions that will be
                   attached to the teleop device

    Returns:
        object: The configured teleoperation device interface

    Raises:
        Exception: If teleop device creation fails
    """
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, handtracking")
                exit(1)

            # Add callbacks to fallback device
            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        exit(1)

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        exit(1)

    return teleop_interface


def setup_wrist_camera_window(env: gym.Env, wrist_prim: str = "/World/envs/env_0/Robot/panda_hand") -> None:
    """Create a floating viewport window that follows the robot's wrist camera.

    The window renders a second viewport locked to a camera placed at the wrist.
    If the robot prim doesn't exist yet or the viewport API is unavailable the
    function exits silently so it never blocks the main loop.

    Args:
        env: The environment instance (used only to check sim is running).
        wrist_prim: USD prim path for the wrist link that the camera will track.
    """
    try:
        import omni.kit.viewport.utility as vp_utils
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return

        cam_path = wrist_prim + "/WristCam"

        # Create the camera prim if it doesn't exist.
        if not stage.GetPrimAtPath(cam_path).IsValid():
            cam_prim = UsdGeom.Camera.Define(stage, cam_path)
            cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10.0))
            cam_prim.GetHorizontalApertureAttr().Set(20.955)
            cam_prim.GetFocalLengthAttr().Set(24.0)
            # Orient so the camera looks forward (down the wrist Z-axis).
            xform = UsdGeom.Xformable(cam_prim.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.12))
            # 180° around Y: correct forward direction.  +90° around Z: 90° CW image.
            xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 180.0, 90.0))

        # Open a new viewport window pointing at the wrist camera.
        viewport = vp_utils.create_viewport_window("Wrist Camera", width=400, height=300)
        if viewport is not None:
            viewport.viewport_api.set_active_camera(cam_path)
    except Exception as e:
        print(f"[WristCam] Could not create wrist camera window: {e}")


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

        setup_wrist_camera_window(env)

    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(
    env: gym.Env,
    success_step_count: int,
    instruction_display: InstructionDisplay,
    label_text: str,
    teleop_interface: object | None = None,
) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, and related state variables.
    Updates the instruction display with current status.  Also resets the
    teleop device's internal state so that, e.g., the spacemouse's persistent
    gripper-toggle flag doesn't immediately re-close the gripper.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status
        teleop_interface: Teleop device whose internal state should be cleared
            so the gripper command returns to "open" after reset.

    Returns:
        int: Reset success step count (0)
    """
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    if teleop_interface is not None and hasattr(teleop_interface, "reset"):
        teleop_interface.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
    existing_demo_count: int = 0,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        teleop_interface: Optional teleop interface (will be created if None)
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed
        existing_demo_count: Number of successful demos already present in the
            dataset file at the start of this run. Used so that ``--num_demos``
            is interpreted as a *total* target (existing + new).

    Returns:
        int: Number of successful demonstrations recorded in this run.
    """
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = not args_cli.xr

    # ---------------------------------------------------------------------------
    # Pre-set camera views.  Press V to cycle through them.
    # Each entry is (eye_xyz, lookat_xyz).
    # ---------------------------------------------------------------------------
    _PRESET_VIEWS = [
        # 1: front close-up — near-head-on, slightly above workspace
        ((1.429, -0.082, 0.76), (0.158, -0.091, 0.206)),
        # 2: low rear-left — looking forward/up toward the workspace
        ((0.232, -0.279, 0.001), (1.367, 0.769, 0.466)),
    ]
    _current_view_idx = [0]  # mutable container so the closure can update it

    def cycle_camera_view():
        _current_view_idx[0] = (_current_view_idx[0] + 1) % len(_PRESET_VIEWS)
        eye, lookat = _PRESET_VIEWS[_current_view_idx[0]]
        env.sim.set_camera_view(eye, lookat)
        print(f"[View] preset {_current_view_idx[0]}: eye={eye}", flush=True)

    def print_camera_view():
        """Print the current viewport camera eye/lookat so it can be saved as a preset.

        Drag/orbit the viewport with the mouse to position the camera, then press
        ``P`` to print a tuple ready to paste into ``_PRESET_VIEWS`` above.
        """
        try:
            from omni.kit.viewport.utility import get_active_viewport
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
        except ImportError as e:
            print(f"[View] viewport utility unavailable: {e}", flush=True)
            return

        viewport_api = get_active_viewport()
        if viewport_api is None:
            print("[View] No active viewport available.", flush=True)
            return

        try:
            cam_state = ViewportCameraState(viewport_api.camera_path, viewport_api)
            eye_vec = cam_state.position_world
            target_vec = cam_state.target_world
        except Exception as e:
            print(f"[View] Could not query camera state: {e}", flush=True)
            return

        eye = (round(float(eye_vec[0]), 3), round(float(eye_vec[1]), 3), round(float(eye_vec[2]), 3))
        lookat = (round(float(target_vec[0]), 3), round(float(target_vec[1]), 3), round(float(target_vec[2]), 3))
        print(
            f"[View] current camera -> eye={eye}, lookat={lookat}\n       preset entry: ({eye}, {lookat}),",
            flush=True,
        )

    def mark_success_and_advance():
        """Mark the current episode as successful, export it, then reset."""
        nonlocal should_reset_recording_instance
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
        env.recorder_manager.export_episodes([0])
        print("Demo manually marked as SUCCESS — saving and advancing.", flush=True)
        should_reset_recording_instance = True

    # Callback closures for the teleop device
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")

    # Set up teleoperation callbacks
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    teleop_interface = setup_teleop_device(teleoperation_callbacks)
    teleop_interface.add_callback("R", reset_recording_instance)

    # Keyboard listener for 'r' → env reset (works alongside spacemouse).
    keyboard_interface = Se3Keyboard(Se3KeyboardCfg())
    keyboard_interface.add_callback("R", reset_recording_instance)
    keyboard_interface.add_callback("V", cycle_camera_view)
    keyboard_interface.add_callback("P", print_camera_view)
    keyboard_interface.add_callback("N", mark_success_and_advance)

    print(
        "\n"
        "─" * 52 + "\n"
        " Keyboard shortcuts\n"
        "─" * 52 + "\n"
        "  R        Reset / discard current episode\n"
        "  N        Mark current episode as SUCCESS and save\n"
        "  V        Cycle through preset camera views\n"
        "  P        Print current camera position (for presets)\n"
        "  Ctrl+C   Quit\n"
        "─" * 52 + "\n"
    )

    # Reset before starting
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    def _format_label(session_count: int) -> str:
        total = existing_demo_count + session_count
        if args_cli.num_demos > 0:
            return (
                f"Recorded {total}/{args_cli.num_demos} total demonstrations"
                f" ({session_count} this session, {existing_demo_count} pre-existing)."
            )
        return (
            f"Recorded {total} total demonstrations"
            f" ({session_count} this session, {existing_demo_count} pre-existing)."
        )

    label_text = _format_label(current_recorded_demo_count)
    instruction_display = setup_ui(label_text, env)

    subtasks = {}

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Get keyboard command
            action = teleop_interface.advance()
            # Expand to batch dimension
            actions = action.repeat(env.num_envs, 1)

            # Perform action on environment
            if running_recording_instance:
                # Compute actions based on environment
                obv = env.step(actions)
                if subtasks is not None:
                    if subtasks == {}:
                        subtasks = obv[0].get("subtask_terms")
                    elif subtasks:
                        show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
            else:
                env.sim.render()

            # Check for success condition
            success_step_count, success_reset_needed = process_success_condition(env, success_term, success_step_count)
            if success_reset_needed:
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = _format_label(current_recorded_demo_count)
                print(label_text)

            # Check if we've reached the desired total number of demos
            # (pre-existing demos in the file count toward --num_demos).
            total_demo_count = existing_demo_count + env.recorder_manager.exported_successful_episode_count
            if args_cli.num_demos > 0 and total_demo_count >= args_cli.num_demos:
                label_text = f"All {total_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Handle reset if requested
            if should_reset_recording_instance:
                success_step_count = handle_reset(
                    env, success_step_count, instruction_display, label_text, teleop_interface
                )
                should_reset_recording_instance = False

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        # Assign the teleop visualization manager to the visualization system
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # --num_demos is interpreted as the desired *total* number of demos in
    # the dataset file. Count what's already there so we record only the
    # remainder this session (and exit immediately if the target is met).
    existing_demo_count = count_existing_demos(args_cli.dataset_file)
    if existing_demo_count > 0:
        print(f"Dataset '{args_cli.dataset_file}' already contains {existing_demo_count} demos.")
    if args_cli.num_demos > 0 and existing_demo_count >= args_cli.num_demos:
        print(
            f"Target of {args_cli.num_demos} total demonstrations already reached"
            f" ({existing_demo_count} present). Nothing to record."
        )
        return

    # Create and configure environment
    global env_cfg  # Make env_cfg available to setup_teleop_device
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    # Create environment
    env = create_environment(env_cfg)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(
        env, None, success_term, rate_limiter, existing_demo_count=existing_demo_count
    )

    # Clean up
    env.close()
    total_demo_count = existing_demo_count + current_recorded_demo_count
    print(
        f"Recording session completed with {current_recorded_demo_count} new successful demonstrations"
        f" ({total_demo_count} total in dataset)."
    )
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
