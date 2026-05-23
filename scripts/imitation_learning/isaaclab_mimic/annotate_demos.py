# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to add mimic annotations to demos to be used as source demos for mimic dataset generation.
"""

import argparse
import math

from isaaclab.app import AppLauncher

# Launching Isaac Sim Simulator first.


# add argparse arguments
parser = argparse.ArgumentParser(description="Annotate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--input_file", type=str, default="./datasets/dataset.hdf5", help="File name of the dataset to be annotated."
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/dataset_annotated.hdf5",
    help="File name of the annotated output dataset file.",
)
parser.add_argument("--auto", action="store_true", default=False, help="Automatically annotate subtasks.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--annotate_subtask_start_signals",
    action="store_true",
    default=False,
    help="Enable annotating start points of subtasks.",
)
parser.add_argument(
    "--grasp_action_bounds",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN_PCT", "MAX_PCT"),
    help=(
        "Optional inclusive bounds on the action index at which the grasp annotation may be placed,"
        " expressed as percentages (0-100) of the episode's action count. Specify two floats"
        " MIN_PCT MAX_PCT. In --auto mode, only the slice of the recorded grasp signal lying within"
        " [MIN_PCT, MAX_PCT] is consulted: values outside the window are ignored, and the grasp"
        " annotation is placed at the first False→True transition observed inside the window."
        " This rejects both later false positives (e.g. during insertion when the object shifts in"
        " the gripper) and earlier false positives that left the signal stuck True at the start of"
        " the window (since no in-window transition is observable). In manual mode, episodes whose"
        " manually marked grasp action index lies outside these bounds are rejected. In both modes,"
        " episodes with no valid in-window grasp are not exported."
    ),
)
parser.add_argument(
    "--grasp_subtask_signal",
    type=str,
    default=None,
    help=(
        "Name of the subtask termination signal that identifies the grasp event for bounds checking."
        " If omitted, the first subtask termination signal of the first end-effector is used."
        " Only relevant when --grasp_action_bounds is provided."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed
    # by IsaacLab and not the one installed by Isaac Sim.
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import gymnasium as gym
import torch

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

# Only enables inputs if this script is NOT headless mode
if not args_cli.headless and not os.environ.get("HEADLESS", 0):
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False
current_action_index = 0
marked_subtask_action_indices = []
skip_episode = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def skip_episode_cb():
    global skip_episode
    skip_episode = True


def mark_subtask_cb():
    global current_action_index, marked_subtask_action_indices
    marked_subtask_action_indices.append(current_action_index)
    print(f"Marked a subtask signal at action index: {current_action_index}")


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskStartsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask start observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_start_signals", self._env.get_subtask_start_signals()


@configclass
class PreStepSubtaskStartsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask start observations recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskStartsObservationsRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic specific recorder terms."""

    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_start_signals = PreStepSubtaskStartsObservationsRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def _flatten_signal_to_bool_tensor(signal_value) -> torch.Tensor | None:
    """Flatten a recorded subtask signal into a 1D boolean tensor.

    The annotated episode stores subtask signals either as a list of per-step tensors (auto mode, one
    entry per env-step) or as a list containing a single full-length tensor (manual mode). This helper
    normalizes both representations into a 1D bool tensor of length ``num_steps``.
    """
    if signal_value is None:
        return None
    if isinstance(signal_value, list):
        if len(signal_value) == 0:
            return None
        if len(signal_value) == 1 and signal_value[0].dim() == 1:
            tensor = signal_value[0]
        else:
            tensor = torch.cat([t.reshape(-1) for t in signal_value])
    else:
        tensor = signal_value.reshape(-1)
    return tensor.to(torch.bool)


def _find_grasp_action_index(annotated_episode: EpisodeData, signal_name: str) -> int | None:
    """Find the first action index at which the named subtask termination signal becomes True."""
    subtask_term_signals = annotated_episode.data.get("obs", {}).get("datagen_info", {}).get("subtask_term_signals", {})
    if signal_name not in subtask_term_signals:
        return None
    flags = _flatten_signal_to_bool_tensor(subtask_term_signals[signal_name])
    if flags is None or flags.numel() == 0:
        return None
    nonzero = flags.nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        return None
    return int(nonzero[0].item())


def _find_grasp_action_index_in_range(
    annotated_episode: EpisodeData,
    signal_name: str,
    min_index: int,
    max_index: int,
) -> int | None:
    """Find the action index of the first False→True transition of the grasp signal inside ``[min_index, max_index]``.

    Only the flags within ``[min_index, max_index]`` are inspected; values outside the window do not
    influence the search at all. Specifically, the function returns the smallest ``k`` such that
    ``min_index < k <= max_index`` and ``flags[k - 1] == False`` and ``flags[k] == True``, with both
    ``k`` and ``k - 1`` lying within the window.

    Returning a transition (rather than the earliest True in the window) rules out the "stuck-True"
    case where the signal flipped earlier in the episode (e.g. an early false positive) and continues
    to read True throughout the window; in that case no in-window F→T edge exists and the function
    returns ``None`` so the caller rejects the episode.
    """
    subtask_term_signals = annotated_episode.data.get("obs", {}).get("datagen_info", {}).get("subtask_term_signals", {})
    if signal_name not in subtask_term_signals:
        return None
    flags = _flatten_signal_to_bool_tensor(subtask_term_signals[signal_name])
    if flags is None or flags.numel() == 0:
        return None
    lo = max(0, min_index)
    hi = min(flags.numel() - 1, max_index)
    if lo >= hi:
        # Need at least two samples inside the window to observe a transition.
        return None
    window = flags[lo : hi + 1]
    # Edges = window[1:] AND NOT window[:-1].  An edge at slice index j (j >= 1) corresponds to a
    # transition into True at global index lo + j, observed strictly inside the window.
    edges = window[1:] & ~window[:-1]
    nonzero = edges.nonzero(as_tuple=False)
    if nonzero.numel() == 0:
        return None
    return int(nonzero[0].item()) + 1 + lo


def _overwrite_grasp_signal(annotated_episode: EpisodeData, signal_name: str, grasp_action_index: int) -> None:
    """Rewrite the recorded grasp signal to a clean monotonic step at ``grasp_action_index``.

    For each recorded step ``k``, the signal is set to ``True`` iff ``k >= grasp_action_index`` and
    ``False`` otherwise. The original tensor shape, dtype, and device are preserved so downstream
    export logic (which stacks the per-step tensors) is unaffected.
    """
    subtask_term_signals = annotated_episode.data.get("obs", {}).get("datagen_info", {}).get("subtask_term_signals", {})
    if signal_name not in subtask_term_signals:
        return
    signal_value = subtask_term_signals[signal_name]
    if isinstance(signal_value, list):
        if len(signal_value) == 1 and signal_value[0].dim() == 1:
            # Manual-mode representation: a single full-length tensor.
            tensor = signal_value[0]
            tensor[:grasp_action_index] = False
            tensor[grasp_action_index:] = True
        else:
            # Auto-mode representation: one tensor per step.
            for k in range(len(signal_value)):
                signal_value[k].fill_(bool(k >= grasp_action_index))
    else:
        # Fallback: an unstacked tensor representation.
        signal_value[:grasp_action_index] = False
        signal_value[grasp_action_index:] = True


def main():
    """Add Isaac Lab Mimic annotations to the given demo dataset file."""
    global is_paused, current_action_index, marked_subtask_action_indices

    # Load input dataset to be annotated
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return 0

    # get output directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)

    env_cfg.env_name = env_name

    # extract success checking function to invoke manually
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # Disable all termination terms
    env_cfg.terminations = None

    # Set up recorder terms for mimic annotations
    env_cfg.recorders = MimicRecorderManagerCfg()
    if not args_cli.auto:
        # disable subtask term signals recorder term if in manual mode
        env_cfg.recorders.record_pre_step_subtask_term_signals = None

    if not args_cli.auto or (args_cli.auto and not args_cli.annotate_subtask_start_signals):
        # disable subtask start signals recorder term if in manual mode or no need for subtask start annotations
        env_cfg.recorders.record_pre_step_subtask_start_signals = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment from loaded config
    env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if args_cli.auto:
        # check if the mimic API env.get_subtask_term_signals() is implemented
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            raise NotImplementedError(
                "The environment does not implement the get_subtask_term_signals method required "
                "to run automatic annotations."
            )
        if (
            args_cli.annotate_subtask_start_signals
            and env.get_subtask_start_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_start_signals
        ):
            raise NotImplementedError(
                "The environment does not implement the get_subtask_start_signals method required "
                "to run automatic annotations."
            )
    else:
        # get subtask termination signal names for each eef from the environment configs
        subtask_term_signal_names = {}
        subtask_start_signal_names = {}
        for eef_name, eef_subtask_configs in env.cfg.subtask_configs.items():
            subtask_start_signal_names[eef_name] = (
                [subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs]
                if args_cli.annotate_subtask_start_signals
                else []
            )
            subtask_term_signal_names[eef_name] = [
                subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs
            ]
            # Validation: if annotating start signals, every subtask (including the last) must have a name
            if args_cli.annotate_subtask_start_signals:
                if any(name in (None, "") for name in subtask_start_signal_names[eef_name]):
                    raise ValueError(
                        f"Missing 'subtask_term_signal' for one or more subtasks in eef '{eef_name}'. When"
                        " '--annotate_subtask_start_signals' is enabled, each subtask (including the last) must"
                        " specify 'subtask_term_signal'. The last subtask's term signal name is used as the final"
                        " start signal name."
                    )
            # no need to annotate the last subtask term signal, so remove it from the list
            subtask_term_signal_names[eef_name].pop()

    # reset environment
    env.reset()

    # Resolve and validate optional grasp action-index bounds configuration (in percent).
    grasp_action_bounds_pct: tuple[float, float] | None = None
    grasp_signal_name: str | None = None
    if args_cli.grasp_action_bounds is not None:
        min_action_pct, max_action_pct = args_cli.grasp_action_bounds
        if not (0.0 <= min_action_pct <= 100.0) or not (0.0 <= max_action_pct <= 100.0):
            raise ValueError("--grasp_action_bounds percentages must lie within [0, 100].")
        if min_action_pct > max_action_pct:
            raise ValueError("--grasp_action_bounds must be specified as MIN_PCT MAX_PCT with MIN_PCT <= MAX_PCT.")
        grasp_action_bounds_pct = (min_action_pct, max_action_pct)

        # Collect all subtask termination signal names across all eefs to validate the chosen one.
        all_signal_names: list[str] = []
        for eef_subtask_configs in env.cfg.subtask_configs.values():
            for subtask_config in eef_subtask_configs:
                if subtask_config.subtask_term_signal not in (None, ""):
                    all_signal_names.append(subtask_config.subtask_term_signal)
        if len(all_signal_names) == 0:
            raise ValueError(
                "--grasp_action_bounds was provided but the environment defines no subtask termination signals."
            )
        grasp_signal_name = (
            args_cli.grasp_subtask_signal if args_cli.grasp_subtask_signal is not None else all_signal_names[0]
        )
        if grasp_signal_name in (None, "") or grasp_signal_name not in all_signal_names:
            raise ValueError(
                f"--grasp_subtask_signal '{grasp_signal_name}' not found among the environment's"
                f" subtask termination signals. Available signals: {all_signal_names}"
            )

        print(
            "Grasp action-index bounds enabled:\n"
            f"\t- signal:          {grasp_signal_name}\n"
            f"\t- action range %:  [{min_action_pct:g}%, {max_action_pct:g}%]"
        )

    # Only enables inputs if this script is NOT headless mode
    if not args_cli.headless and not os.environ.get("HEADLESS", 0):
        keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        keyboard_interface.add_callback("N", play_cb)
        keyboard_interface.add_callback("B", pause_cb)
        keyboard_interface.add_callback("Q", skip_episode_cb)
        if not args_cli.auto:
            keyboard_interface.add_callback("S", mark_subtask_cb)
        keyboard_interface.reset()

    # simulate environment -- run everything in inference mode
    exported_episode_count = 0
    processed_episode_count = 0
    successful_task_count = 0  # Counter for successful task completions
    grasp_bounds_rejection_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # Iterate over the episodes in the loaded dataset file
            for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
                processed_episode_count += 1
                print(f"\nAnnotating episode #{episode_index} ({episode_name})")
                episode = dataset_file_handler.load_episode(episode_name, env.device)

                is_episode_annotated_successfully = False
                if args_cli.auto:
                    is_episode_annotated_successfully = annotate_episode_in_auto_mode(env, episode, success_term)
                else:
                    is_episode_annotated_successfully = annotate_episode_in_manual_mode(
                        env, episode, success_term, subtask_term_signal_names, subtask_start_signal_names
                    )

                # Apply optional grasp action-index bounds filtering before exporting.
                if is_episode_annotated_successfully and not skip_episode and grasp_action_bounds_pct is not None:
                    assert grasp_signal_name is not None
                    annotated_episode = env.recorder_manager.get_episode(0)
                    min_action_pct, max_action_pct = grasp_action_bounds_pct
                    # Convert the percentage window into an inclusive action-index window using the
                    # length of the source episode's action sequence.
                    episode_length = len(episode.data["actions"])
                    if episode_length <= 0:
                        last_index = 0
                    else:
                        last_index = episode_length - 1
                    min_action_index = int(round(min_action_pct / 100.0 * last_index))
                    max_action_index = int(round(max_action_pct / 100.0 * last_index))
                    min_action_index = max(0, min(min_action_index, last_index))
                    max_action_index = max(0, min(max_action_index, last_index))

                    if args_cli.auto:
                        # Search for the earliest signal-True step whose action index lies inside the
                        # configured window, then rewrite the recorded signal to a clean monotonic
                        # step at that index. This snaps the grasp annotation onto the actual grasp
                        # moment instead of any later false-positive (e.g. during the insertion phase).
                        grasp_action_index = _find_grasp_action_index_in_range(
                            annotated_episode,
                            grasp_signal_name,
                            min_action_index,
                            max_action_index,
                        )
                        if grasp_action_index is None:
                            is_episode_annotated_successfully = False
                            grasp_bounds_rejection_count += 1
                            print(
                                f"\tNo '{grasp_signal_name}' False→True transition found within"
                                f" [{min_action_pct:g}%, {max_action_pct:g}%] (action indices"
                                f" [{min_action_index}, {max_action_index}] of"
                                f" {episode_length}); rejecting the episode."
                            )
                        else:
                            _overwrite_grasp_signal(annotated_episode, grasp_signal_name, grasp_action_index)
                            grasp_action_pct = 100.0 * grasp_action_index / last_index if last_index > 0 else 0.0
                            print(
                                f"\tGrasp '{grasp_signal_name}' annotated at action index"
                                f" {grasp_action_index} ({grasp_action_pct:.1f}%) within"
                                f" [{min_action_pct:g}%, {max_action_pct:g}%]."
                            )
                    else:
                        # Manual mode: the user explicitly marked the grasp, so just validate that the
                        # marked action index lies within the configured window.
                        grasp_action_index = _find_grasp_action_index(annotated_episode, grasp_signal_name)
                        if grasp_action_index is None:
                            is_episode_annotated_successfully = False
                            grasp_bounds_rejection_count += 1
                            print(
                                f"\tCould not locate a marked grasp for signal"
                                f" '{grasp_signal_name}'; rejecting due to --grasp_action_bounds."
                            )
                        elif not (min_action_index <= grasp_action_index <= max_action_index):
                            is_episode_annotated_successfully = False
                            grasp_bounds_rejection_count += 1
                            grasp_action_pct = 100.0 * grasp_action_index / last_index if last_index > 0 else 0.0
                            print(
                                f"\tMarked grasp at action index {grasp_action_index}"
                                f" ({grasp_action_pct:.1f}%) is outside"
                                f" [{min_action_pct:g}%, {max_action_pct:g}%];"
                                " rejecting the episode."
                            )
                        else:
                            grasp_action_pct = 100.0 * grasp_action_index / last_index if last_index > 0 else 0.0
                            print(
                                f"\tMarked grasp at action index {grasp_action_index}"
                                f" ({grasp_action_pct:.1f}%) is within"
                                f" [{min_action_pct:g}%, {max_action_pct:g}%]."
                            )

                if is_episode_annotated_successfully and not skip_episode:
                    # set success to the recorded episode data and export to file
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                    exported_episode_count += 1
                    successful_task_count += 1  # Increment successful task counter
                    print("\tExported the annotated episode.")
                else:
                    print("\tSkipped exporting the episode.")
            break

    print(
        f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated"
        f" episode{'s' if exported_episode_count > 1 else ''}."
    )
    if grasp_action_bounds_pct is not None:
        print(
            f"Episodes rejected by --grasp_action_bounds: {grasp_bounds_rejection_count}"
            f" (signal='{grasp_signal_name}',"
            f" range=[{grasp_action_bounds_pct[0]:g}%, {grasp_action_bounds_pct[1]:g}%])."
        )
    print(
        f"Successful task completions: {successful_task_count}"
    )  # This line is used by the dataset generation test case to check if the expected number of demos were annotated
    print("Exiting the app.")

    # Close environment after annotation is complete
    env.close()

    return successful_task_count


def replay_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Replays an episode in the environment.

    This function replays the given recorded episode in the environment. It can optionally check if the task
    was successfully completed using a success termination condition input.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully replayed and the success condition was met (if provided),
        False otherwise.
    """
    global current_action_index, skip_episode, is_paused
    # read initial state and actions from the loaded episode
    initial_state = episode.data["initial_state"]
    actions = episode.data["actions"]
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset_to(initial_state, None, is_relative=True)
    first_action = True
    for action_index, action in enumerate(actions):
        current_action_index = action_index
        if first_action:
            first_action = False
        else:
            while is_paused or skip_episode:
                env.sim.render()
                if skip_episode:
                    return False
                continue
        action_tensor = torch.Tensor(action).reshape([1, action.shape[0]])
        env.step(torch.Tensor(action_tensor))
    if success_term is not None:
        if not bool(success_term.func(env, **success_term.params)[0]):
            return False
    return True


def annotate_episode_in_auto_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Annotates an episode in automatic mode.

    This function replays the given episode in the environment and checks if the task was successfully completed.
    If the task was not completed, it will print a message and return False. Otherwise, it will check if all the
    subtask term signals are annotated and return True if they are, False otherwise.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global skip_episode
    skip_episode = False
    is_episode_annotated_successfully = replay_episode(env, episode, success_term)
    if skip_episode:
        print("\tSkipping the episode.")
        return False
    if not is_episode_annotated_successfully:
        print("\tThe final task was not completed.")
    else:
        # check if all the subtask term signals are annotated
        annotated_episode = env.recorder_manager.get_episode(0)
        subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
        for signal_name, signal_flags in subtask_term_signal_dict.items():
            signal_flags = torch.tensor(signal_flags, device=env.device)
            if not torch.any(signal_flags):
                is_episode_annotated_successfully = False
                print(f'\tDid not detect completion for the subtask "{signal_name}".')
        if args_cli.annotate_subtask_start_signals:
            subtask_start_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_start_signals"]
            for signal_name, signal_flags in subtask_start_signal_dict.items():
                if not torch.any(signal_flags):
                    is_episode_annotated_successfully = False
                    print(f'\tDid not detect start for the subtask "{signal_name}".')
    return is_episode_annotated_successfully


def annotate_episode_in_manual_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    subtask_term_signal_names: dict[str, list[str]] = {},
    subtask_start_signal_names: dict[str, list[str]] = {},
) -> bool:
    """Annotates an episode in manual mode.

    This function replays the given episode in the environment and allows for manual marking of subtask term signals.
    It iterates over each eef and prompts the user to mark the subtask term signals for that eef.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.
        subtask_term_signal_names: Dictionary mapping eef names to lists of subtask term signal names.
        subtask_start_signal_names: Dictionary mapping eef names to lists of subtask start signal names.
    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global is_paused, marked_subtask_action_indices, skip_episode
    # iterate over the eefs for marking subtask term signals
    subtask_term_signal_action_indices = {}
    subtask_start_signal_action_indices = {}
    for eef_name, eef_subtask_term_signal_names in subtask_term_signal_names.items():
        eef_subtask_start_signal_names = subtask_start_signal_names[eef_name]
        # skip if no subtask annotation is needed for this eef
        if len(eef_subtask_term_signal_names) == 0 and len(eef_subtask_start_signal_names) == 0:
            continue

        while True:
            is_paused = True
            skip_episode = False
            print(f'\tPlaying the episode for subtask annotations for eef "{eef_name}".')
            print("\tSubtask signals to annotate:")
            if len(eef_subtask_start_signal_names) > 0:
                print(f"\t\t- Start:\t{eef_subtask_start_signal_names}")
            print(f"\t\t- Termination:\t{eef_subtask_term_signal_names}")

            print('\n\tPress "N" to begin.')
            print('\tPress "B" to pause.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')
            marked_subtask_action_indices = []
            task_success_result = replay_episode(env, episode, success_term)
            if skip_episode:
                print("\tSkipping the episode.")
                return False

            print(f"\tSubtasks marked at action indices: {marked_subtask_action_indices}")
            expected_subtask_signal_count = len(eef_subtask_term_signal_names) + len(eef_subtask_start_signal_names)
            if task_success_result and expected_subtask_signal_count == len(marked_subtask_action_indices):
                print(f'\tAll {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were annotated.')
                for marked_signal_index in range(expected_subtask_signal_count):
                    if args_cli.annotate_subtask_start_signals and marked_signal_index % 2 == 0:
                        subtask_start_signal_action_indices[
                            eef_subtask_start_signal_names[int(marked_signal_index / 2)]
                        ] = marked_subtask_action_indices[marked_signal_index]
                    if not args_cli.annotate_subtask_start_signals:
                        # Direct mapping when only collecting termination signals
                        subtask_term_signal_action_indices[eef_subtask_term_signal_names[marked_signal_index]] = (
                            marked_subtask_action_indices[marked_signal_index]
                        )
                    elif args_cli.annotate_subtask_start_signals and marked_signal_index % 2 == 1:
                        # Every other signal is a termination when collecting both types
                        subtask_term_signal_action_indices[
                            eef_subtask_term_signal_names[math.floor(marked_signal_index / 2)]
                        ] = marked_subtask_action_indices[marked_signal_index]
                break

            if not task_success_result:
                print("\tThe final task was not completed.")
                return False

            if expected_subtask_signal_count != len(marked_subtask_action_indices):
                print(
                    f"\tOnly {len(marked_subtask_action_indices)} out of"
                    f' {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were'
                    " annotated."
                )

            print(f'\tThe episode will be replayed again for re-marking subtask signals for the eef "{eef_name}".\n')

    annotated_episode = env.recorder_manager.get_episode(0)
    for (
        subtask_term_signal_name,
        subtask_term_signal_action_index,
    ) in subtask_term_signal_action_indices.items():
        # subtask termination signal is false until subtask is complete, and true afterwards
        subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
        subtask_signals[:subtask_term_signal_action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{subtask_term_signal_name}", subtask_signals)

    if args_cli.annotate_subtask_start_signals:
        for (
            subtask_start_signal_name,
            subtask_start_signal_action_index,
        ) in subtask_start_signal_action_indices.items():
            subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
            subtask_signals[:subtask_start_signal_action_index] = False
            annotated_episode.add(
                f"obs/datagen_info/subtask_start_signals/{subtask_start_signal_name}", subtask_signals
            )

    return True


if __name__ == "__main__":
    # run the main function
    successful_task_count = main()
    # close sim app
    simulation_app.close()
    # exit with the number of successful task completions as return code
    exit(successful_task_count)
