# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.insertion.config.franka.insertion_osc_env_cfg import (
    FrankaInsertionOSCEnvCfg,
)


@configclass
class FrankaInsertionOSCMimicEnvCfg(FrankaInsertionOSCEnvCfg, MimicEnvCfg):
    """MimicGen config for the Franka insertion task with OSC control.

    The delta-pose math in FrankaInsertionIKRelMimicEnv is identical to OSC
    pose_rel mode, so the same env class is reused.
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_insertion_franka_osc"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 50
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 1

        subtask_configs = []

        subtask_configs.append(
            SubTaskConfig(
                object_ref="held_asset",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp the peg from the table",
                next_subtask_description="Insert the peg into the socket",
            )
        )

        subtask_configs.append(
            SubTaskConfig(
                object_ref="fixed_asset",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Insert the peg into the socket",
            )
        )

        self.subtask_configs["franka"] = subtask_configs
