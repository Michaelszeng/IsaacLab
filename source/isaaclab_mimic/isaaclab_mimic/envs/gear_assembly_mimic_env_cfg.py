# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.gear_assembly.config.franka.gear_assembly_ik_rel_env_cfg import (
    FrankaGearAssemblyIKRelEnvCfg,
)


@configclass
class FrankaGearAssemblyIKRelMimicEnvCfg(FrankaGearAssemblyIKRelEnvCfg, MimicEnvCfg):
    """MimicGen config for the Franka gear mesh assembly task.

    Two subtasks:
      1. Grasp the large gear from the table (term_signal: "grasp").
      2. Place it on the large shaft (no term_signal — episode ends on success).
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_gear_assembly_franka_ik_rel"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 50
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 77

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
                description="Grasp the gear from the table",
                next_subtask_description="Place the gear on the shaft",
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
                description="Place the gear on the shaft",
            )
        )

        self.subtask_configs["franka"] = subtask_configs
