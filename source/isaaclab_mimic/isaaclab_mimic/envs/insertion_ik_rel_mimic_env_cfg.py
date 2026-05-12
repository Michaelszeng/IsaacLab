# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.insertion.config.franka.insertion_ik_rel_env_cfg import (
    FrankaInsertionIKRelEnvCfg,
)


@configclass
class FrankaInsertionIKRelMimicEnvCfg(FrankaInsertionIKRelEnvCfg, MimicEnvCfg):
    """MimicGen config for the Franka peg-in-hole insertion task.

    Two subtasks:
      1. Grasp  – robot picks up the peg from the table.
                  Object reference: peg (held_asset).
                  Termination signal: "grasp".
      2. Insert – robot aligns the peg with the socket and inserts it.
                  Object reference: socket (fixed_asset).
                  This is the final subtask (subtask_term_signal=None).
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_insertion_franka_ik_rel"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 50
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 1

        subtask_configs = []

        # ------------------------------------------------------------------
        # Subtask 1: Grasp the peg.
        # The trajectory segment is transformed relative to the peg frame so
        # that generated grasps adapt to new peg positions.
        # ------------------------------------------------------------------
        subtask_configs.append(
            SubTaskConfig(
                object_ref="held_asset",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                # Low noise is important for tight-tolerance contact tasks.
                action_noise=0.01,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp the peg from the table",
                next_subtask_description="Insert the peg into the socket",
            )
        )

        # ------------------------------------------------------------------
        # Subtask 2: Insert the peg into the socket.
        # The trajectory is transformed relative to the socket frame so that
        # generated insertions adapt to small variations in socket position.
        # This is the final subtask so subtask_term_signal=None.
        # ------------------------------------------------------------------
        subtask_configs.append(
            SubTaskConfig(
                object_ref="fixed_asset",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                # Very low noise for the contact-rich insertion phase.
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Insert the peg into the socket",
            )
        )

        self.subtask_configs["franka"] = subtask_configs
