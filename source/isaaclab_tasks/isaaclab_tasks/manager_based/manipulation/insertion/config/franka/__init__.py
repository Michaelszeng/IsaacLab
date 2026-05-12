# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="Isaac-Insertion-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insertion_ik_rel_env_cfg:FrankaInsertionIKRelEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insertion-Franka-OSC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.insertion_osc_env_cfg:FrankaInsertionOSCEnvCfg",
    },
    disable_env_checker=True,
)
