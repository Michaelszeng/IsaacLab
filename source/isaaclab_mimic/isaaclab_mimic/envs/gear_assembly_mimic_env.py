# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLMimicEnv


class FrankaGearAssemblyIKRelMimicEnv(ManagerBasedRLMimicEnv):
    """MimicGen environment wrapper for the Franka gear mesh assembly task.

    Action space: [delta_pos(3), delta_rot_axisangle(3), gripper(1)] = 7D
    Two subtasks: grasp the gear from the table, then place it on the shaft.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return current EEF pose as (N, 4, 4) matrices."""
        if env_ids is None:
            env_ids = slice(None)
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF pose + gripper action to a 7D env action."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (gripper_action,) = gripper_action_dict.values()
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert a batch of 7D env actions to target EEF poses."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / (delta_rotation_angle + 1e-8)
        is_zero = delta_rotation_angle.squeeze(1) < 1e-6
        delta_rotation_axis[is_zero] = 0.0

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        return {eef_name: PoseUtils.make_pose(target_pos, target_rot).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from (N, T, 7) action sequences."""
        return {list(self.cfg.subtask_configs.keys())[0]: actions[..., -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return gear (held_asset) and gear base (fixed_asset) poses as 4×4 matrices."""
        if env_ids is None:
            env_ids = slice(None)

        gear: RigidObject = self.scene["held_asset"]
        gear_pos = gear.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        gear_quat = gear.data.root_quat_w[env_ids]

        gear_base: RigidObject = self.scene["fixed_asset"]
        base_pos = gear_base.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        base_quat = gear_base.data.root_quat_w[env_ids]

        return {
            "held_asset": PoseUtils.make_pose(gear_pos, PoseUtils.matrix_from_quat(gear_quat)),
            "fixed_asset": PoseUtils.make_pose(base_pos, PoseUtils.matrix_from_quat(base_quat)),
        }

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return binary subtask signals from observation buffer."""
        if env_ids is None:
            env_ids = slice(None)
        return {"grasp": self.obs_buf["subtask_terms"]["grasp"][env_ids]}

    def _reset_idx(self, env_ids):
        if hasattr(self, "termination_manager") and len(env_ids) > 0:
            try:
                tm = self.termination_manager
                time_outs = tm.time_outs
                success_term_names = [n for n in tm.active_terms if n != "time_out"]
                success_fired = tm.terminated
                for env_id in env_ids.tolist():
                    if success_fired[env_id].item():
                        label = success_term_names[0] if success_term_names else "success"
                        print(f"[Env {env_id}] SUCCESS: {label}", flush=True)
                    elif time_outs[env_id].item():
                        print(f"[Env {env_id}] FAILURE: episode timed out", flush=True)
                    else:
                        print(f"[Env {env_id}] RESET (manual)", flush=True)
            except Exception as e:
                print(f"[reset_idx] diagnostic error: {type(e).__name__}: {e}", flush=True)
        super()._reset_idx(env_ids)
