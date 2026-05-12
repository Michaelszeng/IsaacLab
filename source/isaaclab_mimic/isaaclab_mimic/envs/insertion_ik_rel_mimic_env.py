# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.sensors import FrameTransformer


class FrankaInsertionIKRelMimicEnv(ManagerBasedRLMimicEnv):
    """MimicGen environment wrapper for the Franka peg-in-hole insertion task.

    Action space: [delta_pos(3), delta_rot_axisangle(3), gripper(1)] = 7D
    The arm actions use IK-relative delta control (scale=0.5).
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Return current EEF pose as a (N, 4, 4) matrix."""
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
        """Convert a target EEF pose + gripper action to a 7D env action."""
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
        """Convert a batch of 7D env actions to target EEF poses. Inverse of target_eef_pose_to_action."""
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
        """Extract gripper actions from a sequence of env actions. Shape: (N, T, 7) → (N, T, 1)."""
        return {list(self.cfg.subtask_configs.keys())[0]: actions[..., -1:]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return poses of the peg (held_asset) and socket (fixed_asset) as 4x4 matrices.

        The socket is an Articulation with fix_root_link=True, so we read it directly rather
        than relying on the default rigid_object-only implementation.
        """
        if env_ids is None:
            env_ids = slice(None)

        peg: RigidObject = self.scene["held_asset"]
        peg_pos = peg.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        peg_quat = peg.data.root_quat_w[env_ids]

        socket: Articulation = self.scene["fixed_asset"]
        socket_pos = socket.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        socket_quat = socket.data.root_quat_w[env_ids]

        return {
            "held_asset": PoseUtils.make_pose(peg_pos, PoseUtils.matrix_from_quat(peg_quat)),
            "fixed_asset": PoseUtils.make_pose(socket_pos, PoseUtils.matrix_from_quat(socket_quat)),
        }

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return binary subtask termination signals from the observation buffer."""
        if env_ids is None:
            env_ids = slice(None)

        signals = {}
        subtask_terms = self.obs_buf["subtask_terms"]
        # "grasp" marks the end of subtask 1 (peg grasped).
        signals["grasp"] = subtask_terms["grasp"][env_ids]
        # "insert" is the final subtask - no term signal needed, but returning it
        # here allows manual annotation if desired.
        return signals

    def _reset_idx(self, env_ids):
        if hasattr(self, "termination_manager") and len(env_ids) > 0:
            try:
                tm = self.termination_manager
                time_outs = tm.time_outs
                # Find the success term: whichever non-timeout term is registered.
                # (record_demos.py strips "success" and "time_out" before env creation,
                # leaving only the primary term — "peg_inserted" — in the manager.)
                success_term_names = [n for n in tm.active_terms if n != "time_out"]
                success_fired = tm.terminated  # union of all non-timeout terms
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
