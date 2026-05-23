"""Evaluate a diffusion_policy checkpoint on an Isaac Lab MimicGen task.

Mirrors furniture-bench-juicer's evaluate_model_custom.py: same CLI flags,
same output layout (results.csv, summary.txt, results.pkl, videos/), and the
same rollout structure. Adapted for Isaac Lab's gymnasium-style env API and the
gear-assembly / insertion task obs schema.

Usage:
    python scripts/eval/evaluate_model_custom.py \\
        --checkpoint /path/to/checkpoint.ckpt \\
        --task gear_assembly \\
        --n-rollouts 50 \\
        --n-envs 1 \\
        --enable_cameras
"""

# ---------------------------------------------------------------------------
# AppLauncher setup. Must happen before any torch/gym imports because Isaac
# Lab's bootstrap configures CUDA/Vulkan/USD before its own libraries import.
# ---------------------------------------------------------------------------

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a diffusion_policy checkpoint on Isaac Lab.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
parser.add_argument(
    "--task",
    "-t",
    type=str,
    required=True,
    help="Full Isaac Lab task id (e.g., Isaac-GearAssembly-Franka-IK-Rel-Mimic-v0).",
)
parser.add_argument("--n-rollouts", type=int, default=10)
parser.add_argument("--n-envs", type=int, default=1)
parser.add_argument(
    "--randomness",
    type=str,
    default="low",
    help="Compatibility flag (recorded in results.pkl); Isaac Lab tasks don't currently use this.",
)
parser.add_argument("--save-video", action="store_true", default=True)
parser.add_argument("--no-save-video", dest="save_video", action="store_false")
parser.add_argument(
    "--n-video-trials",
    type=int,
    default=20,
    help="Save videos for only the first N trials (default: 20). Set to -1 to save all.",
)
parser.add_argument(
    "--record-failures",
    action="store_true",
    default=False,
    help="If set, only save videos of failed trials, and save all of them (overrides --n-video-trials).",
)
parser.add_argument(
    "--n-action-steps",
    type=int,
    default=None,
    help="Override action horizon (default: use value from checkpoint config)",
)
parser.add_argument(
    "--task-timeout",
    type=int,
    default=None,
    help="Max rollout steps per trial (default: derive from env cfg's episode_length_s/dt/decimation)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to write results (default: outputs/<date>/<time>)",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume from an existing results.pkl in --output-dir (requires --output-dir)",
)
parser.add_argument(
    "--video-fps",
    type=int,
    default=10,
    help="Frame rate for saved videos.",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.resume and args.output_dir is None:
    parser.error("--resume requires --output-dir to be specified")

# Launch Isaac Sim before importing torch / gym / hydra.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Everything else.
# ---------------------------------------------------------------------------

import collections
import csv
import datetime
import math
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import dill
import gymnasium as gym
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

import isaaclab_mimic.envs  # noqa: F401 — registers Isaac-*-Mimic-v0 tasks

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# diffusion_policy configs reference ${eval:…}.
OmegaConf.register_new_resolver("eval", eval, replace=True)


# ---------------------------------------------------------------------------
# Obs preprocessing
# ---------------------------------------------------------------------------


def preprocess_obs(env_obs, device, obs_keys):
    """Transform an Isaac Lab raw env obs into a dict[str -> tensor] keyed by the
    policy's expected obs keys.

    Isaac Lab returns obs as a nested dict like::

        {"policy": {"eef_pos": (N, 3), ..., "wrist_cam": (N, H, W, 3)},
         "subtask_terms": {...}}

    For each key the policy expects (from ``cfg.shape_meta.obs``), we pull it
    from ``obs["policy"]`` and move to ``device`` as float32 (camera tensors
    are kept at source resolution; any RGBA frames have the alpha channel
    dropped).  Normalisation is left to the policy's normalizer — must match
    the training pipeline's resolution.
    """
    pol = env_obs["policy"] if isinstance(env_obs, dict) and "policy" in env_obs else env_obs
    result = {}
    for k in obs_keys:
        if k not in pol:
            available = list(pol.keys()) if hasattr(pol, "keys") else "<not a dict>"
            raise KeyError(
                f"Policy expects obs key '{k}' but it isn't in env.obs['policy']. Available keys: {available}"
            )
        v = pol[k]
        # Camera: (N, H, W, 3 or 4) uint8 tensor.
        if v.ndim == 4 and v.shape[-1] in (3, 4):
            if v.shape[-1] == 4:
                v = v[..., :3]
            result[k] = v.float().to(device)
        else:
            result[k] = v.float().to(device)
    return result


def build_obs_dict(obs_deque: collections.deque, device: torch.device) -> dict:
    """Stack a deque of per-step obs dicts → {"obs": {key: (N, T, ...)}}."""
    keys = obs_deque[0].keys()
    obs_stacked = {k: torch.stack([o[k] for o in obs_deque], dim=1) for k in keys}
    return {"obs": obs_stacked}


# ---------------------------------------------------------------------------
# Policy loading (identical to the reference except for the dataset import path)
# ---------------------------------------------------------------------------


def load_policy(checkpoint_path: str, device: torch.device):
    """Load a diffusion_policy workspace + policy from a .ckpt file.

    Also loads (or generates) the paired normalizer.pt that lives next to the
    checkpoint's parent ``checkpoints/`` directory.
    """
    import hydra  # local import to avoid global hydra side-effects at module load
    from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa

    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "lr_scheduler"], include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model

    ckpt_path = Path(checkpoint_path)
    normalizer_path = ckpt_path.parent.parent / "normalizer.pt"
    if normalizer_path.exists():
        print(f"Loading normalizer from {normalizer_path}")
        # weights_only=False because LinearNormalizer is a pickled Python object,
        # not a plain state dict. PyTorch 2.6 made weights_only=True the default.
        normalizer = torch.load(normalizer_path, map_location="cpu", weights_only=False)
    else:
        print(f"Normalizer not found at {normalizer_path}, generating from dataset…")
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, normalizer_path)
        print(f"Saved normalizer to {normalizer_path}")

    policy.set_normalizer(normalizer)
    policy.to(device).eval()
    if hasattr(policy, "obs_encoder"):
        policy.obs_encoder.eval()  # belt-and-suspenders: keep encoders' BN in eval mode
    return policy, cfg


# ---------------------------------------------------------------------------
# File I/O helpers (output format mirrors the reference exactly)
# ---------------------------------------------------------------------------


def _write_mp4(frames, path, fps=10):
    """Write a list of (H, W, 3) uint8 frames to MP4."""
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_summary(n_success, n_total, trial_records, summary_path):
    n_failure = sum(1 for r in trial_records if r["result"] == "failure")
    n_timeout = sum(1 for r in trial_records if r["result"] == "timeout")
    rate = n_success / n_total if n_total > 0 else 0.0
    avg_steps = sum(r["trial_time"] for r in trial_records) / len(trial_records) if trial_records else 0.0
    with open(summary_path, "w") as f:
        f.write(f"Trials completed : {n_total} / {args.n_rollouts}\n")
        f.write(f"Successes        : {n_success}\n")
        f.write(f"Failures         : {n_failure}\n")
        f.write(f"Timeouts         : {n_timeout}\n")
        f.write(f"Success rate     : {rate:.1%}\n")
        f.write(f"Avg trial steps  : {avg_steps:.1f}\n")


def _repair_csv_and_summary_from_pkl_data(pkl_path: Path):
    """Rewrite results.csv and summary.txt to match results.pkl; return (csv_file, csv_writer)."""
    saved = pickle.load(open(pkl_path, "rb"))
    out_dir, fields = pkl_path.parent, ["trial", "result", "reward", "trial_time"]
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(saved["trials"])
    _write_summary(saved["n_success"], saved["n_total"], saved["trials"], out_dir / "summary.txt")
    csv_file = open(csv_path, "a", newline="")
    return csv_file, csv.DictWriter(csv_file, fieldnames=fields)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _make_video_frame(env_obs, env_idx: int, video_cam_keys: list) -> np.ndarray:
    """Build a side-by-side video frame for env_idx by concatenating cameras
    at source resolution.  All cameras must share the same H.
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
    # Resize lower-resolution panes to the first pane's height (cheap, only if
    # cameras have mismatched resolutions).
    h = panes[0].shape[0]
    panes = [p if p.shape[0] == h else cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panes]
    return np.concatenate(panes, axis=1)


@torch.no_grad()
def run_rollout(
    env,
    policy,
    n_obs_steps: int,
    rollout_max_steps: int,
    device: torch.device,
    obs_keys: set,
    record_video: bool = False,
    n_action_steps: int = None,
    video_cam_keys: list = None,
) -> dict:
    """One round of parallel rollouts. Same output schema as the reference."""
    n_envs = env.num_envs
    obs, _ = env.reset()
    preprocessed = preprocess_obs(obs, device, obs_keys)
    obs_deque: collections.deque = collections.deque([preprocessed] * n_obs_steps, maxlen=n_obs_steps)
    action_queue: collections.deque = collections.deque()

    total_reward = torch.zeros(n_envs, device=device)
    success_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
    done_step = torch.full((n_envs,), -1, dtype=torch.long, device=device)
    step = 0

    if record_video:
        frame_buffers = [[] for _ in range(n_envs)]
        # First frame (post-reset) so the video isn't missing t=0.
        for env_idx in range(n_envs):
            frame_buffers[env_idx].append(_make_video_frame(obs, env_idx, video_cam_keys or []))

    while step < rollout_max_steps:
        if (done_step >= 0).all():
            break

        if len(action_queue) == 0:
            obs_dict = build_obs_dict(obs_deque, device)
            result = policy.predict_action(obs_dict, use_DDIM=True)
            start = n_obs_steps - 1
            actions = result["action_pred"][:, start:]
            n_steps = n_action_steps if n_action_steps is not None else policy.n_action_steps
            for t in range(n_steps):
                action_queue.append(actions[:, t, :])

        action = action_queue.popleft()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except RuntimeError as e:
            # Mirrors the reference's IsaacGym OSC fallback: abort gracefully.
            print(
                f"[run_rollout] env.step raised RuntimeError at step {step}: {e}. "
                "Marking unfinished envs as failures and ending rollout."
            )
            unfinished = done_step == -1
            done_step[unfinished] = step
            break

        # reward may be (n_envs,) or (n_envs, 1)
        if reward.ndim > 1:
            reward = reward.squeeze(-1)
        total_reward = total_reward + reward.float()

        cur_done = terminated | truncated
        newly_done = cur_done & (done_step == -1)
        done_step[newly_done] = step
        # In our env, the only non-time_out termination IS "success" — so
        # terminated[i]==True (and not truncated) means the gear is on the shaft.
        success_mask = success_mask | (newly_done & terminated)

        if record_video:
            for env_idx in range(n_envs):
                # Only append while this env's trial is still ongoing.
                if done_step[env_idx].item() == -1 or done_step[env_idx].item() == step:
                    frame_buffers[env_idx].append(_make_video_frame(obs, env_idx, video_cam_keys or []))

        preprocessed = preprocess_obs(obs, device, obs_keys)
        obs_deque.append(preprocessed)
        step += 1

    success_np = success_mask.cpu().numpy()
    done_step_np = done_step.cpu().numpy()

    # Classify each env's outcome. Our env's only non-time_out termination is
    # success, so anything that's "done but not success" is a timeout in spirit.
    # The "failure" bucket is kept so the output schema matches the reference;
    # add task-specific failure terminations later to populate it.
    results = []
    for i in range(n_envs):
        if success_np[i]:
            results.append("success")
        elif done_step_np[i] >= 0:
            results.append("timeout")
        else:
            results.append("timeout")

    steps_per_env = np.where(done_step_np >= 0, done_step_np + 1, step)

    out = {
        "success": success_np,
        "total_reward": total_reward.cpu().numpy(),
        "result": results,
        "steps": step,
        "steps_per_env": steps_per_env,
    }
    if record_video:
        out["frames"] = frame_buffers
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"No CUDA GPUs are available (device={args.device}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})."
        )

    print(f"Loading policy from {args.checkpoint}")
    policy, cfg = load_policy(args.checkpoint, device)
    n_obs_steps: int = int(cfg.n_obs_steps)
    n_action_steps: int = args.n_action_steps

    policy_obs_keys = set(cfg.shape_meta.obs.keys())
    is_image_based = any(("cam" in k) or ("image" in k) for k in policy_obs_keys)
    print(f"Policy obs keys: {sorted(policy_obs_keys)}")
    print(f"Policy type: {'image-based' if is_image_based else 'state-based'}")
    print(f"Task: {args.task}")
    print(
        f"n_obs_steps={n_obs_steps}, "
        f"n_action_steps={'from_cfg' if n_action_steps is None else n_action_steps}, "
        f"n_envs={args.n_envs}"
    )

    # ---- Create the Isaac Lab env ----
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.n_envs)
    # Default rollout cap: derive from cfg.episode_length_s when --task-timeout is absent.
    if args.task_timeout is not None:
        rollout_max_steps = args.task_timeout
    else:
        rollout_max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    print(f"Creating env (task={args.task}, max_steps={rollout_max_steps})")

    np.random.seed(42)
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    # Cameras to use for videos (raw resolution from env.obs, not the policy-input resized ones).
    video_cam_keys = sorted(
        [k for k in policy_obs_keys if "cam" in k or "image" in k],
        key=lambda k: (0 if "wrist" in k else 1, k),  # wrist first
    )
    if args.save_video and not video_cam_keys:
        print("Warning: --save-video requested but no camera obs keys detected; videos will be blank placeholders.")

    # ---- Output directory (fresh or resumed) ----
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        now = datetime.datetime.now()
        out_dir = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    if args.save_video:
        videos_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_total = 0
    all_trial_records = []
    resuming = False

    if args.resume:
        pkl_path = out_dir / "results.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("n_total", 0) > 0:
                n_success = saved["n_success"]
                n_total = saved["n_total"]
                all_trial_records = saved["trials"]
                resuming = True
            else:
                print(f"Found results.pkl at {pkl_path} but no completed trials; starting fresh.")
        else:
            print(f"--resume set but no results.pkl found in {out_dir}; starting fresh.")

    n_rounds = max(1, math.ceil(args.n_rollouts / args.n_envs))
    i_start = math.ceil(n_total / args.n_envs)

    csv_path = out_dir / "results.csv"
    csv_fields = ["trial", "result", "reward", "trial_time"]
    summary_path = out_dir / "summary.txt"

    if resuming:
        print(
            f"Resuming: {n_total}/{args.n_rollouts} trials done"
            f" ({n_success} successes, {n_success / n_total:.1%});"
            f" starting from round {i_start + 1}/{n_rounds}"
        )
        csv_file, csv_writer = _repair_csv_and_summary_from_pkl_data(out_dir / "results.pkl")
    else:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()
        csv_file.flush()

    for i in range(i_start, n_rounds):
        t_start = time.time()
        video_budget = args.n_video_trials if args.n_video_trials >= 0 else args.n_rollouts
        if args.record_failures:
            record_this_round = args.save_video
        else:
            record_this_round = args.save_video and (n_total < video_budget)

        round_result = run_rollout(
            env=env,
            policy=policy,
            n_obs_steps=n_obs_steps,
            rollout_max_steps=rollout_max_steps,
            device=device,
            obs_keys=policy_obs_keys,
            record_video=record_this_round,
            n_action_steps=n_action_steps,
            video_cam_keys=video_cam_keys,
        )
        rollout_time = time.time() - t_start

        # Stop saving trial records once we hit n_rollouts (exactly N records).
        for env_idx in range(args.n_envs):
            if n_total >= args.n_rollouts:
                break
            trial_num = n_total + 1
            result_str = round_result["result"][env_idx]
            record = {
                "trial": trial_num,
                "result": result_str,
                "reward": float(round_result["total_reward"][env_idx]),
                "trial_time": int(round_result["steps_per_env"][env_idx]),
            }
            all_trial_records.append(record)
            n_success += int(round_result["success"][env_idx])
            n_total += 1

            csv_writer.writerow(record)
            csv_file.flush()

            if record_this_round:
                save_this_video = False
                if args.record_failures:
                    if result_str != "success":
                        save_this_video = True
                elif trial_num <= video_budget:
                    save_this_video = True

                if save_this_video and round_result.get("frames"):
                    video_path = videos_dir / f"trial_{trial_num:04d}_{result_str}.mp4"
                    _write_mp4(round_result["frames"][env_idx], video_path, fps=args.video_fps)
                    print(f"  Saved video: {video_path.name}")

        _write_summary(n_success, n_total, all_trial_records, summary_path)

        pkl_path = out_dir / "results.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "trials": all_trial_records,
                    "n_success": n_success,
                    "n_total": n_total,
                    "success_rate": n_success / n_total if n_total > 0 else 0.0,
                    "checkpoint": args.checkpoint,
                    "task": args.task,
                    "randomness": args.randomness,
                    "n_obs_steps": n_obs_steps,
                    "rollout_max_steps": rollout_max_steps,
                },
                f,
            )

        success_rate = n_success / n_total
        video_tag = "video=on" if record_this_round else "video=off"
        total_time = time.time() - t_start
        print(
            f"Round {i + 1}/{n_rounds} [{video_tag}]: rollout time={rollout_time:.1f}s, total time={total_time:.1f}s, "
            f"result={round_result['result']}  running {n_success}/{n_total} ({success_rate:.1%})"
        )

    csv_file.close()

    final_success_rate = n_success / n_total if n_total > 0 else 0.0
    print(f"\nFinal success rate: {n_success}/{n_total} ({final_success_rate:.1%})")
    print(f"Results written to {out_dir}/")

    # Best-effort clean shutdown of the sim app, then a hard exit to bypass any
    # C++ destructor segfaults during normal Python teardown (mirrors the
    # reference's os._exit(0) pattern).
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        simulation_app.close()
    except Exception:
        pass
    os._exit(0)
