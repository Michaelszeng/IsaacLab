"""Interactively view episodes from an Isaac Lab HDF5 demo dataset.

Reads the dataset format produced by `record_demos.py`:

    file.hdf5
    └── data/
        ├── attrs: env_args, total
        ├── demo_0/
        │   ├── attrs: num_samples, success
        │   ├── obs/
        │   │   ├── eef_pos              (T, 3)
        │   │   ├── eef_quat             (T, 4)
        │   │   ├── actions              (T, 7)
        │   │   ├── wrist_cam            (T, H, W, 3)  uint8
        │   │   ├── scene_cam_front      (T, H, W, 3)  uint8
        │   │   ├── scene_cam_rear_left  (T, H, W, 3)  uint8
        │   │   └── ...
        │   ├── actions                  (T, 7)
        │   └── ...
        ├── demo_1/
        └── ...

Camera datasets are auto-detected as any (T, H, W, 3 or 4) array under `obs/`.

Display uses matplotlib for the window/keyboard (works with both opencv-python
and opencv-python-headless — the latter is what Isaac Sim pulls in).

Controls:
    k / l    step 1 / 10 frames forward
    j / h    step 1 / 10 frames backward
    n / p    next / previous episode
    Space    toggle play/pause
    q        quit
"""

import argparse
import io
import time
from pathlib import Path

import cv2  # used only for non-GUI helpers: resize, putText, imdecode
import h5py
import matplotlib
import numpy as np

# Pick an interactive backend.  Need one of these for the viewer window.
_chosen_backend = None
for _backend in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg"):
    try:
        matplotlib.use(_backend, force=True)
        # Actually try creating a figure - matplotlib.use() doesn't import the
        # backend's bindings, so failures show up later.
        import matplotlib.pyplot as _plt_probe

        _plt_probe.figure()
        _plt_probe.close("all")
        _chosen_backend = _backend
        break
    except Exception:
        continue

if _chosen_backend is None:
    raise SystemExit(
        "No interactive matplotlib backend available.  Install one of:\n"
        "  pip install PyQt5         # → Qt5Agg backend (recommended)\n"
        "  pip install PySide2       # → QtAgg backend\n"
        "  apt-get install python3-tk # → TkAgg backend\n"
        "Then re-run this script."
    )

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert a (w, x, y, z) quaternion to a (3, 3) rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def get_demo_keys(data_group: h5py.Group) -> list[str]:
    keys = [k for k in data_group.keys() if k.startswith("demo_")]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def find_camera_keys(obs_group: h5py.Group) -> list[str]:
    cam_keys = []
    for key in obs_group:
        item = obs_group[key]
        if isinstance(item, h5py.Dataset) and len(item.shape) == 4 and item.shape[-1] in (3, 4):
            cam_keys.append(key)
    cam_keys.sort()
    # wrist_cam first, scene cams after — purely cosmetic.
    cam_keys.sort(key=lambda k: (0 if "wrist" in k else 1, k))
    return cam_keys


def load_episode(data_group: h5py.Group, demo_key: str, cam_keys: list[str], want_state: bool):
    g = data_group[demo_key]
    obs = g["obs"]
    cams = [np.asarray(obs[k]) for k in cam_keys]
    success = bool(g.attrs.get("success", False))

    state = None
    if want_state and "eef_pos" in obs and "eef_quat" in obs:
        ee_pos = np.asarray(obs["eef_pos"], dtype=np.float32)
        ee_quat = np.asarray(obs["eef_quat"], dtype=np.float32)
        if "actions" in g:
            act = np.asarray(g["actions"], dtype=np.float32)
        elif "actions" in obs:
            act = np.asarray(obs["actions"], dtype=np.float32)
        else:
            act = None
        delta_pos = act[:, 0:3] if act is not None and act.shape[1] >= 3 else None
        delta_rot = act[:, 3:6] if act is not None and act.shape[1] >= 6 else None
        gripper = act[:, 6] if act is not None and act.shape[1] >= 7 else None
        state = (ee_pos, ee_quat, delta_pos, delta_rot, gripper)

    return cams, success, state


def concatenate_cameras(cams: list[np.ndarray], frame_idx: int) -> np.ndarray:
    target_h = cams[0].shape[1]
    panes = []
    for c in cams:
        img = c[frame_idx]
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape[0] != target_h:
            new_w = int(round(img.shape[1] * target_h / img.shape[0]))
            img = cv2.resize(img, (new_w, target_h))
        panes.append(img)
    return np.concatenate(panes, axis=1)


# ---------------------------------------------------------------------------
# State / action matplotlib panel (off-screen render → image array)
# ---------------------------------------------------------------------------


def render_state_panel(ee_pos, ee_quat, delta_pos, delta_rot, gripper, frame_idx, panel_h, panel_w):
    T = len(ee_pos)
    fig = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100)

    ax3d = fig.add_axes([0.05, 0.38, 0.90, 0.58], projection="3d")
    ax3d.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], color="lightgray", linewidth=0.8, zorder=1)
    for t in range(0, T - 1, max(1, T // 60)):
        frac = t / max(T - 1, 1)
        ax3d.plot(
            ee_pos[t : t + 2, 0], ee_pos[t : t + 2, 1], ee_pos[t : t + 2, 2],
            color=(frac, 0.0, 1.0 - frac), linewidth=1.5, zorder=2,
        )
    cx, cy, cz = ee_pos[frame_idx]
    ax3d.scatter([cx], [cy], [cz], color="yellow", s=60, zorder=5, edgecolors="black", linewidths=0.5)

    scale = max(np.ptp(ee_pos, axis=0).max() * 0.07, 0.01)
    rot = quat_to_mat(ee_quat[frame_idx])
    for i, col in enumerate(["red", "green", "blue"]):
        dx, dy, dz = rot[:, i] * scale
        ax3d.quiver(cx, cy, cz, dx, dy, dz, color=col, linewidth=1.5, arrow_length_ratio=0.3)

    if delta_pos is not None:
        dp = delta_pos[frame_idx]
        mag = float(np.linalg.norm(dp))
        if mag > 1e-6:
            dp_scaled = dp / mag * min(mag * 5, scale * 1.5)
            ax3d.quiver(
                cx, cy, cz, dp_scaled[0], dp_scaled[1], dp_scaled[2],
                color="orange", linewidth=2, linestyle="dashed", arrow_length_ratio=0.3,
            )

    ax3d.set_xlabel("X", fontsize=7, labelpad=0)
    ax3d.set_ylabel("Y", fontsize=7, labelpad=0)
    ax3d.set_zlabel("Z", fontsize=7, labelpad=0)
    ax3d.tick_params(labelsize=6, pad=0)
    ax3d.set_title(f"EE trajectory  (frame {frame_idx}/{T - 1})", fontsize=8, pad=2)

    ax_ts = fig.add_axes([0.10, 0.04, 0.85, 0.28])
    t_axis = np.arange(T)
    ax_ts.plot(t_axis, ee_pos[:, 0], color="red", linewidth=0.8, label="X")
    ax_ts.plot(t_axis, ee_pos[:, 1], color="green", linewidth=0.8, label="Y")
    ax_ts.plot(t_axis, ee_pos[:, 2], color="blue", linewidth=0.8, label="Z")
    if gripper is not None:
        ax_ts.plot(t_axis, gripper * 0.03, color="purple", linewidth=0.8, linestyle="dotted", label="grip×0.03")
    ax_ts.axvline(frame_idx, color="yellow", linewidth=1.2, zorder=5)
    ax_ts.set_xlim(0, max(T - 1, 1))
    ax_ts.tick_params(labelsize=6)
    ax_ts.set_ylabel("ee_pos (m)", fontsize=7)
    ax_ts.legend(fontsize=6, loc="upper right", ncol=2)
    ax_ts.set_facecolor("#1a1a1a")
    ax_ts.grid(color="gray", linewidth=0.3)

    fig.patch.set_facecolor("#1a1a1a")
    ax3d.set_facecolor("#1a1a1a")
    ax3d.grid(True, linewidth=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#1a1a1a", dpi=100)
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), np.uint8)
    panel_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    if panel_rgb.shape[:2] != (panel_h, panel_w):
        panel_rgb = cv2.resize(panel_rgb, (panel_w, panel_h))
    return panel_rgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Interactively view episodes from an Isaac Lab HDF5 demo dataset.")
    parser.add_argument("hdf5_path", help="Path to the .hdf5 dataset file")
    parser.add_argument("--episode", "-e", type=int, default=0, help="Episode index to start at (default: 0)")
    parser.add_argument("--fps", type=int, default=10, help="Playback speed in frames per second (default: 10)")
    parser.add_argument("--state", action="store_true", help="Show 3D state/action panel below the camera feeds.")
    parser.add_argument("--scale", type=float, default=1.0, help="Camera strip scale factor (default: 1.0).")
    parser.add_argument(
        "--max-width",
        type=int,
        default=1400,
        help="Maximum width of the camera strip in pixels.  Auto-shrinks --scale if needed (default: 1400).",
    )
    args = parser.parse_args()

    if not Path(args.hdf5_path).exists():
        raise SystemExit(f"File not found: {args.hdf5_path}")

    f = h5py.File(args.hdf5_path, "r")
    try:
        data = f["data"]
        demo_keys = get_demo_keys(data)
        if not demo_keys:
            raise SystemExit("No demos found under 'data/' in the dataset.")

        first_obs = data[demo_keys[0]]["obs"]
        cam_keys = find_camera_keys(first_obs)
        if not cam_keys:
            raise SystemExit("No camera (T,H,W,3/4) datasets found under obs/ in the first demo.")

        print(f"Dataset: {args.hdf5_path}")
        print(f"Episodes: {len(demo_keys)}")
        print(f"Cameras:  {cam_keys}")
        for k in cam_keys:
            arr = np.asarray(first_obs[k])
            # Quick stats — flag cameras whose mean is suspicious.
            per_frame_mean = arr.reshape(arr.shape[0], -1).mean(axis=1)
            overall_mean = float(per_frame_mean.mean())
            overall_std = float(per_frame_mean.std())
            tag = ""
            if overall_mean > 240:
                tag = "   ← LIKELY ALL-WHITE (camera did not capture the scene)"
            elif overall_mean < 15:
                tag = "   ← LIKELY ALL-BLACK"
            elif overall_std < 1.0:
                tag = "   ← LIKELY STATIC (no per-frame variation)"
            print(
                f"  {k}: shape {arr.shape} dtype {arr.dtype}  "
                f"mean={overall_mean:5.1f}  per-frame std={overall_std:4.1f}{tag}"
            )
        print(f"Obs keys: {sorted(first_obs.keys())}")
        print()
        print("Controls:")
        print("  k / l     step 1 / 10 frames forward")
        print("  j / h     step 1 / 10 frames backward")
        print("  n / p     next / previous episode")
        print("  Space     toggle play/pause")
        print("  q         quit")
        print()

        # Probe to figure out window dimensions.
        probe_cams, _, _ = load_episode(data, demo_keys[0], cam_keys, want_state=False)
        sample = concatenate_cameras(probe_cams, 0)
        base_h, base_w = sample.shape[:2]
        scale = args.scale
        # Auto-shrink if the requested scale would blow past max-width.
        if base_w * scale > args.max_width:
            scale = args.max_width / base_w
            print(f"Auto-scaling cameras down to scale={scale:.3f} to fit max-width={args.max_width}px.")
        cam_h = max(1, int(round(base_h * scale)))
        cam_w = max(1, int(round(base_w * scale)))
        state_h = 720 if args.state else 0
        win_w = cam_w
        win_h = cam_h + state_h
        print(f"Window: {win_w} × {win_h} px (camera strip {cam_w} × {cam_h}).")

        # ---- Set up the matplotlib figure / axes ----
        dpi = 100
        fig = plt.figure(figsize=(win_w / dpi, win_h / dpi), dpi=dpi)
        fig.canvas.manager.set_window_title("Dataset Viewer")
        # Disable tight_layout / constrained_layout — they re-position axes on each draw.
        try:
            fig.set_layout_engine("none")
        except AttributeError:
            pass

        if args.state:
            ax_cam = fig.add_axes([0, state_h / win_h, 1, cam_h / win_h])
            ax_state = fig.add_axes([0, 0, 1, state_h / win_h])
            ax_state.axis("off")
            ax_state.set_position([0, 0, 1, state_h / win_h])
            state_img_handle = ax_state.imshow(np.zeros((state_h, cam_w, 3), dtype=np.uint8), aspect="auto")
        else:
            ax_cam = fig.add_axes([0, 0, 1, 1])
            state_img_handle = None
        ax_cam.axis("off")
        # Lock axes to the full figure rect and disable any auto-rescaling.
        ax_cam.set_position([0, 0, 1, 1] if not args.state else [0, state_h / win_h, 1, cam_h / win_h])
        ax_cam.set_autoscale_on(False)
        cam_img_handle = ax_cam.imshow(np.zeros((cam_h, cam_w, 3), dtype=np.uint8), aspect="auto")
        # Lock data limits to the image extent.  set_data alone doesn't change extent,
        # but if something else were resizing the axes this would prevent it.
        ax_cam.set_xlim(0, cam_w)
        ax_cam.set_ylim(cam_h, 0)

        # ---- Shared UI state ----
        ep_idx = max(0, min(args.episode, len(demo_keys) - 1))
        cams, success, state_data = load_episode(data, demo_keys[ep_idx], cam_keys, args.state)
        ui = {
            "frame_idx": 0,
            "ep_idx": ep_idx,
            "playing": False,
            "quit": False,
            "redraw": True,
            "panel_cache": [None] * len(cams[0]),
        }

        def on_episode_change(new_ep):
            nonlocal cams, success, state_data
            cams, success, state_data = load_episode(data, demo_keys[new_ep], cam_keys, args.state)
            ui["panel_cache"] = [None] * len(cams[0])

        def step(delta):
            T = len(cams[0])
            new = ui["frame_idx"] + delta
            new = max(0, min(T - 1, new))
            if new == ui["frame_idx"]:
                if delta > 0:
                    ui["playing"] = False  # hit the end
                return
            ui["frame_idx"] = new
            ui["redraw"] = True

        def goto_episode(delta):
            new_ep = max(0, min(ui["ep_idx"] + delta, len(demo_keys) - 1))
            if new_ep != ui["ep_idx"]:
                ui["ep_idx"] = new_ep
                on_episode_change(new_ep)
                ui["frame_idx"] = 0
                ui["playing"] = False
                ui["redraw"] = True

        def on_key(event):
            k = (event.key or "").lower()
            if k == "q" or k == "escape":
                ui["quit"] = True
                plt.close(fig)
                return
            if k == " ":
                ui["playing"] = not ui["playing"]
            elif k == "k":
                step(1)
            elif k == "l":
                step(10)
            elif k == "j":
                step(-1)
            elif k == "h":
                step(-10)
            elif k == "n":
                goto_episode(1)
            elif k == "p":
                goto_episode(-1)

        def on_close(event):
            ui["quit"] = True

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_close)

        plt.show(block=False)

        last_step_t = time.time()
        step_interval = 1.0 / max(args.fps, 1)

        while not ui["quit"]:
            if not plt.fignum_exists(fig.number):
                break

            if ui["playing"]:
                now = time.time()
                if now - last_step_t >= step_interval:
                    step(1)
                    last_step_t = now

            if ui["redraw"]:
                T = len(cams[0])
                cam_rgb = concatenate_cameras(cams, ui["frame_idx"])
                if (cam_rgb.shape[0], cam_rgb.shape[1]) != (cam_h, cam_w):
                    cam_rgb = cv2.resize(cam_rgb, (cam_w, cam_h))
                label = (
                    f"Ep {ui['ep_idx'] + 1}/{len(demo_keys)} ({demo_keys[ui['ep_idx']]})  |  "
                    f"Frame {ui['frame_idx']}/{T - 1}  |  "
                    f"{'SUCCESS' if success else 'FAILURE'}"
                )
                cv2.putText(cam_rgb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cam_img_handle.set_data(cam_rgb)

                if args.state and state_data is not None and state_img_handle is not None:
                    fi = ui["frame_idx"]
                    if ui["panel_cache"][fi] is None:
                        ui["panel_cache"][fi] = render_state_panel(
                            *state_data, frame_idx=fi, panel_h=state_h, panel_w=cam_w,
                        )
                    state_img_handle.set_data(ui["panel_cache"][fi])

                fig.canvas.draw_idle()
                ui["redraw"] = False

            plt.pause(0.01)

    finally:
        f.close()


if __name__ == "__main__":
    main()
