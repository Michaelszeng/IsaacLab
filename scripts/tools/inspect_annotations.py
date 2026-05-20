"""Inspect MimicGen subtask annotations in an annotated Isaac Lab HDF5 dataset.

`annotate_demos.py --auto` writes per-demo subtask boundary info under
`demo_N/obs/datagen_info/`.  This tool reads that and:

 1. Prints, for every demo, the frame at which each subtask term signal first
    becomes True — i.e. the subtask boundary MimicGen will use.
 2. Optionally pops a matplotlib timeline showing every demo's subtask segments
    overlaid as a stacked bar chart (one row per demo, coloured by subtask).

Headless-friendly: with `--no-plot`, only stdout output is produced — no GUI
backend needed.

Usage:
    python scripts/tools/inspect_annotations.py ./datasets/gear_assembly_annotated.hdf5
    python scripts/tools/inspect_annotations.py ./datasets/gear_assembly_annotated.hdf5 --no-plot
    python scripts/tools/inspect_annotations.py ./datasets/gear_assembly_annotated.hdf5 --save-plot timeline.png
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def get_demo_keys(data_group: h5py.Group) -> list[str]:
    keys = [k for k in data_group.keys() if k.startswith("demo_")]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def first_true(arr: np.ndarray) -> int | None:
    """Index of the first True in a bool array, or None if never True."""
    idx = int(np.argmax(arr))
    return idx if bool(arr[idx]) else None


def load_subtask_signals(demo_group: h5py.Group) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Return (term_signals, start_signals) for this demo, both dict[str -> (T,) bool].

    start_signals is None if the annotation didn't record subtask-start signals.
    """
    try:
        di = demo_group["obs/datagen_info"]
    except KeyError as e:
        raise KeyError(
            f"{demo_group.name}: missing obs/datagen_info — was this file produced by annotate_demos.py?"
        ) from e

    term = {k: np.asarray(di["subtask_term_signals"][k]).astype(bool) for k in di["subtask_term_signals"].keys()}
    starts = None
    if "subtask_start_signals" in di:
        starts = {k: np.asarray(di["subtask_start_signals"][k]).astype(bool) for k in di["subtask_start_signals"].keys()}
    return term, starts


def print_demo_summary(demo_key: str, T: int, term: dict, starts: dict | None, success: bool) -> None:
    suc_tag = "SUCCESS" if success else "FAILURE"
    print(f"\n{demo_key}  ({T} frames, {suc_tag})")
    if not term:
        print("  (no subtask term signals found)")
        return
    for name, sig in term.items():
        idx = first_true(sig)
        if idx is None:
            print(f"  term  {name:24s} → NEVER True   ⚠")
        else:
            frac = idx / max(T - 1, 1)
            bar_w = 40
            bar = "─" * int(bar_w * frac) + "│" + "─" * (bar_w - int(bar_w * frac))
            print(f"  term  {name:24s} → frame {idx:4d}/{T - 1}  ({frac * 100:5.1f}% )  {bar}")
    if starts is not None:
        for name, sig in starts.items():
            idx = first_true(sig)
            if idx is None:
                print(f"  start {name:24s} → NEVER True   ⚠")
            else:
                frac = idx / max(T - 1, 1)
                print(f"  start {name:24s} → frame {idx:4d}/{T - 1}  ({frac * 100:5.1f}% )")


def make_timeline_plot(rows: list[dict], term_signal_names: list[str], output_path: str | None) -> None:
    """Render a stacked-bar timeline of subtask segments, one row per demo.

    Each row's bar is normalised to the demo's length (0..1).  Subtask 0 runs
    from frame 0 up to the first True of the first term signal, subtask 1 from
    there to the first True of the next term signal, etc.  The final subtask
    runs from the last boundary to the end of the episode.
    """
    import matplotlib

    if output_path is not None:
        matplotlib.use("Agg")
    else:
        for backend in ("Qt5Agg", "QtAgg", "TkAgg", "GTK3Agg"):
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as _probe
                _probe.figure()
                _probe.close("all")
                break
            except Exception:
                continue
        else:
            print(
                "No interactive matplotlib backend available; use --save-plot <path> or `pip install PyQt5`.",
                flush=True,
            )
            return
    import matplotlib.pyplot as plt

    n = len(rows)
    palette = ["#3b8ed0", "#d05b3b", "#3bd07b", "#d0c43b", "#8a3bd0", "#3bd0c4"]

    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.25 * n + 1.5)))
    for r_idx, r in enumerate(rows):
        T = r["T"]
        boundaries = r["boundaries"]
        # Build segments: [0, b0], [b0, b1], ..., [bN, T-1]
        seg_starts = [0] + boundaries
        seg_ends = boundaries + [T - 1]
        for s_idx, (lo, hi) in enumerate(zip(seg_starts, seg_ends)):
            color = palette[s_idx % len(palette)]
            ax.barh(r_idx, (hi - lo) / max(T - 1, 1), left=lo / max(T - 1, 1), color=color, edgecolor="black", linewidth=0.4)
        # Per-row label including success
        ax.text(-0.01, r_idx, f"{r['demo_key']} ({'OK' if r['success'] else 'FAIL'})", va="center", ha="right", fontsize=8)

    # Legend
    legend_labels = []
    for i, name in enumerate(term_signal_names + ["(final)"]):
        legend_labels.append((name, palette[i % len(palette)]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in legend_labels]
    ax.legend(handles, [l for l, _ in legend_labels], loc="upper right", fontsize=8, title="subtask segment")

    ax.set_xlim(0, 1)
    ax.set_xlabel("episode progress (normalised)")
    ax.set_yticks([])
    ax.set_title("MimicGen subtask segmentation by demo")
    ax.invert_yaxis()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"\nWrote timeline plot to: {output_path}")
    else:
        print("\n(close the matplotlib window to exit)")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Inspect MimicGen annotations in an Isaac Lab HDF5 dataset.")
    parser.add_argument("hdf5_path", help="Path to the annotated .hdf5 dataset")
    parser.add_argument("--no-plot", action="store_true", help="Skip the matplotlib timeline (stdout only).")
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Save the timeline to this path instead of opening a window (uses Agg backend).",
    )
    args = parser.parse_args()

    if not Path(args.hdf5_path).exists():
        raise SystemExit(f"File not found: {args.hdf5_path}")

    rows: list[dict] = []
    term_signal_names_master: list[str] = []

    with h5py.File(args.hdf5_path, "r") as f:
        data = f["data"]
        demo_keys = get_demo_keys(data)
        if not demo_keys:
            raise SystemExit("No demos under 'data/' in this file.")

        print(f"Dataset: {args.hdf5_path}")
        print(f"Episodes: {len(demo_keys)}")

        for demo_key in demo_keys:
            g = data[demo_key]
            success = bool(g.attrs.get("success", False))
            try:
                term, starts = load_subtask_signals(g)
            except KeyError as e:
                print(f"\n{demo_key}: {e}")
                continue

            T = next(iter(term.values())).shape[0] if term else int(g.attrs.get("num_samples", 0))
            print_demo_summary(demo_key, T, term, starts, success)

            # Build boundary list in subtask-config order (which is dict-insertion order in Python 3.7+).
            term_signal_names = list(term.keys())
            if not term_signal_names_master:
                term_signal_names_master = term_signal_names
            boundaries = []
            for name in term_signal_names_master:
                if name not in term:
                    continue
                idx = first_true(term[name])
                boundaries.append(idx if idx is not None else T - 1)
            rows.append({"demo_key": demo_key, "T": T, "boundaries": boundaries, "success": success})

    if rows and (args.save_plot is not None or not args.no_plot):
        make_timeline_plot(rows, term_signal_names_master, args.save_plot)


if __name__ == "__main__":
    main()
