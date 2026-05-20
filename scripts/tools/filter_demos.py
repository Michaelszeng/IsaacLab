"""Filter demos out of an Isaac Lab HDF5 dataset.

Use cases:
  1. Drop demos whose subtask term signal never fires (annotation never found a
     boundary — MimicGen would silently skip them).
  2. Drop demos whose subtask boundary is outside a given percentile range
     (outliers that hurt MimicGen consistency).
  3. Drop explicit demos by name (after eyeballing inspect_annotations.py).

The output file is a fresh HDF5 with demos renumbered demo_0, demo_1, ... in the
same order they appeared in the source.  Top-level attrs (env_args, total) are
copied/updated.  All sub-groups inside each kept demo (obs/, actions, states,
datagen_info, etc.) are deep-copied as-is.

Usage:
    # Drop demos that never grasped:
    python scripts/tools/filter_demos.py \\
        ./datasets/annotated.hdf5 ./datasets/annotated_clean.hdf5 \\
        --drop-no-signal

    # Drop specific demos by name:
    python scripts/tools/filter_demos.py \\
        ./datasets/annotated.hdf5 ./datasets/annotated_clean.hdf5 \\
        --drop demo_3 demo_7

    # Drop demos with boundary outside the [10%, 85%] band:
    python scripts/tools/filter_demos.py \\
        ./datasets/annotated.hdf5 ./datasets/annotated_clean.hdf5 \\
        --drop-boundary-outside 10 85

    # Combine multiple filters:
    python scripts/tools/filter_demos.py \\
        ./datasets/annotated.hdf5 ./datasets/annotated_clean.hdf5 \\
        --drop-no-signal --drop-boundary-outside 10 85
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
    idx = int(np.argmax(arr))
    return idx if bool(arr[idx]) else None


def collect_drop_reasons(g: h5py.Group, args) -> list[str]:
    """Return a list of reasons this demo should be dropped (empty = keep)."""
    reasons = []

    # 1. Explicit drop list.
    name = g.name.rsplit("/", 1)[-1]
    if args.drop and name in args.drop:
        reasons.append("in --drop list")

    # 2. Subtask signal never fires.
    if args.drop_no_signal or args.drop_boundary_outside is not None:
        if "obs/datagen_info" not in g:
            reasons.append("missing obs/datagen_info (not annotated)")
        else:
            term_group = g["obs/datagen_info"].get("subtask_term_signals")
            if term_group is None:
                reasons.append("no subtask_term_signals in datagen_info")
            else:
                for sig_name in term_group.keys():
                    sig = np.asarray(term_group[sig_name]).astype(bool)
                    T = len(sig)
                    idx = first_true(sig)
                    if idx is None and args.drop_no_signal:
                        reasons.append(f"term signal '{sig_name}' never True")
                    elif idx is not None and args.drop_boundary_outside is not None:
                        lo, hi = args.drop_boundary_outside
                        pct = 100.0 * idx / max(T - 1, 1)
                        if pct < lo or pct > hi:
                            reasons.append(
                                f"term signal '{sig_name}' boundary at {pct:.1f}% (outside [{lo}, {hi}]%)"
                            )

    # 3. Failed demos.
    if args.drop_failures and not bool(g.attrs.get("success", True)):
        reasons.append("success=False")

    return reasons


def main():
    parser = argparse.ArgumentParser(description="Filter demos out of an Isaac Lab HDF5 dataset.")
    parser.add_argument("input_file", help="Path to source .hdf5")
    parser.add_argument("output_file", help="Path for filtered .hdf5")
    parser.add_argument(
        "--drop", nargs="*", default=[], metavar="DEMO_KEY",
        help="Explicit demo names to drop (e.g. demo_3 demo_7).",
    )
    parser.add_argument(
        "--drop-no-signal", action="store_true",
        help="Drop demos whose subtask term signals never fire (no valid boundary).",
    )
    parser.add_argument(
        "--drop-boundary-outside", nargs=2, type=float, default=None, metavar=("LO", "HI"),
        help="Drop demos whose subtask boundary is outside [LO, HI] percent of episode length.",
    )
    parser.add_argument(
        "--drop-failures", action="store_true",
        help="Drop demos with success=False.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be kept/dropped without writing the output file.",
    )
    args = parser.parse_args()

    if not Path(args.input_file).exists():
        raise SystemExit(f"Input not found: {args.input_file}")
    if Path(args.output_file).exists() and not args.dry_run:
        raise SystemExit(f"Output already exists, refusing to overwrite: {args.output_file}")

    with h5py.File(args.input_file, "r") as fin:
        if "data" not in fin:
            raise SystemExit(f"{args.input_file}: no 'data' group at the root.")
        demo_keys = get_demo_keys(fin["data"])
        if not demo_keys:
            raise SystemExit("No demos in source file.")

        kept: list[tuple[str, int]] = []  # (original_name, num_samples)
        dropped: list[tuple[str, list[str]]] = []
        total_samples_kept = 0
        for demo_key in demo_keys:
            g = fin[f"data/{demo_key}"]
            reasons = collect_drop_reasons(g, args)
            if reasons:
                dropped.append((demo_key, reasons))
            else:
                ns = int(g.attrs.get("num_samples", 0))
                kept.append((demo_key, ns))
                total_samples_kept += ns

        # --- Report ---
        print(f"Source: {args.input_file}")
        print(f"  Total demos: {len(demo_keys)}")
        print(f"  Keeping:     {len(kept)}")
        print(f"  Dropping:    {len(dropped)}")
        if dropped:
            print("\nDropped demos:")
            for name, reasons in dropped:
                print(f"  {name}: {'; '.join(reasons)}")

        if args.dry_run:
            print("\n(--dry-run, no output file written)")
            return

        if not kept:
            raise SystemExit("All demos would be dropped — refusing to write empty output.")

        # --- Copy ---
        print(f"\nWriting {len(kept)} demos to: {args.output_file}")
        with h5py.File(args.output_file, "w") as fout:
            out_data = fout.create_group("data")
            # Copy top-level attrs (e.g. env_args).  Keep total in sync.
            for k, v in fin["data"].attrs.items():
                out_data.attrs[k] = v
            out_data.attrs["total"] = total_samples_kept

            for new_idx, (orig_name, _) in enumerate(kept):
                new_name = f"demo_{new_idx}"
                fin.copy(f"data/{orig_name}", out_data, name=new_name)

        print("Done.")


if __name__ == "__main__":
    main()
