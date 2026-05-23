"""Convert an Isaac Lab HDF5 demo dataset into the diffusion-policy zarr format.

Layout produced:

    output.zarr/
    ├── data/
    │   ├── <obs_key>           (T_total, D)              float (scalar obs)
    │   ├── <camera_key>        (T_total, H, W, 3)        uint8 (source resolution)
    │   └── actions             (T_total, A)              float
    └── meta/
        └── episode_ends        (N_episodes,)             int64 (cumulative)

Each top-level key in the source HDF5's per-demo ``obs/`` group becomes a flat
``data/<key>`` array with all demos concatenated end-to-end.  Cameras are
detected by their (T, H, W, 3/4) shape and copied at source resolution (any
4-channel RGBA frames have the alpha channel dropped).  ``meta/episode_ends``
gives the cumulative frame count at the end of each demo (matching the
diffusion-policy convention).

Usage:
    # Default — float64 scalars, native-resolution cameras.
    python scripts/tools/hdf5_to_zarr.py \\
        ./datasets/gear_assembly_generated.hdf5 \\
        ./datasets/gear_assembly_generated.zarr

    # float32 scalars (half the storage cost):
    python scripts/tools/hdf5_to_zarr.py \\
        ./datasets/gear_assembly_generated.hdf5 \\
        ./datasets/gear_assembly_generated.zarr \\
        --dtype float32

    # Include only some cameras:
    python scripts/tools/hdf5_to_zarr.py \\
        ./datasets/gear_assembly_generated.hdf5 \\
        ./datasets/gear_assembly_generated.zarr \\
        --cameras wrist_cam scene_cam_front

    # Drop failed demos before conversion:
    python scripts/tools/hdf5_to_zarr.py \\
        ./datasets/gear_assembly_generated.hdf5 \\
        ./datasets/gear_assembly_generated.zarr \\
        --drop-failures
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

try:
    import zarr
    from numcodecs import Blosc
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Install with:\n"
        "  pip install zarr numcodecs"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_camera_dataset(item) -> bool:
    """Return True if `item` looks like a camera array: (T, H, W, 3|4)."""
    return (
        hasattr(item, "shape")
        and len(item.shape) == 4
        and item.shape[-1] in (3, 4)
    )


def get_demo_keys(group: h5py.Group) -> list[str]:
    keys = [k for k in group.keys() if k.startswith("demo_")]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def resolve_dtype(arg_dtype: str, source_dtype) -> np.dtype:
    if arg_dtype == "preserve":
        return np.dtype(source_dtype)
    return np.dtype(arg_dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert Isaac Lab HDF5 → diffusion-policy zarr.")
    parser.add_argument("input", help="Path to source .hdf5")
    parser.add_argument("output", help="Path for output .zarr (must not exist)")
    parser.add_argument(
        "--cameras", nargs="+", default=None,
        help="Restrict to these camera obs keys (default: auto-detect all).",
    )
    parser.add_argument(
        "--scalar-keys", nargs="+", default=None,
        help="Restrict scalar obs keys (default: all non-camera datasets in obs/).",
    )
    parser.add_argument(
        "--action-key", default="actions",
        help="Top-level demo dataset for actions (default: 'actions'). Use 'none' to skip.",
    )
    parser.add_argument(
        "--dtype", default="float64", choices=["float64", "float32", "preserve"],
        help="Dtype for non-camera arrays (default: float64 to match the example zarr).",
    )
    parser.add_argument(
        "--drop-failures", action="store_true",
        help="Skip demos with success=False.",
    )
    parser.add_argument(
        "--compression-level", type=int, default=5,
        help="Blosc compression level 0-9 (default: 5).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow overwriting an existing output zarr (deletes it first).",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise SystemExit(f"Input not found: {args.input}")
    if Path(args.output).exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists, refusing to overwrite without --overwrite: {args.output}")
        import shutil
        shutil.rmtree(args.output)

    with h5py.File(args.input, "r") as fin:
        if "data" not in fin:
            raise SystemExit(f"{args.input}: no 'data' group at root.")
        data_in = fin["data"]
        demo_keys = get_demo_keys(data_in)

        if args.drop_failures:
            kept = [k for k in demo_keys if bool(data_in[k].attrs.get("success", False))]
            print(f"--drop-failures: keeping {len(kept)}/{len(demo_keys)} successful demos")
            demo_keys = kept

        if not demo_keys:
            raise SystemExit("No demos to convert.")

        # ---- Discover keys from first demo ----
        first = data_in[demo_keys[0]]
        if "obs" not in first:
            raise SystemExit(f"First demo missing 'obs/' group — is this an Isaac Lab demo file?")
        first_obs = first["obs"]

        all_cam_keys, all_scalar_keys = [], []
        for k in first_obs.keys():
            item = first_obs[k]
            if isinstance(item, h5py.Group):
                continue  # skip nested groups like subtask_terms
            if is_camera_dataset(item):
                all_cam_keys.append(k)
            else:
                all_scalar_keys.append(k)

        cam_keys = args.cameras if args.cameras is not None else all_cam_keys
        scalar_keys = args.scalar_keys if args.scalar_keys is not None else all_scalar_keys
        for k in cam_keys:
            if k not in all_cam_keys:
                raise SystemExit(f"Requested camera '{k}' not found in obs/. Available: {all_cam_keys}")
        for k in scalar_keys:
            if k not in all_scalar_keys:
                raise SystemExit(f"Requested scalar key '{k}' not found in obs/. Available: {all_scalar_keys}")

        has_action = args.action_key != "none" and args.action_key in first

        # If the action key also exists under obs/ (e.g. IsaacLab records
        # `obs/actions` as the previous action observed by the policy AND
        # top-level `actions` as the action taken this step), prefer the
        # top-level one and drop the obs duplicate to avoid creating the
        # same zarr dataset twice.
        action_obs_dropped = False
        if has_action and args.action_key in scalar_keys:
            scalar_keys = [k for k in scalar_keys if k != args.action_key]
            action_obs_dropped = True

        # ---- Per-demo bookkeeping ----
        demo_lengths = [int(data_in[k].attrs.get("num_samples", 0)) for k in demo_keys]
        # Fall back to actual length if num_samples attr missing.
        for i, k in enumerate(demo_keys):
            if demo_lengths[i] == 0 and has_action:
                demo_lengths[i] = data_in[k][args.action_key].shape[0]
            elif demo_lengths[i] == 0 and scalar_keys:
                demo_lengths[i] = data_in[k][f"obs/{scalar_keys[0]}"].shape[0]
        total = int(sum(demo_lengths))
        episode_ends = np.cumsum(demo_lengths).astype(np.int64)

        print(f"\nInput:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Demos:  {len(demo_keys)}  Total frames: {total}")
        print(f"Scalar obs keys ({len(scalar_keys)}): {scalar_keys}")
        print(f"Cameras ({len(cam_keys)}):           {cam_keys}")
        print(f"Action key: {args.action_key if has_action else '(skipped)'}"
              + ("  (also present under obs/; using top-level dataset)" if action_obs_dropped else ""))
        print("Image size: source resolution (no resize)")
        print(f"Scalar dtype: {args.dtype}")
        print()

        # ---- Create output zarr ----
        # Explicit zarr_format=2 so the output is compatible with the existing
        # diffusion-policy training pipeline (which reads v2-format zarrs).
        # zarr v3 defaults to writing v3-format stores otherwise and rejects
        # numcodecs.Blosc as a non-BytesBytesCodec.
        out = zarr.open_group(args.output, mode="w", zarr_format=2)
        out_data = out.create_group("data")
        out_meta = out.create_group("meta")
        compressor = Blosc(cname="lz4", clevel=args.compression_level, shuffle=Blosc.SHUFFLE)

        # Pre-allocate per-key arrays.
        out_arrays = {}

        def make_dataset(name, shape, dtype, chunks):
            # Use the modern create_array API.  `compressors=<codec>` is the
            # non-deprecated spelling; for zarr_format=2 a single codec is used
            # as the v2 `compressor` field.
            return out_data.create_array(
                name=name, shape=shape, dtype=dtype, chunks=chunks, compressors=compressor
            )

        # Scalar obs
        for k in scalar_keys:
            sample = np.asarray(first_obs[k][0:1])
            shape = (total,) + sample.shape[1:]
            dtype = resolve_dtype(args.dtype, sample.dtype)
            chunks = (min(1024, max(total, 1)),) + sample.shape[1:]
            out_arrays[k] = make_dataset(k, shape=shape, dtype=dtype, chunks=chunks)

        # Cameras — native resolution, RGBA stripped to RGB if needed.
        for k in cam_keys:
            sample = first_obs[k]
            H, W = sample.shape[1], sample.shape[2]
            shape = (total, H, W, 3)
            chunks = (min(128, max(total, 1)), H, W, 3)
            out_arrays[k] = make_dataset(k, shape=shape, dtype=np.uint8, chunks=chunks)

        # Actions
        if has_action:
            sample = np.asarray(first[args.action_key][0:1])
            shape = (total,) + sample.shape[1:]
            dtype = resolve_dtype(args.dtype, sample.dtype)
            chunks = (min(1024, max(total, 1)),) + sample.shape[1:]
            out_arrays[args.action_key] = make_dataset(args.action_key, shape=shape, dtype=dtype, chunks=chunks)

        # ---- Stream each demo ----
        offset = 0
        for i, demo_key in enumerate(demo_keys):
            g = data_in[demo_key]
            T = demo_lengths[i]

            for k in scalar_keys:
                arr = np.asarray(g[f"obs/{k}"])
                if args.dtype != "preserve":
                    arr = arr.astype(out_arrays[k].dtype)
                out_arrays[k][offset : offset + T] = arr

            for k in cam_keys:
                src = np.asarray(g[f"obs/{k}"])
                if src.shape[-1] == 4:
                    src = src[..., :3]
                out_arrays[k][offset : offset + T] = src.astype(np.uint8)

            if has_action:
                arr = np.asarray(g[args.action_key])
                if args.dtype != "preserve":
                    arr = arr.astype(out_arrays[args.action_key].dtype)
                out_arrays[args.action_key][offset : offset + T] = arr

            offset += T
            print(f"  [{i + 1:4d}/{len(demo_keys)}] {demo_key}: {T:4d} frames  → zarr[{offset - T}:{offset}]")

        assert offset == total, f"offset {offset} != total {total}"

        # ---- meta/episode_ends ----
        episode_ends_arr = out_meta.create_array(
            name="episode_ends",
            shape=episode_ends.shape,
            chunks=(min(512, max(len(episode_ends), 1)),),
            dtype=np.int64,
            compressors=compressor,
        )
        episode_ends_arr[:] = episode_ends

        # Stash a small summary as an attribute for debugging.
        out.attrs["source_hdf5"] = args.input
        out.attrs["num_episodes"] = len(demo_keys)
        out.attrs["total_frames"] = int(total)

    print(f"\n✓ Wrote zarr to: {args.output}")
    print(f"  {len(demo_keys)} episodes, {total} total frames")
    print(f"  episode_ends[:5] = {episode_ends[:5].tolist()}{'...' if len(episode_ends) > 5 else ''}")


if __name__ == "__main__":
    main()
