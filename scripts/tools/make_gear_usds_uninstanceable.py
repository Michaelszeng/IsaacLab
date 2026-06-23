# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""One-time generator: produce local, non-instanceable copies of the four
Factory gear USD files used by the gear-assembly task.

WHY THIS EXISTS
---------------
The Factory gear USDs (``factory_gear_large/small/medium/base.usd``) ship
with ``instanceable=True`` on the ``visuals``/``collisions`` subprims (a
Nucleus optimisation for many-instance scenes). USD does not allow
material binding on instance proxies, so the ``bind_slippery_gear_material``
startup event in ``gear_assembly_env_cfg.py`` silently no-ops on every
gear and every env, leaving the gears on PhysX defaults (friction=0.5,
rigid contact) instead of the intended slippery + compliant material.

Uninstancing at startup-event time was tried first and broke things --
the SetInstanceable call mutates the stage after IsaacLab has already
created its tensor view, invalidating the view ("Simulation view object
is invalidated and cannot be used again to call setTransforms"). The
only robust fix is to uninstance the USDs *before* the stage is ever
loaded into IsaacLab. That's what this script does: it generates
modified copies on disk, once. We commit those copies; eval / slurm
pulls them with the rest of the repo.

WORKFLOW
--------
1. Run this script once, on a machine with Nucleus access::

      ./isaaclab.sh -p scripts/tools/make_gear_usds_uninstanceable.py

2. Commit the generated files in
   ``source/.../gear_assembly/assets/`` to the repo.

3. SLURM jobs (and any other consumer) automatically use the local
   copies because ``gear_assembly_env_cfg.py`` already points
   ``usd_path=`` at them.

External asset references (textures) inside the flattened USDs remain
as absolute Nucleus URIs. The cluster still needs Nucleus access for
those, but the USD scene-graph structure -- including the modified
``instanceable`` flags -- is baked in.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Force headless: we don't need a window for USD layer ops.
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Imports below must come after AppLauncher: pxr is shipped inside Isaac Sim.
# ---------------------------------------------------------------------------
from pxr import Usd  # noqa: E402

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

GEAR_NAMES = [
    "factory_gear_large",
    "factory_gear_small",
    "factory_gear_medium",
    "factory_gear_base",
]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_OUTPUT_DIR = os.path.join(
    _REPO_ROOT,
    "source",
    "isaaclab_tasks",
    "isaaclab_tasks",
    "manager_based",
    "manipulation",
    "gear_assembly",
    "assets",
)


def _uninstance_all(stage: Usd.Stage) -> int:
    """Walk the entire stage, including into instance proxies, and turn off
    ``instanceable`` on every instance prim. Returns the number changed.

    Mirrors :func:`isaaclab.sim.utils.prims.make_uninstanceable` but operates
    on every prim in the stage (we want all four gears made fully
    uninstanced, not a single root).
    """
    n_changed = 0
    to_visit = list(stage.GetPseudoRoot().GetChildren())
    while to_visit:
        prim = to_visit.pop(0)
        if prim.IsInstance():
            prim.SetInstanceable(False)
            n_changed += 1
        # GetFilteredChildren with TraverseInstanceProxies descends into the
        # prototypes that instance prims point at, so we actually reach the
        # inner visuals/collisions that hold the instanceable flag.
        to_visit.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
    return n_changed


def main() -> None:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {_OUTPUT_DIR}\n")

    for gear_name in GEAR_NAMES:
        src = f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets/{gear_name}/{gear_name}.usd"
        dst = os.path.join(_OUTPUT_DIR, f"{gear_name}.usd")

        print(f"[{gear_name}] opening {src}")
        stage = Usd.Stage.Open(src)
        if stage is None or not stage.GetPseudoRoot().IsValid():
            print(f"[{gear_name}] FAILED to open; skipping.\n")
            continue

        n = _uninstance_all(stage)
        print(f"[{gear_name}] uninstanced {n} prim(s)")

        # Flatten composition into a single self-contained layer so the
        # output doesn't depend on the original USD's reference graph.
        # External asset paths (textures) become absolute Nucleus URIs.
        flat_layer = stage.Flatten()
        flat_layer.Export(dst)
        size_kib = os.path.getsize(dst) / 1024.0
        print(f"[{gear_name}] wrote {dst} ({size_kib:.1f} KiB)\n")

    print("Done. Commit the .usd files in:")
    print(f"  {_OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
