"""E9: how close does the carried blue brick get to the red one (xy, mm) in a folder of rollouts or takes?

    uv run python experiments/obs_ablation/diagnostics/e9_approach.py outputs/<run>/eval/<time> [more folders]

A graded measure for policies that never click level 1. The expert takes: 50/50 within 10 mm, median 8 mm.
"""
import glob
import sys

import numpy as np

sys.path.insert(0, ".")
from questcollect.recorder import read_episode  # noqa: E402

for folder in sys.argv[1:]:
    closest, no_release = [], 0
    for f in sorted(glob.glob(f"{folder}/episode_*.hdf5")):
        _, d = read_episode(f, keys=["obs/objects", "action/joint_pos"])
        ob, grip_left = d["obs/objects"], d["action/joint_pos"][:, 7]
        closed = np.flatnonzero(grip_left < 0.5)
        no_release += not (len(closed) > 0 and (grip_left[closed[0]:] > 0.5).any())
        dist = np.linalg.norm(ob[:, 1, :2] - ob[:, 0, :2], axis=1) * 1000
        held = ob[:, 1, 2] > ob[0, 1, 2] + 0.03
        closest.append(dist[held].min() if held.any() else np.nan)
    c = np.array(closest)
    print(f"{folder}: {len(c)} episodes; blue lifted {(~np.isnan(c)).sum()}; within 30 mm of red {(c < 30).sum()}, "
          f"within 10 mm {(c < 10).sum()}; median closest {np.nanmedian(c):.0f} mm; left never releases red {no_release}")
