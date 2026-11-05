"""Experimental: generate takes from one recorded take in new layouts (questcollect/generate.py).

usage: uv run python scripts/generate_demos.py <src.hdf5> <seed> [<seed> ...] [--out DIR] [--identity] [options]
Writes gen_<src>_s<seed>.hdf5 for every take that passes the check; scripts/generate_set.py builds a whole set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from questcollect.generate import generate
from questcollect.sim import SPAWN_JITTER_M, SPAWN_YAW

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--identity", action="store_true", help="replay the source unchanged (checks the replay)")
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--spawn-jitter", type=float, default=SPAWN_JITTER_M, help="m, +- around each SPAWN centre")
    ap.add_argument("--spawn-yaw", type=float, default=float(np.rad2deg(SPAWN_YAW)), help="deg, +- brick start yaw")
    ap.add_argument("--tower-yaw", type=float, default=0.0, help="deg, +- turn of the whole tower")
    ap.add_argument("--tower-frac", type=float, default=1.0, help="share of takes with a turned tower")
    ap.add_argument("--base-xy", type=float, default=0.0, help="m, +- shift of the base in the build area")
    ap.add_argument("--arm-xy", type=float, default=0.0, help="m, +- start TCP offset per arm (z: half)")
    ap.add_argument("--arm-yaw", type=float, default=0.0, help="deg, +- start TCP yaw offset per arm")
    a = ap.parse_args()
    for s in a.seeds:
        r = generate(Path(a.src), s, Path(a.out) if a.out else None, a.identity, keep_failed=a.keep_failed,
                     jitter=a.spawn_jitter, spawn_yaw=np.deg2rad(a.spawn_yaw), tower_yaw=np.deg2rad(a.tower_yaw),
                     tower_frac=a.tower_frac, base_xy=a.base_xy, arm_xy=a.arm_xy, arm_yaw=np.deg2rad(a.arm_yaw))
        print(json.dumps(r), flush=True)
