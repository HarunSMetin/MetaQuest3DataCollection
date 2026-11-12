"""Experimental: off-centre grasps from recorded takes (questcollect/generate.py, `grasp_offset`).

The bricks start shifted along the axis between the finger pads while each hand reaches where the source hand did,
so it grasps them off centre; the place follows the brick through the in-hand correction. Every take is simulated
and kept only if it passes `generate.check`. Teaches what a policy that grasps 1-2 cm off needs: align the brick,
not the hand.

    uv run python scripts/generate_grasps.py data/episodes_retimed_v4 --out data/episodes_grasp_v4 --seeds 6 --jobs 6
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from questcollect.audit import load_curation
from questcollect.generate import generate


def _one(job):
    src, seed, a = job
    try:
        r = generate(Path(src), seed, Path(a.out), layout=False, grasp_offset=a.offset, grasp_across=a.across,
                     grasp_bricks=tuple(a.bricks), min_cam_m=a.min_cam)
        return {k: r.get(k) for k in ("src", "seed", "ok", "progress", "grasp_shift_mm", "arm_m", "cam_m")}
    except Exception as e:  # one bad source shouldn't stop the set
        return {"src": Path(src).name, "seed": seed, "ok": False, "error": repr(e)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=6, help="takes tried per source")
    ap.add_argument("--seed-base", type=int, default=70000)
    ap.add_argument("--offset", type=float, default=0.02, help="m, max off-centre shift along the brick")
    ap.add_argument("--across", type=float, default=0.002, help="m, +- shift across (the fingers re-centre it)")
    ap.add_argument("--bricks", type=int, nargs="+", default=[0, 1], help="brick indices (0 red, 1 blue, ...)")
    ap.add_argument("--min-cam", type=float, default=0.025, help="m, closest wrist cameras accepted")
    ap.add_argument("--jobs", type=int, default=6, help="parallel sims (~1.2 GB each; 14 GB RAM: <= 6)")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    excluded = {f for f, c in load_curation(Path(a.src_dir)).items() if c.get("exclude")}  # `qc audit --exclude`
    srcs = [s for s in sorted(Path(a.src_dir).glob("episode_*.hdf5")) if s.name not in excluded]
    print(f"{len(srcs)} sources ({len(excluded)} excluded by hand: {sorted(excluded)})", flush=True)
    jobs = [(str(s), a.seed_base + 100 * n + k, a) for n, s in enumerate(srcs) for k in range(a.seeds)]
    ok = 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(_one, jobs):
            ok += bool(r.get("ok"))
            print(json.dumps(r), flush=True)
    print(f"{ok}/{len(jobs)} takes kept -> {a.out}")
