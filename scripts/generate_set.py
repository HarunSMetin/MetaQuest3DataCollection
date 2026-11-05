"""Experimental: a balanced generated set, `per` kept takes from each source demo (questcollect/generate.py).

usage: uv run python scripts/generate_set.py <out_dir> [--per 10] [--max-tries 30] [--jobs 6] [--seed-base N] [generate options]
Writes episode_XXXXXX.hdf5 (qc audit / export read those) and names.txt (episode -> source, seed, layout).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from questcollect.generate import generate
from questcollect.recorder import episode_files

SEED_BASE = 10_000  # far from the eval seeds (EVAL_SEED_BASE); each source gets its own block of 1000 seeds


def _one(src, seed, tmp, kw):
    return generate(Path(src), seed, Path(tmp), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--src", default="data/episodes")
    ap.add_argument("--per", type=int, default=10)
    ap.add_argument("--max-tries", type=int, default=30)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE, help="another value gives other layouts (top-ups)")
    ap.add_argument("--jobs", type=int, default=6)  # ~0.8 GB each; 14 GB laptop
    ap.add_argument("--spawn-jitter", type=float, default=0.06)
    ap.add_argument("--spawn-yaw", type=float, default=25.0)
    ap.add_argument("--tower-yaw", type=float, default=20.0)
    ap.add_argument("--tower-frac", type=float, default=0.5)
    ap.add_argument("--base-xy", type=float, default=0.01)
    ap.add_argument("--arm-xy", type=float, default=0.05)
    ap.add_argument("--arm-yaw", type=float, default=15.0)
    a = ap.parse_args()
    out, tmp = Path(a.out), Path(a.out) / "_tmp"
    kw = dict(jitter=a.spawn_jitter, spawn_yaw=np.deg2rad(a.spawn_yaw), tower_yaw=np.deg2rad(a.tower_yaw),
              tower_frac=a.tower_frac, base_xy=a.base_xy, arm_xy=a.arm_xy, arm_yaw=np.deg2rad(a.arm_yaw))
    sources = episode_files(Path(a.src))
    kept = {s.name: [] for s in sources}
    tries = dict.fromkeys(kept, 0)
    # Seeds go in rounds so every source keeps a few jobs in flight and stops once it has `per`.
    with ProcessPoolExecutor(a.jobs) as pool:
        futs = {}

        def submit(src):
            if len(kept[src.name]) + sum(f_src == src.name for f_src, _ in futs.values()) >= a.per \
                    or tries[src.name] >= a.max_tries:
                return
            seed = a.seed_base + sources.index(src) * 1000 + tries[src.name]
            tries[src.name] += 1
            futs[pool.submit(_one, str(src), seed, str(tmp), kw)] = (src.name, seed)

        for _ in range(2):
            for s in sources:
                submit(s)
        while futs:
            f = next(as_completed(futs))
            name, _ = futs.pop(f)
            r = f.result()
            if r["ok"] and len(kept[name]) < a.per:
                kept[name].append(r)
            elif r.get("path"):
                Path(r["path"]).unlink()
            print(f"{name}: {len(kept[name])}/{a.per} kept, {tries[name]} tried", flush=True)
            submit(next(s for s in sources if s.name == name))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in (r for s in sources for r in kept[s.name]):
        dst = out / f"episode_{len(rows):06d}.hdf5"
        Path(r["path"]).rename(dst)
        rows.append({"episode": dst.name, **{k: r[k] for k in ("src", "seed", "tower_yaw_deg", "base_shift_mm",
                                                               "len_s", "arm_m", "cam_m")}})
    for p in tmp.glob("*.hdf5"):
        p.unlink()
    tmp.rmdir()
    (out / "names.txt").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"kept": len(rows), "tried": sum(tries.values()),
                      "per_source": {k: len(v) for k, v in kept.items()}, "options": vars(a)}))


if __name__ == "__main__":
    main()
