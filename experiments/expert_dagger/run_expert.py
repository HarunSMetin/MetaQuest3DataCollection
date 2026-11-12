"""Run the scripted expert (expert.py) on many scenes: test it, or record its takes as episodes.

    cd experiments/expert_dagger
    MUJOCO_GL=egl uv run python run_expert.py --eval-seeds 50            # success rate on the policy eval scenes
    MUJOCO_GL=egl uv run python run_expert.py --n 200 --out ../../data/expert_v2   # record 200 takes (other seeds)

Recorded takes are ordinary episodes (episode_*.hdf5, recorder.write_episode), only successful ones are written.
Seeds for recording come from their own stream (EXPERT_SEED_BASE), never the eval seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
import multiprocessing as mp
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

EXPERT_SEED_BASE = 31_000_000
CONFIG = HERE / "config.yaml"


def job(args):
    seed, out, index = args
    from expert import run_episode

    from questcollect.config import load_config
    from questcollect.recorder import episode_meta, write_episode

    if out is not None and (Path(out) / f"episode_{index:06d}.hdf5").exists():
        return {"seed": seed, "clicked": True, "click_s": None, "phases": {}, "failed": None, "skipped": True}
    cfg = load_config(CONFIG)
    info, steps, env = run_episode(cfg, seed, record=out is not None)
    if out is not None and info["clicked"] and not info["failed"]:
        path = Path(out) / f"episode_{index:06d}.hdf5"
        meta = episode_meta(cfg, env, len(steps), env.success(), input_mode="expert", operator="expert:v1",
                            source="expert")
        write_episode(path, steps, meta)
        info["file"] = path.name
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-seeds", type=int, default=0, help="test on the first N policy eval seeds")
    ap.add_argument("--n", type=int, default=0, help="record this many takes (seeds from EXPERT_SEED_BASE)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=int, default=0, help="first seed index (to add more takes later)")
    ap.add_argument("--jobs", type=int, default=3)  # 6 ran the 14 GB laptop out of memory
    args = ap.parse_args()
    if args.eval_seeds:
        from questcollect.learn.evaluate import eval_seeds  # numpy only, no torch needed
        tasks = [(s, None, i) for i, s in enumerate(eval_seeds(args.eval_seeds))]
    else:
        seeds = np.random.SeedSequence(EXPERT_SEED_BASE).generate_state(args.start + args.n, dtype=np.uint64)
        tasks = [(int(s % 2**31), args.out, i) for i, s in enumerate(seeds)][args.start:]
        Path(args.out).mkdir(parents=True, exist_ok=True)
    infos = []
    # fresh worker processes every few takes, so memory can't creep up (6 jobs once ran the laptop out of memory)
    with mp.get_context("spawn").Pool(args.jobs, maxtasksperchild=4) as pool:
        for info in pool.imap(job, tasks):
            infos.append(info)
            print(f"[{len(infos)}/{len(tasks)}] seed {info['seed']:>10}  {'clicked' if info['clicked'] else 'FAILED '}"
                  f"  {'' if info['click_s'] is None else format(info['click_s'], '.1f') + ' s'}  phases {info['phases']}"
                  f"  {info['failed'] or ''}", flush=True)
    infos = [i for i in infos if not i.get("skipped")]
    ok = [i for i in infos if i["clicked"]]
    t = [i["click_s"] for i in ok]
    if not infos:
        return
    print(f"\nlevel 1 clicked: {len(ok)}/{len(infos)} ({100 * len(ok) / len(infos):.0f}%)"
          + (f"; time to click median {np.median(t):.1f} s ({min(t):.1f}-{max(t):.1f})" if t else ""))
    if args.out:
        (Path(args.out) / "expert_runs.json").write_text(json.dumps(infos, indent=1, default=float))


if __name__ == "__main__":
    main()
