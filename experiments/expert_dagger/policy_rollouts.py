"""Policy rollouts on fresh layouts, saved as episodes to rewind and correct in the headset (offline DAgger).

    MUJOCO_GL=egl uv run --extra learn python experiments/expert_dagger/policy_rollouts.py \
        outputs/20261111-1316_dp_240_pre_expert_v2 data/dagger_v1_policy [--n 50] [--first 0]

Same rollout as `qc eval` (safety filter, chunk length as trained), but on seeds from their own stream, never the
eval seeds, so corrections recorded on them don't leak into the evaluation. The folder shows in the dashboard's
Dataset menu; pick a rollout, pause where it starts to go wrong, "Take over here": the take goes to the
recording folder (record.out_dir of the config you serve with, data/dagger_v1).
`--first` continues the seed stream (e.g. --first 50 for the next 50 layouts).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from questcollect.config import load_config  # noqa: E402
from questcollect.learn.evaluate import eval_seeds, max_steps_for, rollout  # noqa: E402
from questcollect.learn.policy import PolicyRunner  # noqa: E402
from questcollect.recorder import episode_meta, write_episode  # noqa: E402
from questcollect.safety import SafetyFilter  # noqa: E402
from questcollect.sim import BimanualEnv  # noqa: E402

DAGGER_SEED_BASE = 20_261_114  # its own stream: not the eval seeds (EVAL_SEED_BASE), not random recording seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--first", type=int, default=0, help="index of the first seed in the stream")
    ap.add_argument("--config", default=str(ROOT / "experiments/expert_dagger/config_dagger.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = [int(s % 2**31) for s in np.random.SeedSequence(DAGGER_SEED_BASE).generate_state(
        args.first + args.n, dtype=np.uint64)][args.first:]
    assert not set(seeds) & set(eval_seeds(1000)), "seed overlaps the eval seeds"

    env = BimanualEnv(cfg)
    runner = PolicyRunner(Path(args.run), cfg)
    safety = SafetyFilter(cfg.sim, cfg.ik, cfg.safety, env.control_dt)
    max_steps = max_steps_for(runner.run, round(1 / env.control_dt))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset.json").write_text(json.dumps({
        "title": "DAgger: policy rollouts",
        "description": f"{runner.run.name} on fresh layouts (not eval seeds). Rewind to where it goes wrong and "
                       "Take over here; takes go to data/dagger_v1."}, indent=1))
    for i, seed in enumerate(seeds, start=args.first):
        r, steps = rollout(env, runner, safety, seed, max_steps, record=True)
        meta = episode_meta(cfg, env, len(steps), r.success, input_mode="policy",
                            operator=f"policy:{runner.run.name}", source="policy")
        write_episode(out / f"episode_{i:06d}.hdf5", steps, meta)
        print(f"[{i}] seed {seed}  base {'in' if r.base_in_area else 'out'}  levels {r.max_progress}", flush=True)
    runner.close()


if __name__ == "__main__":
    main()
