"""`qc eval` for the state oracles of this folder, with the (almost) constant brick-pose inputs pinned.

    MUJOCO_GL=egl uv run --extra learn python experiments/obs_ablation/diagnostics/e8_eval_pinned.py \
        outputs/review_oracle2_novel experiments/obs_ablation/config_novel.yaml [--n 50] [--n-action-steps 16]

Why: the `dp_state1` preset min-max normalizes `observation.environment_state`. Bricks 3 and 4 are never touched in
the expert takes, so their z and two quaternion components have a range of 0 (or float noise). Min-max divides by
that range: while the bricks settle in the first steps of an eval episode (which the trimmed takes never show) the
normalized z is 11 and 23 instead of -1, and the first chunk is garbage (every episode: base pushed out, safety
filter correcting by 1.5-2 rad). Here every input whose range in the dataset is below `PIN_BELOW` is replaced by the
middle of that range, which is what the network saw in training. No retraining needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from questcollect.config import load_config  # noqa: E402
from questcollect.learn import policy as policy_mod  # noqa: E402
from questcollect.learn.evaluate import evaluate  # noqa: E402

PIN_BELOW = 1e-4


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-action-steps", type=int, default=None)
    args = ap.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "questcollect_run.json").read_text())
    stats = json.loads((Path(manifest["dataset_root"]) / "meta/stats.json").read_text())
    pins = {}
    for key in ("observation.state", "observation.environment_state"):
        if key in stats:
            lo, hi = np.array(stats[key]["min"]), np.array(stats[key]["max"])
            idx = np.flatnonzero(hi - lo < PIN_BELOW)
            pins[key] = (torch.from_numpy(idx), torch.from_numpy(((lo + hi) / 2)[idx].astype(np.float32)))
            print(f"{key}: pinning {len(idx)} of {len(lo)} inputs {idx.tolist()}")

    observation = policy_mod.PolicyRunner.observation

    def pinned(self, env):
        batch = observation(self, env)
        for key, (idx, value) in pins.items():
            batch[key][0, idx] = value
        return batch

    policy_mod.PolicyRunner.observation = pinned
    kwargs = {} if args.n_action_steps is None else {"n_action_steps": args.n_action_steps}
    summary, out = evaluate(run, load_config(args.config), n=args.n, **kwargs)
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
