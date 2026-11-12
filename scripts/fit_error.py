"""How precisely does a policy reproduce its demos? Open-loop action error, in millimetres at the TCP.

    MUJOCO_GL=egl uv run --extra learn python scripts/fit_error.py <run or checkpoint> <episodes dir> [--n 15]

For sampled frames t of each episode: put the simulation in the recorded state sim/state[t], let the policy predict
a fresh chunk of n_action_steps targets, and compare them with the targets the demo actually executed at t..t+k
(executed labels, as in training), converted to TCP positions by forward kinematics. Frames are grouped by phase:
`grasp` = the last 0.6 s before a brick lifts off, `place` = the last 0.6 s before a level clicks, `other`.

Two trivial predictors are scored on the same frames, because a small error alone says little: the arm's motion is
smooth, so the next 0.3 s of targets follow from the current motion without knowing the task.
  `coast`  = measured joints + LAG_S x measured joint velocity, continued at that velocity (in the demos the target
             leads the measured joints by ~0.12 s of motion: R^2 0.96-0.99, experiments/REVIEW_2026-11-12.md)
  `repeat` = the previous step's target, held
A policy only shows that it knows where to go where it beats `coast`; in a slow phase (the last descent onto the
studs) `coast` is at 0.1-0.5 mm and the open-loop error cannot tell a good policy from a bad one. What the closed
loop does with the error is measured by scripts/drift_test.py.

Large error already here = the model does not fit the demos to the millimetre (fix: model / action representation /
more epochs). Small error here and failing rollouts = look at drift_test.py before concluding "covariate shift".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from questcollect.config import load_config  # noqa: E402
from questcollect.learn.policy import PolicyRunner  # noqa: E402
from questcollect.recorder import read_episode  # noqa: E402
from questcollect.sim import executed_actions, env_from_episode  # noqa: E402

WINDOW = 30  # 0.6 s at 50 Hz
LAG_S = 0.12  # the joint targets lead the measured joints by this much motion (PD tracking lag, measured)


def phases(data, stop):
    z = data["obs/objects"][:, :, 2]
    events = {"grasp": [], "place": []}
    for b in range(z.shape[1]):
        up = np.flatnonzero(z[:stop, b] > z[0, b] + 0.01)
        if len(up):
            events["grasp"].append(int(up[0]))
    prog = data["progress"][:stop]
    for lvl in range(1, int(prog.max()) + 1):
        events["place"].append(int(np.argmax(prog >= lvl)))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("episodes")
    ap.add_argument("--n", type=int, default=15, help="episodes")
    ap.add_argument("--every", type=int, default=40, help="also sample every this many steps")
    ap.add_argument("--level", type=int, default=1, help="stop each episode 1 s after this level clicked")
    ap.add_argument("--glob", default="episode_*.hdf5")
    ap.add_argument("--config", default=str(ROOT / "experiments/two_bricks/config.yaml"))
    ap.add_argument("--steps", type=int, default=None, help="compare this many steps of each predicted chunk "
                                                            "(default: n_action_steps as trained)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    runner = PolicyRunner(Path(args.run), cfg, n_action_steps=args.steps)
    k = runner.policy.config.n_action_steps
    dt = 1.0 / cfg.sim.control_hz
    files = sorted(Path(args.episodes).glob(args.glob))[: args.n]
    err = {"grasp": [], "place": [], "other": []}
    base = {"coast": {n: [] for n in err}, "repeat": {n: [] for n in err}}
    grip_err = []
    for f in files:
        meta, data = read_episode(f)
        env = env_from_episode(meta)
        data = executed_actions(env, data)
        prog = data["progress"]
        stop = len(prog) if prog.max() < args.level else min(len(prog), int(np.argmax(prog >= args.level)) + 50)
        stop -= k
        ev = phases(data, stop + k)
        frames = {t: "other" for t in range(0, stop, args.every)}
        for name, ts in ev.items():
            for e in ts:
                for t in range(max(0, e - WINDOW), min(stop, e), 6):
                    frames[t] = name
        A = data["action/joint_pos"]
        S = data["sim/state"]
        q, v = data["obs/joint_pos"], data["obs/joint_vel"]
        ahead = (LAG_S + dt * np.arange(k))[:, None]
        for t, name in sorted(frames.items()):
            env.set_state(S[t])
            runner.reset(env)
            pred = np.stack([runner.raw_action(env) for _ in range(k)])  # (k, 16)
            true = A[t:t + k]
            tp = env.tcp_poses_at(np.c_[pred[:, :7], pred[:, 8:15]])[:, :, :3]
            tt = env.tcp_poses_at(np.c_[true[:, :7], true[:, 8:15]])[:, :, :3]
            # the arm the demo moves most in this chunk (the other may just hold still)
            arm = int(np.argmax(np.linalg.norm(tt[-1] - tt[0], axis=-1) +
                                np.abs(np.diff(true[:, [7, 15]], axis=0)).sum(0)))
            e = np.linalg.norm(tp[:, arm] - tt[:, arm], axis=-1) * 1000  # mm, per step
            err[name].append((float(e[0]), float(e.mean()), float(e[-1])))
            prev = np.r_[A[max(t - 1, 0), :7], A[max(t - 1, 0), 8:15]]
            for kind, guess in (("coast", q[t] + ahead * v[t]), ("repeat", np.repeat(prev[None], k, axis=0))):
                e = np.linalg.norm(env.tcp_poses_at(guess)[:, arm, :3] - tt[:, arm], axis=-1) * 1000
                base[kind][name].append((float(e[0]), float(e.mean()), float(e[-1])))
            grip_err.append(float(np.abs(pred[:, [7, 15]] - true[:, [7, 15]]).mean()))
        print(f"{f.name}: {len(frames)} frames", flush=True)
    print(f"\n{runner.path}\n{len(files)} episodes from {args.episodes}; TCP error of the moving arm, mm "
          f"(first step / mean over the {k}-step chunk / last step)")
    for name, v in err.items():
        if not v:
            continue
        v = np.array(v)
        print(f"  {name:6s} n={len(v):4d}  median {np.median(v[:, 0]):5.1f} / {np.median(v[:, 1]):5.1f} / "
              f"{np.median(v[:, 2]):5.1f}   p90 {np.percentile(v[:, 0], 90):5.1f} / "
              f"{np.percentile(v[:, 1], 90):5.1f} / {np.percentile(v[:, 2], 90):5.1f}")
        for kind in base:
            b = np.median(np.array(base[kind][name]), axis=0)
            print(f"         {kind:6s}       median {b[0]:5.1f} / {b[1]:5.1f} / {b[2]:5.1f}")
    print(f"  gripper |pred - demo| mean {np.mean(grip_err):.3f} (0 closed .. 1 open)")
    runner.close()


if __name__ == "__main__":
    main()
