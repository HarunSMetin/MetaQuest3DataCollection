"""Does the closed loop stay on a demo it was trained on? Drift from the demo, in millimetres at the TCP.

    MUJOCO_GL=egl uv run --extra learn python scripts/drift_test.py <run or checkpoint> <episodes dir> \
        [--config experiments/two_bricks/config.yaml] [--n 6] [--starts 5 3.5] [--steps 128] \
        [--n-action-steps 1 8 16 64] [--force vel pos wrench images]

For some episodes (use the policy's own training takes) and some start times before the level clicks: put the
simulation in the recorded state, let the policy drive for `--steps` control steps through the safety filter (as in
`qc eval`), and measure how far the TCP of the arm that moves is from where the demo had it at the same step.

Rows:
  replay                 the demo's executed targets, replayed: must stay at 0.0 (checks this script and the scene)
  policy, k per chunk    the policy re-planning every k steps (one row per --n-action-steps)
  ... X forced           the same at the trained chunk length, but channel X of each observation is replaced by the
                         demo's value at that step (teacher forcing). `all` forced = the open-loop fit.

How to read it (fit_error.py measures the open-loop error; this measures what the loop does with it):
- Drift grows when the policy re-plans more often: every fresh prediction feeds an error back. The channel whose
  forcing removes the drift is the one that carries it (2026-11-12: joint velocity; experiments/REVIEW_2026-11-12.md).
- Drift is as small as the open-loop fit: the loop is stable around the demos, and failures come from states the
  demos don't cover (classic covariate shift: DAgger, perturbed demos).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from questcollect.config import load_config  # noqa: E402
from questcollect.learn.features import VEL_NAMES, WRENCH_NAMES, episode_obs  # noqa: E402
from questcollect.learn.policy import PolicyRunner  # noqa: E402
from questcollect.recorder import read_episode  # noqa: E402
from questcollect.safety import SafetyFilter  # noqa: E402
from questcollect.sim import CameraRenderer, env_from_episode, executed_actions  # noqa: E402


def channels(spec) -> dict[str, slice]:
    """Where each proprioceptive channel sits in observation.state."""
    n = len(spec.names)
    out = {"pos": slice(0, n)}
    if spec.joint_vel:
        out["vel"] = slice(n, n + len(VEL_NAMES))
        n += len(VEL_NAMES)
    if spec.ee_wrench:
        out["wrench"] = slice(n, n + len(WRENCH_NAMES))
    return out


class Forcing:
    """Wraps PolicyRunner.observation: replaces the chosen channels by the demo's at step `self.t`."""

    def __init__(self, runner: PolicyRunner):
        self.runner, self.slices = runner, channels(runner.spec)
        self._observation = runner.observation
        runner.observation = self.observation
        self.forced: tuple[str, ...] = ()
        self.t = 0
        self.demo_state = self.states = self.env = self.renderer = None

    def episode(self, meta: dict, data: dict) -> None:
        spec = self.runner.spec
        if self.renderer is not None:
            self.renderer.close()
        self.env = env_from_episode(meta)  # a second scene, to render the demo's frames
        self.renderer = CameraRenderer(self.env.model, spec.height, spec.width, list(spec.cameras))
        self.states = data["sim/state"]
        obs = episode_obs(data)
        if spec.ee_wrench and "ee_wrench" not in obs:
            raise SystemExit("episodes without obs/ee_wrench (schema < 7) are not supported here")
        self.demo_state = spec.state(obs).astype(np.float32)

    def observation(self, env) -> dict:
        batch = self._observation(env)
        for ch in self.forced:
            if ch == "images":
                self.env.set_state(self.states[self.t])
                for cam, img in self.renderer.render(self.env.data).items():
                    batch[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
            else:
                sl = self.slices[ch]
                batch["observation.state"][0, sl] = torch.from_numpy(self.demo_state[self.t, sl])
        return batch


def drive(env, safety, state, n: int, act) -> np.ndarray:
    """From `state`, n control steps with act() -> (14 joint targets, 2 grippers); (n, 2, 3) measured TCPs."""
    env.set_state(state)
    safety.reset(env.data.qpos[env.arm_qadr])
    tcp = np.empty((n, 2, 3))
    for t in range(n):
        q, grippers = act(t)
        q, _ = safety.filter(q, env.data.qpos[env.arm_qadr], env.data.qpos[env.finger_qadr])
        env.apply_action(q, grippers)
        obs = env.observe()
        tcp[t] = obs["ee_pose_left"][:3], obs["ee_pose_right"][:3]
        env.step()
    return tcp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("episodes")
    ap.add_argument("--config", default=str(ROOT / "experiments/two_bricks/config.yaml"))
    ap.add_argument("--n", type=int, default=6, help="episodes (spread over the folder)")
    ap.add_argument("--glob", default="episode_*.hdf5")
    ap.add_argument("--level", type=int, default=1, help="start times count back from this level's click")
    ap.add_argument("--starts", type=float, nargs="+", default=[5.0, 3.5], help="seconds before the click")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--n-action-steps", type=int, nargs="+", default=None, help="default: 1, half, trained, max")
    ap.add_argument("--force", nargs="*", default=None, help="channels to teacher-force, one row each "
                                                             "(default: every channel the policy has, and all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    runner = PolicyRunner(Path(args.run), cfg)
    pcfg = runner.policy.config
    trained = pcfg.n_action_steps
    longest = pcfg.horizon - pcfg.n_obs_steps + 1 if pcfg.type == "diffusion" else getattr(pcfg, "chunk_size", trained)
    chunk_lengths = args.n_action_steps or sorted({1, max(1, trained // 2), trained, longest})
    forcing = Forcing(runner)
    has = [*forcing.slices, *(["images"] if runner.spec.cameras else [])]
    force_rows = [(c,) for c in (has if args.force is None else args.force)] + ([tuple(has)] if args.force is None else [])
    safety = SafetyFilter(cfg.sim, cfg.ik, cfg.safety, 1.0 / cfg.sim.control_hz)
    files = sorted(Path(args.episodes).glob(args.glob))
    files = files[:: max(1, len(files) // args.n)][: args.n]
    marks = [k for k in (15, 31, 63, 95, 127, 191, 255) if k < args.steps]
    rows: dict[str, dict[float, list]] = {}

    def policy_act(t0):
        def act(t):
            forcing.t = t0 + t
            return runner.act(env)
        return act

    for f in files:
        meta, data = read_episode(f)
        prog = data["progress"]
        if prog.max() < args.level:
            continue
        env = env_from_episode(meta)
        forcing.episode(meta, data)
        A = executed_actions(env, data)["action/joint_pos"]
        demo = np.stack([data["obs/ee_pose_left"][:, :3], data["obs/ee_pose_right"][:, :3]], axis=1)
        click = int(np.argmax(prog >= args.level))
        for s in args.starts:
            t0 = click - round(s * cfg.sim.control_hz)
            if t0 < 0 or t0 + args.steps > len(A):
                continue
            ref = demo[t0:t0 + args.steps]
            arm = int(np.argmax(np.linalg.norm(ref[-1] - ref[0], axis=-1)))  # the arm the demo moves in this window

            def add(name, tcp):
                e = np.linalg.norm(tcp[:, arm] - ref[:, arm], axis=-1) * 1000
                rows.setdefault(name, {}).setdefault(s, []).append(e[marks])

            add("replay (must be 0)", drive(env, safety, data["sim/state"][t0], args.steps,
                                             lambda t: (np.r_[A[t0 + t, :7], A[t0 + t, 8:15]], A[t0 + t, [7, 15]])))
            for k in chunk_lengths:
                pcfg.n_action_steps, forcing.forced = k, ()
                runner.reset(env)
                add(f"policy, {k} per chunk" + (" (trained)" if k == trained else ""),
                    drive(env, safety, data["sim/state"][t0], args.steps, policy_act(t0)))
            pcfg.n_action_steps = trained
            for forced in force_rows:
                forcing.forced = forced
                runner.reset(env)
                add(f"  {'all' if len(forced) > 1 else forced[0]} forced",
                    drive(env, safety, data["sim/state"][t0], args.steps, policy_act(t0)))
        print(f"{f.name} done", flush=True)

    print(f"\n{runner.path}\nTCP distance from the demo, mm, median over {len(files)} episodes, after k steps; "
          f"k = {[m + 1 for m in marks]}")
    for s in args.starts:
        print(f"-- started {s:g} s before level {args.level} clicks")
        for name, per_start in rows.items():
            if s in per_start:
                print(f"  {name:28s}" + " ".join(f"{v:7.1f}" for v in np.median(per_start[s], axis=0)))
    runner.close()


if __name__ == "__main__":
    main()
