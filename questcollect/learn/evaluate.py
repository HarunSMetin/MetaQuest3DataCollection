"""Policy evaluation in simulation: rollouts on fixed held-out seeds, with the safety filter, as fast as the PC
allows (no wall clock).

Per rollout: success, how far the tower got (base in the area, levels clicked), time to each stage, contacts
and how much the safety filter had to change. Results go to <run>/eval/<timestamp>/: metrics.json plus every
rollout as a normal episode file (source = policy), so `qc replay`, `qc audit` and the dashboard work on them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from questcollect.audit import CONTACT_KINDS, true_runs
from questcollect.config import Config
from questcollect.learn.policy import PolicyRunner
from questcollect.learn.train import run_manifest
from questcollect.recorder import (AUTO_STOP_TAIL_S, SOURCE_POLICY, episode_meta, make_step, teleop_fields,
                                   write_episode)
from questcollect.safety import SafetyFilter
from questcollect.sim import LEVELS, BimanualEnv

log = logging.getLogger(__name__)
EVAL_SEED_BASE = 20_260_000  # eval seeds come from their own stream, apart from the random recording seeds
DEFAULT_MAX_S = 150.0


def eval_seeds(n: int) -> list[int]:
    return [int(s % 2**31) for s in np.random.SeedSequence(EVAL_SEED_BASE).generate_state(n, dtype=np.uint64)]


@dataclass
class RolloutResult:
    seed: int
    success: bool
    duration_s: float
    base_in_area: bool
    max_progress: int                      # tower levels clicked in on the base (0..3)
    level_times: list[float | None]
    contact_events: dict[str, int]
    contact_frac: dict[str, float]
    safety_mean_rad: float                 # mean change the safety filter made to the policy's joint targets
    safety_max_rad: float
    file: str | None = None


def max_steps_for(run: Path, fps: int) -> int:
    """1.5 x the 95th percentile length of the training segments (the policy has had enough time by then)."""
    try:
        segs = run_manifest(run)["dataset_manifest"]["segments"]
        return int(1.5 * np.percentile([s["stop"] - s["start"] for s in segs], 95))
    except (FileNotFoundError, KeyError):
        return int(DEFAULT_MAX_S * fps)


def rollout(env: BimanualEnv, runner: PolicyRunner, safety: SafetyFilter, seed: int, max_steps: int,
            record: bool = True) -> tuple[RolloutResult, list[dict]]:
    env.reset(seed)
    runner.reset(env)
    safety.reset(env.data.qpos[env.arm_qadr])
    fps = round(1 / env.control_dt)
    steps, contacts, changes, progress = [], [], [], []
    in_area = False
    success_since = None
    for t in range(max_steps):
        obs, state, flags = env.observe(), env.get_state(), env.contact_flags()
        q_policy, grippers = runner.act(env)
        q, change = safety.filter(q_policy, env.data.qpos[env.arm_qadr])
        env.apply_action(q, grippers)
        success, prog = env.success(), env.progress()
        if record:
            steps.append(make_step(env, obs, state, q, grippers, safety.tcp_poses(), success, prog, flags,
                                   teleop_fields(None), SOURCE_POLICY))
        env.step()
        contacts.append(flags)
        changes.append(change)
        progress.append(prog)
        in_area |= env.in_area()
        if success:
            success_since = t if success_since is None else success_since
            if (t - success_since) / fps >= AUTO_STOP_TAIL_S:
                break
        else:
            success_since = None
    contacts, progress = np.array(contacts), np.array(progress)
    result = RolloutResult(
        seed=seed, success=success_since is not None, duration_s=len(progress) / fps, base_in_area=in_area,
        max_progress=int(progress.max()),
        level_times=[float(np.argmax(progress >= k)) / fps if (progress >= k).any() else None
                     for k in range(1, LEVELS + 1)],
        contact_events={k: len(true_runs(contacts[:, i])) for i, k in enumerate(CONTACT_KINDS)},
        contact_frac={k: float(contacts[:, i].mean()) for i, k in enumerate(CONTACT_KINDS)},
        safety_mean_rad=float(np.mean(changes)), safety_max_rad=float(np.max(changes)))
    return result, steps


def summarize(results: list[RolloutResult]) -> dict:
    n = len(results)
    frac = lambda f: sum(map(f, results)) / n  # noqa: E731
    return {
        "rollouts": n,
        "success_rate": frac(lambda r: r.success),
        "stage_rates": {"base_in_area": frac(lambda r: r.base_in_area),
                        **{f"level_{k}": frac(lambda r, k=k: r.max_progress >= k) for k in range(1, LEVELS + 1)}},
        "mean_success_time_s": (float(np.mean([r.duration_s for r in results if r.success]))
                                if any(r.success for r in results) else None),
        "rollouts_with_contact": {k: frac(lambda r, k=k: r.contact_events[k] > 0) for k in CONTACT_KINDS},
        "safety_mean_rad": float(np.mean([r.safety_mean_rad for r in results])),
        "safety_max_rad": float(np.max([r.safety_max_rad for r in results])),
    }


def evaluate(run: Path, cfg: Config, n: int = 50, max_steps: int | None = None, use_safety: bool = True,
             save_rollouts: bool = True, temporal_ensemble: float | None = None, n_action_steps: int | None = None,
             out: Path | None = None, progress=None) -> tuple[dict, Path]:
    """Evaluate a run's last checkpoint (or a checkpoint folder); returns (summary, results folder)."""
    env = BimanualEnv(cfg)
    runner = PolicyRunner(Path(run), cfg, n_action_steps=n_action_steps, temporal_ensemble=temporal_ensemble)
    run = runner.run  # the run folder, also when a checkpoint folder was given
    trained_in = {s["scene_hash"] for s in run_manifest(run)["dataset_manifest"]["segments"]}
    if trained_in != {env.scene_hash}:
        log.warning("evaluating in scene %s, but the training data was recorded in %s: results may not be "
                    "comparable", env.scene_hash, sorted(trained_in))
    safety = SafetyFilter(cfg.sim, cfg.ik, cfg.safety.model_copy(update={"enabled": use_safety}), env.control_dt)
    fps = round(1 / env.control_dt)
    max_steps = max_steps or max_steps_for(run, fps)
    out = out or (run / "eval" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, seed in enumerate(eval_seeds(n)):
        result, steps = rollout(env, runner, safety, seed, max_steps, record=save_rollouts)
        if save_rollouts:
            path = out / f"episode_{i:06d}.hdf5"
            meta = episode_meta(cfg, env, len(steps), result.success, input_mode="policy",
                                operator=f"policy:{run.name}", source="policy")
            write_episode(path, steps, meta)
            result.file = path.name
        results.append(result)
        if progress:
            progress(i + 1, n, result)
    runner.close()
    summary = summarize(results)
    (out / "metrics.json").write_text(json.dumps({
        "run": str(run), "checkpoint": str(runner.path), "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {"n": n, "max_steps": max_steps, "safety": use_safety, "temporal_ensemble": temporal_ensemble,
                     "n_action_steps": n_action_steps, "scene_hash": env.scene_hash},
        "summary": summary, "rollouts": [asdict(r) for r in results]}, indent=1))
    return summary, out
