"""Check, before a long training run, that the training data and the evaluation see the same world.

    uv run python scripts/check_pipeline.py --config experiments/two_bricks/config.yaml
    MUJOCO_GL=egl uv run --extra learn python scripts/check_pipeline.py --config <cfg> --dataset <name>  # + dataset

Every check uses the environment `qc eval` builds (BimanualEnv(cfg) from the given config), not the episode's own
saved scene, so "replays exactly" means "the evaluation physics is the physics the data was recorded in".

1. config   sim / ik / safety / teleop are the same as config.yaml and as the episodes' saved configs
2. scene    every selected episode was recorded in the evaluation scene (scene_hash)
3. dart     record.action_noise in the config; whether the episodes' actions are the executed targets
4. replay   the recorded actions, applied in the eval env from sim/state[0], reproduce sim/state step by step
            (also contacts, progress, the clutch welds and obs/ee_wrench), and reach the curated stop level
5. starts   the eval start layouts (eval seeds) lie in the range of the training episodes' start layouts
6. horizon  the eval step limit covers the training segments (fps match)
7. dataset  (with --dataset) frame counts, observation.state and action equal to the source episodes,
            images equal to what the policy renders live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from questcollect.audit import audit_dir, select  # noqa: E402
from questcollect.config import load_config  # noqa: E402
from questcollect.recorder import read_episode  # noqa: E402
from questcollect.sim import BRICKS, BimanualEnv, CameraRenderer, executed_actions  # noqa: E402

results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def check_config(cfg, base, metas):
    for part in ("sim", "ik", "safety", "teleop"):
        a, b = getattr(cfg, part).model_dump(), getattr(base, part).model_dump()
        diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
        report(f"config.{part} == config.yaml", not diff, str(diff) if diff else "identical")
    bad = {}
    for name, meta in metas.items():
        saved = yaml.safe_load(meta["config_yaml"])
        for part in ("sim", "ik", "safety"):
            mine = getattr(cfg, part).model_dump()
            for k, v in (saved.get(part) or {}).items():
                if k in mine and mine[k] != v and not (isinstance(v, list) and list(mine[k]) == v):
                    bad.setdefault(f"{part}.{k}", set()).add(f"{v} (eval {mine[k]})")
    report("episodes' saved sim/ik/safety == eval config", not bad,
           "; ".join(f"{k}: {sorted(v)}" for k, v in bad.items()) or f"{len(metas)} episodes identical")


def check_replay(env, path, seg, data, until_level):
    """Recorded actions from sim/state[0] in the eval env -> the recorded states, step by step."""
    S, A = data["sim/state"], data["action/joint_pos"]
    env.set_state(S[0])
    worst_state, worst_wrench, contacts_off, first_bad = 0.0, 0.0, 0, None
    for t in range(len(A) - 1):
        obs = env.observe()
        worst_wrench = max(worst_wrench, float(np.abs(obs["ee_wrench"] - data["obs/ee_wrench"][t]).max()))
        contacts_off += int((env.contact_flags() != data["contacts"][t]).any())
        env.apply_action(np.r_[A[t, :7], A[t, 8:15]], A[t, [7, 15]])
        env.step()
        if t + 1 >= seg[1]:
            break
        d = float(np.abs(env.get_state() - S[t + 1]).max())
        if d > 1e-9 and first_bad is None:
            first_bad = t + 1
        worst_state = max(worst_state, d)
    level = env.progress()
    ok = first_bad is None and worst_wrench < 1e-6 and contacts_off == 0 and \
        (until_level is None or level >= until_level)
    return ok, (f"{path.name}: state err {worst_state:.1e}" + (f" from step {first_bad}" if first_bad else "") +
                f", wrench err {worst_wrench:.1e} N, contact mismatches {contacts_off}, level {level}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--dataset", default=None, help="exported dataset name (learn.datasets_dir/<name>)")
    ap.add_argument("--replay", type=int, default=0, help="replay only this many episodes (0 = all)")
    args = ap.parse_args()
    cfg, base = load_config(args.config), load_config(ROOT / "config.yaml")
    episodes_dir = ROOT / cfg.record.out_dir
    segments, rejected = select(episodes_dir, audit_dir(episodes_dir), cfg.curate)
    files = sorted({s.path for s in segments})
    print(f"{len(segments)} segments from {len(files)} episodes in {episodes_dir} ({len(rejected)} left out)")
    env = BimanualEnv(cfg)  # what qc eval builds
    metas, datas = {}, {}
    for f in files:
        metas[f.name], datas[f.name] = read_episode(f)

    # 1-3
    check_config(cfg, base, metas)
    hashes = {m["scene_hash"] for m in metas.values()}
    report("scene_hash == eval scene", hashes == {env.scene_hash}, f"episodes {sorted(hashes)}, eval {env.scene_hash}")
    report("record.action_noise (DART) for new recordings", cfg.record.action_noise == 0.0,
           f"{cfg.record.action_noise}")
    drift = []
    for name, data in datas.items():
        A = data["action/joint_pos"]
        drift.append(float(np.abs(executed_actions(env, data)["action/joint_pos"][:-1] - A[:-1]).max()))
    src_dart = sorted({yaml.safe_load(m["config_yaml"])["record"].get("action_noise") for m in metas.values()})
    report("episode actions == executed targets", max(drift) < 1e-9 or cfg.learn.action_label == "executed",
           f"max |recorded - executed| {max(drift):.2e} rad; recorded with action_noise {src_dart}; "
           f"export uses learn.action_label: {cfg.learn.action_label}")

    # 4
    bad, n = [], 0
    for seg in segments[: args.replay or None]:
        data = datas[seg.path.name]
        if cfg.learn.action_label == "executed":
            data = executed_actions(env, data)
        ok, msg = check_replay(env, seg.path, (seg.start, seg.stop), data, cfg.curate.until_level)
        n += 1
        if not ok:
            bad.append(msg)
    report("recorded actions replay exactly in the eval env", not bad,
           f"{n - len(bad)}/{n} exact" + ("; " + " | ".join(bad[:5]) if bad else ""))

    # 5
    from questcollect.learn.evaluate import eval_seeds
    train0 = np.stack([d["obs/objects"][0] for d in datas.values()])  # (E, 4, 7)
    ev = []
    for seed in eval_seeds(50):
        env.reset(seed)
        ev.append(env.observe()["objects"])
    ev = np.stack(ev)
    out = []
    for b, brick in enumerate(BRICKS):
        lo, hi = train0[:, b, :2].min(0), train0[:, b, :2].max(0)
        outside = int((((ev[:, b, :2] < lo - 0.005) | (ev[:, b, :2] > hi + 0.005)).any(axis=1)).sum())
        out.append(f"{brick.name} {outside}/50 outside train xy [{lo.round(3)}..{hi.round(3)}]")
    report("eval start layouts inside the training range (5 mm slack)",
           all(o.split()[1].startswith("0/") for o in out), "; ".join(out))
    q0 = np.stack([d["obs/joint_pos"][0] for d in datas.values()])
    env.reset(eval_seeds(1)[0])
    dq = np.abs(q0 - env.observe()["joint_pos"]).max(axis=1)  # per episode; some start after a small nudge
    report("eval start arm pose == training start pose", np.median(dq) < 1e-4 and dq.max() < 2e-3,
           f"|q0 diff| median {np.median(dq):.1e}, max {dq.max():.1e} rad (2e-3 rad ~ 1-2 mm at the gripper)")

    # 6
    fps = {int(m["fps"]) for m in metas.values()}
    lens = np.array([s.num_steps for s in segments])
    max_steps = int(1.5 * np.percentile(lens, 95))
    report("fps == control_hz and eval horizon covers the segments",
           fps == {cfg.sim.control_hz} and max_steps >= lens.max(),
           f"fps {fps}, control_hz {cfg.sim.control_hz}; segments {lens.min()}-{lens.max()} steps, "
           f"eval limit {max_steps} steps ({max_steps / cfg.sim.control_hz:.0f} s)")

    # 7
    if args.dataset:
        check_dataset(cfg, env, args.dataset, segments, datas)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


def check_dataset(cfg, env, name, segments, datas):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from questcollect.learn.features import FeatureSpec, episode_obs

    root = ROOT / cfg.learn.datasets_dir / name
    man = json.loads((root / "questcollect_manifest.json").read_text())
    spec = FeatureSpec.from_dict(man["features"])
    report("dataset features == config", spec == FeatureSpec.from_config(cfg), str(spec))
    segs = [(s["file"], s["start"], s["stop"]) for s in man["segments"]]
    report("dataset segments == current selection", segs == [(s.path.name, s.start, s.stop) for s in segments],
           f"{len(segs)} in dataset, {len(segments)} selected now")
    ds = LeRobotDataset(man["repo_id"], root=root, tolerance_s=0.001)
    report("dataset frame count", ds.num_frames == sum(b - a for _, a, b in segs),
           f"{ds.num_frames} frames, segments sum {sum(b - a for _, a, b in segs)}")
    renderer = CameraRenderer(env.model, spec.height, spec.width, list(spec.cameras))
    worst_s, worst_a, worst_img, k = 0.0, 0.0, 0.0, 0
    for file, a, b in segs:
        data = datas[file]
        if spec.action_label == "executed":
            data = executed_actions(env, data)
        state, action = spec.state(episode_obs(data)), spec.recorded_action(data)
        idx = np.linspace(a, b - 1, 5).astype(int)
        for t in idx:
            fr = ds[k + int(t - a)]
            worst_s = max(worst_s, float((fr["observation.state"] - torch.from_numpy(state[t]).float()).abs().max()))
            worst_a = max(worst_a, float((fr["action"] - torch.from_numpy(action[t]).float()).abs().max()))
            # what the policy sees live: the eval env right after its physics step (step 0: right after reset)
            if t == 0:
                env.set_state(data["sim/state"][0])
            else:
                env.set_state(data["sim/state"][t - 1])
                env.apply_action(np.r_[data["action/joint_pos"][t - 1, :7], data["action/joint_pos"][t - 1, 8:15]],
                                 data["action/joint_pos"][t - 1, [7, 15]])
                env.step()
            for cam, img in renderer.render(env.data).items():
                diff = np.abs(fr[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255 - img).mean()
                worst_img = max(worst_img, float(diff))
        k += b - a
    renderer.close()
    report("dataset state/action == source episodes", worst_s < 1e-4 and worst_a < 1e-4,
           f"max |state diff| {worst_s:.1e}, max |action diff| {worst_a:.1e}")
    report("dataset images == live policy render", worst_img < 3.0,
           f"worst mean abs pixel diff {worst_img:.2f}/255 (video compression ~1-2)")


if __name__ == "__main__":
    main()
