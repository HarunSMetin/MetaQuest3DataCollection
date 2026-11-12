"""E3: how fast does the closed loop leave the demo? From training-demo states, run N steps and compare the measured
right-TCP with the demo's at the same time index. Variants: steps per chunk, safety filter on/off, demo replay."""
import sys; sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from harness import *
cfg = load_config(EXPERT_CFG)
runner = PolicyRunner(EXPERT_RUN, cfg)
saf_on = SafetyFilter(cfg.sim, cfg.ik, cfg.safety, 0.02)
saf_off = SafetyFilter(cfg.sim, cfg.ik, cfg.safety.model_copy(update={"enabled": False}), 0.02)
N = 160
KS = [0, 15, 31, 47, 63, 95, 127, 159]
variants = {"replay demo actions": (None, saf_on), "replay, no safety": (None, saf_off),
            "policy nas=16": (16, saf_on), "policy nas=16 no safety": (16, saf_off), "policy nas=64": (64, saf_on),
            "policy nas=32": (32, saf_on), "policy nas=8": (8, saf_on), "policy nas=1": (1, saf_on)}
out = {v: {o: [] for o in (5.0, 3.5, 2.5)} for v in variants}
for i in range(6):
    meta, d = read_episode(ROOT / f"data/expert_v2/episode_{i * 7:06d}.hdf5")
    env = env_from_episode(meta)
    A = executed_actions(env, d)["action/joint_pos"]
    tc = int(np.argmax(d["progress"] >= 1))
    demo_tcp = d["obs/ee_pose_right"][:, :3]
    for o in (5.0, 3.5, 2.5):
        t0 = tc - int(o * 50)
        for name, (nas, saf) in variants.items():
            if nas is not None:
                runner.policy.config.n_action_steps = nas
            L = closed_loop(env, runner, saf, d["sim/state"][t0], N, use_policy=nas is not None, actions=A[t0:t0 + N],
                            stop_on_click=False)
            e = np.linalg.norm(L["tcp"][:, 1] - demo_tcp[t0:t0 + N], axis=1) * 1000
            out[name][o].append(e[KS])
    print("episode", i * 7, "done", flush=True)
print("right-TCP distance from the demo [mm] after k steps (median over 6 training episodes)")
print("k = ", KS)
for o in (5.0, 3.5, 2.5):
    print(f"-- start {o} s before the demo's click")
    for name in variants:
        print(f"  {name:26s}", np.round(np.median(out[name][o], axis=0), 1))
