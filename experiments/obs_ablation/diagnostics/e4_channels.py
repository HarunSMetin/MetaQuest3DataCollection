"""E4: which observation channel carries the closed-loop instability? Closed loop from training-demo states, but
with one channel of the observation replaced by the demo's value at the same time index (teacher forcing)."""
import sys; sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from harness import *
import torch
from questcollect.learn.features import episode_obs
from questcollect.sim import CameraRenderer
cfg = load_config(EXPERT_CFG)
runner = PolicyRunner(EXPERT_RUN, cfg)
safety = SafetyFilter(cfg.sim, cfg.ik, cfg.safety, 0.02)
spec = runner.spec
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
NAS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
runner.policy.config.n_action_steps = NAS
KS = [15, 31, 63, 95, 127][: 3 if N < 96 else 5]
SL = {"pos": slice(0, 16), "vel": slice(16, 30), "wrench": slice(30, 42)}
orig_observation = runner.observation
ctx = {}

def patched(env):
    b = orig_observation(env)
    k = ctx["k"]; forced = ctx["forced"]
    demo_state = torch.from_numpy(ctx["demo_state"][k].astype(np.float32))
    for ch in forced:
        if ch in SL:
            b["observation.state"][0, SL[ch]] = demo_state[SL[ch]]
        elif ch == "zero_vel":
            b["observation.state"][0, SL["vel"]] = 0.0
        elif ch == "images":
            ctx["env2"].set_state(ctx["S"][k])
            for cam, img in ctx["rend2"].render(ctx["env2"].data).items():
                b[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
    return b
runner.observation = patched

def run(env, state, n):
    env.set_state(state); runner.reset(env); safety.reset(env.data.qpos[env.arm_qadr])
    tcp = []
    for t in range(n):
        ctx["k"] = ctx["t0"] + t
        q_pol, grip = runner.act(env)
        q, _ = safety.filter(q_pol, env.data.qpos[env.arm_qadr], env.data.qpos[env.finger_qadr])
        env.apply_action(q, grip)
        tcp.append(env.observe()["ee_pose_right"][:3].copy())
        env.step()
    return np.array(tcp)

variants = {"closed loop (nothing forced)": [], "vel forced": ["vel"], "wrench forced": ["wrench"], "pos forced": ["pos"],
            "images forced": ["images"], "vel+wrench forced": ["vel", "wrench"], "pos+vel+wrench forced": ["pos", "vel", "wrench"],
            "all forced (= open-loop fit)": ["pos", "vel", "wrench", "images"], "vel zeroed": ["zero_vel"]}
offs = (5.0, 3.5)
out = {v: {o: [] for o in offs} for v in variants}
for i in range(6):
    meta, d = read_episode(ROOT / f"data/expert_v2/episode_{i * 7:06d}.hdf5")
    env = env_from_episode(meta); env2 = env_from_episode(meta)
    ctx.update(env2=env2, rend2=CameraRenderer(env2.model, spec.height, spec.width, list(spec.cameras)), S=d["sim/state"],
               demo_state=spec.state(episode_obs(d)))
    tc = int(np.argmax(d["progress"] >= 1)); demo_tcp = d["obs/ee_pose_right"][:, :3]
    for o in offs:
        t0 = tc - int(o * 50); ctx["t0"] = t0
        for name, forced in variants.items():
            ctx["forced"] = forced
            tcp = run(env, d["sim/state"][t0], N)
            out[name][o].append((np.linalg.norm(tcp - demo_tcp[t0:t0 + N], axis=1) * 1000)[KS])
    ctx["rend2"].close(); print("episode", i * 7, flush=True)
print(f"n_action_steps = {NAS}: right-TCP distance from the demo [mm] after k steps, median of 6 training episodes; k = {KS}")
for o in offs:
    print(f"-- start {o} s before the demo's click")
    for name in variants:
        print(f"  {name:30s}", np.round(np.median(out[name][o], axis=0), 1))
