"""E2: restart the expert-data policy from states of its own TRAINING demos, at several times before the click.
Does it click? Where does blue end relative to its goal?"""
import sys; sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from harness import *
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--nas", type=int, default=None); ap.add_argument("--no-safety", action="store_true")
ap.add_argument("--eps", type=int, default=6); ap.add_argument("--offsets", default="-1,8,6,4,3,2,1"); args = ap.parse_args()
cfg = load_config(EXPERT_CFG)
runner = PolicyRunner(EXPERT_RUN, cfg, n_action_steps=args.nas)
safety = SafetyFilter(cfg.sim, cfg.ik, cfg.safety.model_copy(update={"enabled": not args.no_safety}), 0.02)
offs = [float(x) for x in args.offsets.split(",")]
res = {o: [] for o in offs}
for i in range(args.eps):
    meta, d = read_episode(ROOT / f"data/expert_v2/episode_{i * 7:06d}.hdf5")
    env = env_from_episode(meta)
    tc = int(np.argmax(d["progress"] >= 1))
    for o in offs:
        t0 = 0 if o < 0 else max(0, tc - int(o * 50))
        n = (tc - t0) + 400  # 8 s of slack past the demo's click time
        L = closed_loop(env, runner, safety, d["sim/state"][t0], n)
        e = blue_err(env)
        clicked = bool(L["prog"][-1] >= 1) or env.progress() >= 1
        res[o].append((clicked, np.linalg.norm(e[:2]), e[2], len(L["prog"]) - (tc - t0), np.abs(L["q_safe"] - L["q_pol"]).max()))
        print(f"ep {i*7:3d} start {'t=0' if o < 0 else f'click-{o:g}s'}: clicked {clicked}  blue xy err {np.linalg.norm(e[:2]):6.1f} mm dz {e[2]:6.1f}  steps vs demo {len(L['prog']) - (tc - t0):+5d}  safety max {np.abs(L['q_safe'] - L['q_pol']).max():.3f}", flush=True)
print(f"\nnas={args.nas} safety={not args.no_safety}")
for o, v in res.items():
    v = np.array(v, float)
    print(f"  start {'t=0     ' if o < 0 else f'click-{o:g}s'}: clicked {int(v[:,0].sum())}/{len(v)}  median xy err {np.median(v[:,1]):.1f} mm")
