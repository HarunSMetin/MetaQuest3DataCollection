"""E6: what do trivial predictors score on fit_error's metric (same frames, same 16-step window, moving arm, mm)?"""
import sys, glob, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from fit_error import phases, WINDOW
from questcollect.recorder import read_episode
from questcollect.sim import env_from_episode, executed_actions
K, TAU, DT = 16, 0.12, 0.02
def run(name, files):
    err = {}
    for f in files:
        meta, data = read_episode(f)
        env = env_from_episode(meta)
        data = executed_actions(env, data)
        prog = data["progress"]
        stop = (len(prog) if prog.max() < 1 else min(len(prog), int(np.argmax(prog >= 1)) + 50)) - K
        ev = phases(data, stop + K)
        frames = {t: "other" for t in range(40, stop, 40)}
        for nm, ts in ev.items():
            for e in ts:
                for t in range(max(2, e - WINDOW), min(stop, e), 6):
                    frames[t] = nm
        A = data["action/joint_pos"]; A14 = np.c_[A[:, :7], A[:, 8:15]]
        q, v = data["obs/joint_pos"], data["obs/joint_vel"]
        for t, nm in sorted(frames.items()):
            true = A14[t:t + K]
            k = np.arange(K)[:, None]
            preds = {"hold current joints": np.repeat(q[t][None], K, 0),
                     "repeat previous target": np.repeat(A14[t - 1][None], K, 0),
                     "previous target + its last step, extrapolated": A14[t - 1] + (k + 1) * (A14[t - 1] - A14[t - 2]),
                     "coast: q + tau*qdot + k*dt*qdot": q[t] + TAU * v[t] + k * DT * v[t]}
            tt = env.tcp_poses_at(true)[:, :, :3]
            arm = int(np.argmax(np.linalg.norm(tt[-1] - tt[0], axis=-1) + np.abs(np.diff(A[t:t + K][:, [7, 15]], axis=0)).sum(0)))
            for pn, p in preds.items():
                e = np.linalg.norm(env.tcp_poses_at(p)[:, arm, :3] - tt[:, arm], axis=-1) * 1000
                err.setdefault(pn, {}).setdefault(nm, []).append((e[0], e.mean(), e[-1]))
    print(f"{name}: {len(files)} episodes; TCP error of the moving arm, mm, median (first step / mean over 16 / last step)")
    for pn, d in err.items():
        print(f"  {pn:48s}" + "   ".join(f"{nm} {np.median(np.array(v)[:,0]):4.1f}/{np.median(np.array(v)[:,1]):4.1f}/{np.median(np.array(v)[:,2]):4.1f}" for nm, v in sorted(d.items())))
run("expert_v2", sorted(glob.glob("data/expert_v2/episode_*.hdf5"))[:8])
run("retimed_v4 (human)", sorted(glob.glob("data/episodes_retimed_v4/episode_*.hdf5"))[:15])
