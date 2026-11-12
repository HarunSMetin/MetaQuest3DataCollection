"""E5: is the demo action predictable from proprioception alone?  lead = target - measured joints, regressed on the
measured joint velocity (per joint, linear). A high R^2 means 'action = q + tau * qdot' is available as a shortcut."""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from questcollect.recorder import read_episode
from questcollect.sim import env_from_episode, executed_actions
for name, pattern, n in (("expert_v2", "data/expert_v2/episode_*.hdf5", 30), ("retimed_v4 (human)", "data/episodes_retimed_v4/*.hdf5", 30),
                         ("human raw (DART)", "data/episodes/episode_*.hdf5", 30)):
    files = sorted(glob.glob(pattern))[:n]
    Q, V, L, L1 = [], [], [], []
    env = None
    for f in files:
        meta, d = read_episode(f, keys=["obs/joint_pos", "obs/joint_vel", "action/joint_pos", "sim/state", "action/ee_pose", "progress"])
        env = env or env_from_episode(meta)
        a = executed_actions(env, d)["action/joint_pos"]; a = np.c_[a[:, :7], a[:, 8:15]]
        p = d["progress"]; stop = len(p) if p.max() < 1 else int(np.argmax(p >= 1)) + 50
        q, v = d["obs/joint_pos"][:stop], d["obs/joint_vel"][:stop]
        Q.append(q); V.append(v); L.append(a[:stop] - q)
    V, L = np.concatenate(V), np.concatenate(L)
    r2, tau = [], []
    for j in range(14):
        x, y = V[:, j], L[:, j]
        k = (x @ y) / (x @ x); tau.append(k); r2.append(1 - ((y - k * x) ** 2).sum() / (y ** 2).sum())
    # all-joint linear map
    W, *_ = np.linalg.lstsq(np.c_[V, np.ones(len(V))], L, rcond=None)
    res = L - np.c_[V, np.ones(len(V))] @ W
    moving = np.abs(V).max(1) > 0.02
    print(f"{name}: {len(files)} episodes, {len(V)} steps ({moving.mean():.0%} with any joint > 0.02 rad/s)")
    print("   per-joint R^2 of lead ~ tau*qdot:", np.round(r2, 2), " tau [s]:", np.round(tau, 3))
    print(f"   lead rms {np.sqrt((L**2).mean()):.4f} rad -> residual after the linear velocity model {np.sqrt((res**2).mean()):.4f} rad"
          f" (R^2 {1 - (res**2).sum() / ((L - L.mean(0))**2).sum():.3f})")
