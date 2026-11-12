"""E7: the right arm's final transit to the tower, expert demos vs the expert-data policy's eval rollouts:
peak speed, and how far from the blue goal (xy) the hand is when it starts to descend."""
import sys, glob, numpy as np
sys.path.insert(0, ".")
from questcollect.recorder import read_episode
from questcollect.sim import tower_poses, _quat_yaw
KEYS = ["obs/ee_pose_right", "obs/objects", "progress", "action/joint_pos"]
def transit(f):
    meta, d = read_episode(f, keys=KEYS)
    tcp, ob = d["obs/ee_pose_right"][:, :3], d["obs/objects"]
    held = ob[:, 1, 2] > ob[0, 1, 2] + 0.03          # blue carried
    if not held.any(): return None
    # the last stretch in which the carried blue moves towards the red brick by > 8 cm in xy
    red_xy = ob[:, 0, :2]; dist = np.linalg.norm(ob[:, 1, :2] - red_xy, axis=1)
    idx = np.flatnonzero(held)
    far = idx[dist[idx] > 0.12]
    if len(far) == 0: return None
    t0 = far[-1]                                      # last time blue was > 12 cm from red while carried
    seg = slice(t0, min(len(tcp), t0 + 250))
    v = np.linalg.norm(np.diff(tcp[seg, :2], axis=0), axis=1) * 50
    z = tcp[seg, 2]; ztrans = z[0]
    down = np.flatnonzero(z < ztrans - 0.015)
    if len(down) == 0: return None
    td = t0 + down[0]
    best = min((np.linalg.norm(ob[td, 1, :2] - tower_poses(ob[td, 0, :3], _quat_yaw(ob[td, 0, 3:]) + fl)[1][0][:2]) for fl in (0, np.pi)))
    return v.max(), best * 1000, (td - t0) / 50, int(d["progress"].max() >= 1)
for name, files in (("expert demos", sorted(glob.glob("data/expert_v2/episode_*.hdf5"))[:50]),
                    ("policy rollouts (16/chunk)", sorted(glob.glob("outputs/20261111-1316_dp_240_pre_expert_v2/eval/20261112-185224/episode_*.hdf5"))),
                    ("policy rollouts (8/chunk)", sorted(glob.glob("outputs/20261111-1316_dp_240_pre_expert_v2/eval/20261112-190154/episode_*.hdf5")))):
    r = np.array([x for x in map(transit, files) if x is not None])
    print(f"{name}: {len(r)}/{len(files)} with a final transit; peak hand speed median {np.median(r[:,0]):.3f} m/s (p10 {np.percentile(r[:,0],10):.3f}, p90 {np.percentile(r[:,0],90):.3f}); "
          f"blue-to-goal xy when the descent starts: median {np.median(r[:,1]):.1f} mm (p90 {np.percentile(r[:,1],90):.1f}); transit time {np.median(r[:,2]):.2f} s")
    if "policy" in name:
        print("    corr(peak speed, xy shortfall at descent) = %.2f" % np.corrcoef(r[:, 0], r[:, 1])[0, 1])
