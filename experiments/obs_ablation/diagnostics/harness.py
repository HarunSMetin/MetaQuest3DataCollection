"""Shared helpers for the review diagnostics (scratch)."""
import sys, numpy as np
from pathlib import Path
ROOT = Path("/home/laptop/Desktop/Bimanual /MetaQuest3DataCollection")
sys.path.insert(0, str(ROOT))
from questcollect.config import load_config
from questcollect.learn.policy import PolicyRunner
from questcollect.recorder import read_episode
from questcollect.safety import SafetyFilter
from questcollect.sim import env_from_episode, executed_actions, BimanualEnv, tower_poses, _quat_yaw, BRICKS

EXPERT_RUN = ROOT / "outputs/20261111-1316_dp_240_pre_expert_v2"
V4_RUN = ROOT / "outputs/20261106-2147_dp_240_pre_two_bricks_v4"
EXPERT_CFG = ROOT / "experiments/expert_dagger/config.yaml"


def blue_err(env):
    """blue brick position minus its goal on red as red lies now: (xy error mm (2,), dz mm), best of the 2 symmetric goals."""
    red, blue = env.data.body(BRICKS[0].name), env.data.body(BRICKS[1].name)
    best = None
    for flip in (0.0, np.pi):
        gp, _ = tower_poses(red.xpos, _quat_yaw(red.xquat) + flip)[1]
        e = (blue.xpos - gp) * 1000
        if best is None or np.linalg.norm(e[:2]) < np.linalg.norm(best[:2]):
            best = e
    return best


def closed_loop(env, runner, safety, state, n_steps, use_policy=True, actions=None, stop_on_click=True):
    """From `state`, run the policy (or replay `actions`) n_steps; returns dict of per-step logs."""
    env.set_state(state)
    runner.reset(env)
    safety.reset(env.data.qpos[env.arm_qadr])
    log = {k: [] for k in ("tcp", "blue", "prog", "q_pol", "q_safe", "grip", "qmeas")}
    for t in range(n_steps):
        if use_policy:
            q_pol, grip = runner.act(env)
        else:
            a = actions[t]
            q_pol, grip = np.r_[a[:7], a[8:15]], a[[7, 15]]
        q, _ = safety.filter(q_pol, env.data.qpos[env.arm_qadr], env.data.qpos[env.finger_qadr])
        log["qmeas"].append(env.data.qpos[env.arm_qadr].copy())
        env.apply_action(q, grip)
        obs = env.observe()
        log["tcp"].append(np.stack([obs["ee_pose_left"][:3], obs["ee_pose_right"][:3]]))
        log["blue"].append(obs["objects"][1, :3].copy()); log["prog"].append(env.progress())
        log["q_pol"].append(q_pol); log["q_safe"].append(q); log["grip"].append(grip)
        env.step()
        if stop_on_click and env.progress() >= 1:
            break
    return {k: np.array(v) for k, v in log.items()}
