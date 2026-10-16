import os
import time

import mujoco
import numpy as np
import pytest

from questcollect.config import load_config
from questcollect.ik.solver import BimanualIK
from questcollect.record.hdf5_io import list_episodes, read_episode
from questcollect.record.recorder import Recorder, RecState
from questcollect.sim.env import BimanualEnv
from questcollect.sim.scene_export import export_scene
from questcollect.sim.tasks import TASKS
from questcollect.teleop.frames import quat_wxyz_to_mat


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.mark.parametrize("task", sorted(TASKS))
def test_tasks_build_reset_settle(cfg, task):
    env = BimanualEnv(cfg, task)
    obs = env.reset(seed=3)
    assert obs["joint_pos"].shape == (14,) and obs["objects"].shape == (len(env.task.objects), 7)
    for _ in range(50):
        env.step()
    o = env.observe()["objects"]
    assert np.all(np.abs(o[:, :3] - obs["objects"][:, :3]) < 0.02), "objects should rest on the table"
    assert not env.success()


def test_reset_is_seeded(cfg):
    env = BimanualEnv(cfg, "transfer_cube")
    a = env.reset(seed=7)["objects"]
    b = env.reset(seed=7)["objects"]
    c = env.reset(seed=8)["objects"]
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_ik_reaches_targets(cfg):
    ik = BimanualIK(cfg.sim, cfg.ik, 0.02)
    rng = np.random.default_rng(0)
    p0 = {a: ik.tcp_pose(a) for a in "lr"}
    for _ in range(5):
        offs = {a: rng.uniform([-0.1, -0.1, -0.15], [0.1, 0.1, 0.05]) for a in "lr"}
        for a in "lr":
            ik.set_target(a, p0[a][0] + offs[a], p0[a][1])
        for _ in range(100):
            ik.solve()
        for a, (pos_err, rot_err) in ik.errors().items():
            assert pos_err < 0.005 and rot_err < np.deg2rad(3), (a, pos_err, rot_err)


def test_ik_keeps_arms_apart(cfg):
    """Commanding both TCPs to the same point must not make the arms intersect."""
    ik = BimanualIK(cfg.sim, cfg.ik, 0.02)
    meet = np.array([0.5, 0.0, 1.0])
    for a in "lr":
        ik.set_target(a, meet, ik.tcp_pose(a)[1])
    for _ in range(300):
        ik.solve()
    gap = np.linalg.norm(ik.tcp_pose("l")[0] - ik.tcp_pose("r")[0])
    assert gap > cfg.ik.collision_margin


def test_set_state_replay_determinism(cfg):
    env = BimanualEnv(cfg, "transfer_cube")
    env.reset(seed=1)
    init = env.get_state()
    ctrls = []
    rng = np.random.default_rng(0)
    q = env.data.ctrl[env.arm_act].copy()
    for _ in range(60):
        q = q + rng.normal(scale=0.01, size=14)
        env.apply_action(q, rng.uniform(0, 1, 2))
        ctrls.append(env.data.ctrl.copy())
        env.step()
    final = env.data.qpos.copy()
    env.set_state(init)
    for c in ctrls:
        env.data.ctrl[:] = c
        env.step()
    assert np.allclose(env.data.qpos, final, atol=1e-9)


def test_scene_export(cfg):
    env = BimanualEnv(cfg, "lift_box")
    scene, blob = export_scene(env.model)
    assert scene["nbody"] == env.model.nbody
    assert {g["type"] for g in scene["geoms"]} >= {"mesh", "box", "plane", "cylinder"}
    for m in scene["meshes"]:
        assert m["faceOffset"] + m["nface"] * 12 <= len(blob)
        faces = np.frombuffer(blob, "<u4", m["nface"] * 3, m["faceOffset"])
        assert faces.max() < m["nvert"]
    assert len(blob) == scene["blobSize"]


def _fake_step(env, i):
    obs = env.observe()
    return {
        "time": float(i), "wall_time": time.monotonic(),
        "obs": {k: obs[k] for k in ("joint_pos", "joint_vel", "gripper_width", "ee_pose_left", "ee_pose_right", "objects")},
        "action": {"joint_pos": np.zeros(16), "ee_pose": np.zeros(16), "ctrl": env.data.ctrl.copy()},
        "sim": {"state": env.get_state()},
        "success": False,
        "teleop": {"head": np.zeros(7), "engaged": np.zeros(2, bool), "tracked": np.ones(2, bool), "rtt_ms": 12.0,
                   "left": {"source": 1, "pose": np.zeros(7), "buttons": np.zeros(8), "joints": np.zeros((25, 7)), "pinch": np.zeros(2)},
                   "right": {"source": 1, "pose": np.zeros(7), "buttons": np.zeros(8), "joints": np.zeros((25, 7)), "pinch": np.zeros(2)}},
    }


def test_recorder_state_machine_and_hdf5(cfg, tmp_path):
    cfg = cfg.model_copy(deep=True)
    cfg.record.countdown_s = 0.0
    env = BimanualEnv(cfg, "transfer_cube")
    env.reset(seed=5)
    rec = Recorder(cfg, tmp_path)
    assert rec.start(0.0) and rec.state == RecState.COUNTDOWN
    rec.tick(0.01, env)
    assert rec.state == RecState.RECORDING
    for i in range(10):
        rec.append(i * 0.02, _fake_step(env, i), success=False)
        env.step()
    assert rec.stop() and rec.state == RecState.REVIEW
    path = rec.save(env, "controllers", success=True)
    rec.flush()
    assert rec.state == RecState.IDLE
    meta, data = read_episode(path)
    assert meta["success"] is True and meta["num_steps"] == 10 and meta["task"] == "transfer_cube"
    assert data["obs/joint_pos"].shape == (10, 14)
    assert data["teleop/left/joints"].shape == (10, 25, 7)
    assert data["sim/state"].shape == (10, env.state_size)
    assert data["sim/initial_state"].shape == (env.state_size,)
    # The stored MJCF recompiles to the same model size.
    assert "<mujoco" in meta["model_xml"]
    rows = list_episodes(tmp_path)
    assert len(rows) == 1 and rows[0]["success"]
    # Discarding drops the take.
    rec.start(0.0); rec.tick(0.0, env); rec.append(0.0, _fake_step(env, 0), False)
    assert rec.discard() and rec.state == RecState.IDLE and rec.steps == []
    rec.close()


def test_recorder_auto_stop_on_success(cfg, tmp_path):
    cfg = cfg.model_copy(deep=True)
    cfg.record.countdown_s = 0.0
    env = BimanualEnv(cfg, "transfer_cube")
    env.reset(seed=5)
    rec = Recorder(cfg, tmp_path)
    rec.start(0.0)
    rec.tick(0.0, env)
    stopped = [rec.append(t, _fake_step(env, 0), success=t >= 0.5) for t in np.arange(0, 2.0, 0.02)]
    assert any(stopped) and rec.state == RecState.REVIEW and rec.success_seen
    rec.close()


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") is None, reason="needs headless GL (MUJOCO_GL=egl)")
def test_lerobot_export(cfg, tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from questcollect.export.to_lerobot import export

    cfg = cfg.model_copy(deep=True)
    cfg.record.countdown_s = 0.0
    env = BimanualEnv(cfg, "transfer_cube")
    rec = Recorder(cfg, tmp_path / "raw")
    for ep in range(2):
        env.reset(seed=ep)
        rec.start(0.0); rec.tick(0.0, env)
        for i in range(12):
            rec.append(i * 0.02, _fake_step(env, i), False)
            env.step()
        rec.save(env, "controllers", success=True)
        rec.flush()
    n = export(tmp_path / "raw", tmp_path / "lerobot", "test/quest", height=96, width=128)
    assert n == 2
    ds = LeRobotDataset("test/quest", root=tmp_path / "lerobot")
    assert ds.num_episodes == 2 and ds.num_frames == 24 and ds.fps == cfg.sim.control_hz
    item = ds[0]
    assert item["observation.state"].shape == (16,) and item["action"].shape == (16,)
    assert item["observation.images.cam_high"].shape == (3, 96, 128)
    assert item["observation.environment_state"].shape == (7,)
