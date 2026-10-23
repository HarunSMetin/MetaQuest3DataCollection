import os
import time

import numpy as np
import pytest

from questcollect.config import load_config
from questcollect.ik import BimanualIK
from questcollect.recorder import Recorder, RecState, list_episodes, read_episode
from questcollect.runtime import HAPTIC_CLICK, HAPTIC_COMPLETE, HAPTIC_PLACED, HAPTIC_UNCLICK, Runtime
from questcollect.scene_export import export_scene
from questcollect.sim import (BRICK_H, BRICKS, BUILD_AREA_CENTER, BUILD_AREA_HALF, CLUTCH_PRESS_N, CLUTCH_PULL_N,
                              HOME_Q, LEVELS, PITCH, BimanualEnv, set_brick_pose, task_spec, tower_poses)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def env(cfg):
    return BimanualEnv(cfg)


def test_task_builds_resets_and_settles(env):
    obs = env.reset(seed=3)
    assert obs["joint_pos"].shape == (14,) and obs["objects"].shape == (len(BRICKS), 7)
    for _ in range(50):
        env.step()
    o = env.observe()["objects"]
    assert np.all(np.abs(o[:, :3] - obs["objects"][:, :3]) < 0.002), "bricks should rest on the table"
    assert np.allclose(o[:, 2], env.table_height, atol=0.001)
    assert env.progress() == 0 and not env.success()


def _push(env, brick: int, force, steps: int = 10, hold_base: bool = False) -> list:
    """Apply a world-frame force (N) to one brick for `steps` control ticks; returns the clutch events."""
    d, m = env.data, env.model
    bid, base = m.body(BRICKS[brick].name).id, m.body(BRICKS[0].name).id
    events = []
    for _ in range(steps):
        d.xfrc_applied[bid, :3] = force
        if hold_base:
            d.xfrc_applied[base, :3] = [0, 0, -60]
        env.step()
        events += env.clutch_events
    d.xfrc_applied[:] = 0
    return events


def _build(env, base_xy, base_yaw, order=None, offset=(0.0, 0.0), press=True):
    """Teleport the bricks into the goal tower (optionally in another order / shifted top brick), settle, press."""
    env.reset(seed=1)
    goals = tower_poses([*base_xy, env.table_height], base_yaw)
    order = order or list(range(len(BRICKS)))
    for level, i in enumerate(order):
        p, yaw = goals[level]
        if level == len(order) - 1:
            p = p + [*offset, 0]
        set_brick_pose(env.data, BRICKS[i].name, p + [0, 0, 0.0005], yaw)
    for _ in range(25):
        env.step()
    if press:
        _push(env, order[-1], [0, 0, -15])
    for _ in range(25):
        env.step()


@pytest.mark.parametrize("yaw", [0.0, 0.7, 0.7 + np.pi])
def test_tower_complete_is_success(env, yaw):
    _build(env, (0.45, 0.0), yaw)
    assert env.locked_pairs() == [(0, 1), (1, 2), (2, 3)]
    assert env.progress() == LEVELS and env.success()
    top = env.data.body(BRICKS[-1].name).xpos
    assert abs(top[2] - (env.table_height + LEVELS * BRICK_H)) < 0.001, "studs must seat, not stand on each other"


def test_tower_outside_build_area_does_not_count(env):
    """The same clicked tower built next to the taped area is not progress; one corner over the tape is enough."""
    _build(env, (0.45, 0.25), 0.0)
    assert env.locked_pairs() == [(0, 1), (1, 2), (2, 3)]
    assert not env.in_area() and env.progress() == 0 and not env.success()
    edge = BUILD_AREA_CENTER[0] + BUILD_AREA_HALF[0] - BRICKS[0].half[0] + 0.003  # base pokes 3 mm out
    _build(env, (edge, 0.0), 0.0)
    assert not env.in_area() and env.progress() == 0


def test_bricks_never_start_in_build_area(env):
    for seed in range(30):
        env.reset(seed=seed)
        for b in BRICKS:
            xy = env.data.body(b.name).xpos[:2]
            gap = np.maximum(np.abs(xy - BUILD_AREA_CENTER) - BUILD_AREA_HALF, 0)
            assert np.linalg.norm(gap) > 0.02, (seed, b.name)


def test_task_spec_describes_goal():
    """The UI draws the goal from task_spec alone, so it must match the tower the success check uses."""
    spec = task_spec()
    assert [g["brick"] for g in spec["goal"]] == list(range(len(BRICKS)))
    for g, (pos, yaw) in zip(spec["goal"], tower_poses([0, 0, 0], 0.0)):
        assert np.allclose(g["pos"], pos) and g["yaw"] == pytest.approx(yaw)
    assert spec["area"]["center"] == list(BUILD_AREA_CENTER)
    assert len(spec["bricks"]) == len(BRICKS) and len(spec["bricks"][0]["size"]) == 3


def test_tower_must_be_pressed(env):
    """Bricks that are only set down don't count: like real DUPLO, they have to be pressed until they click."""
    _build(env, (0.45, 0.0), 0.0, press=False)
    assert env.locked_pairs() == [] and env.progress() == 0 and not env.success()
    events = _push(env, len(BRICKS) - 1, [0, 0, -15])
    assert [e[0] for e in events] == ["lock"] * LEVELS
    assert env.progress() == LEVELS


def test_light_press_does_not_click(env):
    _build(env, (0.45, 0.0), 0.0, press=False)
    _push(env, len(BRICKS) - 1, [0, 0, -0.5 * CLUTCH_PRESS_N])
    assert env.locked_pairs() == []


def test_clutch_holds_until_pulled_hard(env):
    _build(env, (0.45, 0.0), 0.0)
    top = len(BRICKS) - 1
    assert _push(env, top, [0, 0, 0.5 * CLUTCH_PULL_N], steps=25, hold_base=True) == []
    assert env.locked_pairs() == [(0, 1), (1, 2), (2, 3)]
    events = _push(env, top, [0, 0, 2 * CLUTCH_PULL_N], steps=25, hold_base=True)
    assert ("unlock", 2, 3) in events
    assert env.data.body(BRICKS[top].name).xpos[2] > env.table_height + LEVELS * BRICK_H + 0.01


def test_hard_press_keeps_tower_clicked_and_rigid(env):
    """Pressing a clicked tower hard (like an arm pushing down) neither pops a click nor squashes the bricks."""
    _build(env, (0.45, 0.0), 0.0)
    m, d = env.model, env.data
    bricks = {m.body(b.name).id for b in BRICKS}
    deepest = 0.0
    for _ in range(25):
        _push(env, len(BRICKS) - 1, [0, 0, -60], steps=1)
        for c in d.contact[:d.ncon]:
            if m.geom_bodyid[c.geom1] in bricks and m.geom_bodyid[c.geom2] in bricks:
                deepest = max(deepest, -c.dist)
    assert env.locked_pairs() == [(0, 1), (1, 2), (2, 3)]
    assert deepest < 0.0005, f"bricks sank {deepest * 1e3:.2f} mm into each other"


def test_locked_tower_lifts_as_one(env):
    """Lifting a clicked brick carries the bricks clicked under it (they don't stay on the table)."""
    _build(env, (0.45, 0.0), 0.0)
    _push(env, 1, [0, 0, 1.0], steps=25)  # 1 N up on the blue brick; the stack weighs ~0.36 N
    assert env.data.body(BRICKS[0].name).xpos[2] > env.table_height + 0.02
    assert len(env.locked_pairs()) == LEVELS


def test_wrong_tower_is_not_success(env):
    _build(env, (0.45, 0.0), 0.0, order=[0, 2, 1, 3])  # yellow and blue swapped
    assert env.progress() == 0 and not env.success()
    # Blue clicked onto red across it (turned 90 degrees, on the stud grid, stable): a click, not progress.
    env.reset(seed=1)
    th = env.table_height
    set_brick_pose(env.data, BRICKS[0].name, [0.45, 0.0, th])
    set_brick_pose(env.data, BRICKS[1].name, [0.45, PITCH / 2, th + BRICK_H + 0.0005], np.pi / 2)
    for _ in range(25):
        env.step()
    _push(env, 1, [0, 0, -15])
    assert env.locked_pairs() == [(0, 1)]
    assert env.progress() == 0 and not env.success()


def test_studs_center_an_offset_brick(env):
    """A brick dropped slightly off the stud grid slides into place, and pressing then clicks it in."""
    env.reset(seed=1)
    goals = tower_poses([0.45, 0.0, env.table_height], 0.0)
    set_brick_pose(env.data, BRICKS[0].name, goals[0][0])
    set_brick_pose(env.data, BRICKS[1].name, goals[1][0] + [0.0015, 0.001, 0.003], 0.03)
    for _ in range(100):
        env.step()
    assert np.linalg.norm(env.data.body(BRICKS[1].name).xpos - goals[1][0]) < 0.001
    _push(env, 1, [0, 0, -15])
    assert env.progress() == 1


def test_state_roundtrip_keeps_clicks(env):
    """sim/state carries the clutch welds, so replaying a recorded state reproduces the clicked tower."""
    _build(env, (0.45, 0.0), 0.4)
    state, pos = env.get_state(), env.observe()["objects"]
    env.reset(seed=3)
    assert env.locked_pairs() == []
    env.set_state(state)
    assert env.locked_pairs() == [(0, 1), (1, 2), (2, 3)]
    for _ in range(25):
        env.step()
    assert np.allclose(env.observe()["objects"][:, :3], pos[:, :3], atol=5e-4)


def test_grasp_detection_and_force(cfg, env):
    """Close the left gripper on a brick: held() reports it, the squeeze is the configured grasp force,
    and the brick lifts with the hand."""
    env.reset(seed=1)
    th = env.table_height
    for i, o in enumerate(BRICKS):  # square yellow 2x2 under the left hand, the others out of the way
        set_brick_pose(env.data, o.name, [0.45, 0.25, th] if i == 2 else [0.35 + 0.08 * i, -0.3, th])
    ik = BimanualIK(cfg.sim, cfg.ik, env.control_dt)
    down = np.array([0.0, 1.0, 0.0, 0.0])
    z_grasp = BRICK_H / 2 + 0.002
    for z, grip, n in ((0.1, 1.0, 60), (z_grasp, 1.0, 50), (z_grasp, 0.0, 40), (0.1, 0.0, 50)):
        ik.set_target("l", [0.45, 0.25, th + z], down)
        for _ in range(n):
            env.apply_action(ik.solve(), np.array([grip, 1.0]))
            env.step()
        if grip == 0.0 and z == z_grasp:
            assert env.held() == {"l": 2, "r": None}
    assert env.held()["l"] == 2
    squeeze = -env.data.actuator_force[env.grip_act[0]] / 2  # each finger gets half the tendon force
    assert abs(squeeze - cfg.sim.gripper_force) < 1.0
    assert env.data.body(BRICKS[2].name).xpos[2] > th + 0.05, "brick should be lifted"


def test_fingers_move_at_franka_speed(env):
    env.reset(seed=1)
    width = lambda: env.data.qpos[env.finger_qadr[0]].sum()  # noqa: E731
    w0 = width()
    for _ in range(10):  # 0.2 s
        env.apply_action(np.tile(HOME_Q, 2), np.array([0.0, 1.0]))
        env.step()
    per_finger = (w0 - width()) / 2 / 0.2
    assert 0.04 < per_finger < 0.055, per_finger


def test_reset_is_seeded(env):
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
    """With avoid_arm_collisions, commanding both TCPs to the same point must not make the arms intersect."""
    ik = BimanualIK(cfg.sim, cfg.ik.model_copy(update={"avoid_arm_collisions": True}), 0.02)
    meet = np.array([0.5, 0.0, 1.0])
    for a in "lr":
        ik.set_target(a, meet, ik.tcp_pose(a)[1])
    for _ in range(300):
        ik.solve()
    gap = np.linalg.norm(ik.tcp_pose("l")[0] - ik.tcp_pose("r")[0])
    assert gap > cfg.ik.collision_margin


def test_set_state_replay_determinism(env):
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


def test_scene_export(env):
    scene, blob = export_scene(env.model)
    assert scene["nbody"] == env.model.nbody
    assert {g["type"] for g in scene["geoms"]} >= {"mesh", "box", "plane"}
    for m in scene["meshes"]:
        assert m["faceOffset"] + m["nface"] * 12 <= len(blob)
        faces = np.frombuffer(blob, "<u4", m["nface"] * 3, m["faceOffset"])
        assert faces.max() < m["nvert"]
    assert len(blob) == scene["blobSize"]


def _fake_step(env, i):
    return {
        "time": float(i), "wall_time": time.monotonic(),
        "obs": env.observe(),
        "action": {"joint_pos": np.zeros(16), "ee_pose": np.zeros(16)},
        "sim": {"state": env.get_state()},
        "success": False,
        "teleop": {"head": np.zeros(7), "engaged": np.zeros(2, bool), "tracked": np.ones(2, bool), "rtt_ms": 12.0,
                   "left": {"source": 1, "pose": np.zeros(7), "buttons": np.zeros(8), "joints": np.zeros((25, 7)), "pinch": np.zeros(2)},
                   "right": {"source": 1, "pose": np.zeros(7), "buttons": np.zeros(8), "joints": np.zeros((25, 7)), "pinch": np.zeros(2)}},
    }


def _no_countdown(cfg):
    cfg = cfg.model_copy(deep=True)
    cfg.record.countdown_s = 0.0
    return cfg


def test_recorder_state_machine_and_hdf5(cfg, env, tmp_path):
    env.reset(seed=5)
    rec = Recorder(_no_countdown(cfg), tmp_path)
    assert rec.start(0.0) and rec.state == RecState.COUNTDOWN
    rec.tick(0.01)
    assert rec.state == RecState.RECORDING
    for i in range(10):
        rec.append(i * 0.02, _fake_step(env, i), success=False)
        env.step()
    assert rec.stop() and rec.state == RecState.REVIEW
    path = rec.save(env, "controllers", success=True)
    rec.flush()
    assert rec.state == RecState.IDLE
    meta, data = read_episode(path)
    assert meta["success"] is True and meta["num_steps"] == 10 and meta["task"] == "build_tower"
    assert data["obs/joint_pos"].shape == (10, 14)
    assert data["teleop/left/joints"].shape == (10, 25, 7)
    assert data["sim/state"].shape == (10, env.state_size)
    assert "<mujoco" in meta["model_xml"]
    rows = list_episodes(tmp_path)
    assert len(rows) == 1 and rows[0]["success"]
    # Discarding drops the take.
    rec.start(0.0); rec.tick(0.0); rec.append(0.0, _fake_step(env, 0), False)
    assert rec.discard() and rec.state == RecState.IDLE and rec.steps == []
    rec.close()


def test_recorder_auto_stop_on_success(cfg, env, tmp_path):
    env.reset(seed=5)
    rec = Recorder(_no_countdown(cfg), tmp_path)
    rec.start(0.0)
    rec.tick(0.0)
    stopped = [rec.append(t, _fake_step(env, 0), success=t >= 0.5) for t in np.arange(0, 2.0, 0.02)]
    assert any(stopped) and rec.state == RecState.REVIEW and rec.success_seen
    rec.close()


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") is None, reason="needs headless GL (MUJOCO_GL=egl)")
def test_lerobot_export(cfg, env, tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from questcollect.lerobot_export import export

    rec = Recorder(_no_countdown(cfg), tmp_path / "raw")
    for ep in range(2):
        env.reset(seed=ep)
        rec.start(0.0); rec.tick(0.0)
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
    assert item["observation.environment_state"].shape == (7 * len(BRICKS),)


def test_tower_haptics(cfg):
    """Each placed level fires the double pulse once (not the plain click), the finished tower the triple pulse
    on both hands, and pulling a brick off the unclick blip."""
    rt = Runtime(cfg)
    events = []
    rt.add_listener(events.append)
    env = rt.env
    for b, (p, yaw) in zip(BRICKS, tower_poses([0.45, 0.0, env.table_height], 0.3)):
        set_brick_pose(env.data, b.name, p + [0, 0, 0.0005], yaw)
    top = env.model.body(BRICKS[-1].name).id
    for t in range(60):
        env.data.xfrc_applied[top, :3] = [0, 0, -15] if 10 <= t < 20 else [0, 0, 0]
        env.step()
        rt._task_feedback(t * 0.02, env.success())
    pulses = lambda: [(e["intensity"], e["ms"], e["repeat"], e["gap_ms"]) for e in events if e["type"] == "haptic"]  # noqa: E731
    assert pulses().count(HAPTIC_PLACED) == 2 * LEVELS  # no holder recorded -> both hands
    assert pulses().count(HAPTIC_CLICK) == 0
    assert pulses().count(HAPTIC_COMPLETE) == 2
    sounds = [e["name"] for e in events if e["type"] == "sound"]
    assert sounds.count("placed") == LEVELS and sounds.count("complete") == 1 and sounds.count("area") == 1
    events.clear()
    _push(env, len(BRICKS) - 1, [0, 0, 2 * CLUTCH_PULL_N], steps=1, hold_base=True)
    rt._task_feedback(2.0, env.success())
    assert HAPTIC_UNCLICK in pulses()
    sounds = [e["name"] for e in events if e["type"] == "sound"]
    assert "unclick" in sounds
