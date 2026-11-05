import numpy as np
import pytest

from questcollect.config import load_config
from questcollect.ik import BimanualIK
from questcollect.recorder import episode_meta, make_step, read_episode, teleop_fields, write_episode
from questcollect.retime import Move, _moves, _plan, _Streams, _Timeline
from questcollect.sim import ARMS, ArmClearance, BimanualEnv

FPS = 50


def _streams(left, right, left_xy=(0.2, 0.3), right_xy=(0.2, -0.3)) -> _Streams:
    n = len(left)
    a = np.zeros((n, 16))
    a[:, 0], a[:, 8] = left, right
    a[:, 7] = a[:, 15] = 1.0
    ee = np.zeros((n, 16))
    ee[:, 0:2], ee[:, 7:9] = left_xy, right_xy
    return _Streams({"action/joint_pos": a, "action/ee_pose": ee})


def test_moves_are_split_by_holds_and_know_their_zone_part():
    t = np.arange(300) / FPS
    left = np.where(t < 1, 0.5 * t, np.where(t < 4, 0.5, 0.5 + 0.5 * (t - 4)))  # moves, holds 3 s, moves
    s = _streams(left, np.zeros(300))
    s.ee["l"][150:, :2] = 0.0  # the second move happens at the tower
    moves = _moves(s, np.zeros(2), FPS)
    (m1, m2) = [m for m in moves if m.arm == "l"]
    assert abs(m1.e - 50) <= 1 and abs(m2.s - 200) <= 1 and not m1.zone
    assert m2.zone and m2.zone_s == 200 and m2.zone_e == 300
    assert [m for m in moves if m.arm == "r"] == []


def test_plan_overlaps_the_arms_and_keeps_the_tower_zone_in_order():
    # Recorded one after the other: left 0-100 (zone 50-100), right 200-300 (zone 250-300), right 400-450.
    moves = [Move("l", 0, 100, 50, 100), Move("r", 200, 300, 250, 300), Move("r", 400, 450)]
    tl = _plan(moves, 500, FPS, lambda arm, own, other: False, {})
    starts = {(m.arm, m.s): t for a in ARMS for t, m in tl[a].items}
    # Right starts its approach while left is still at the tower, and reaches the zone once left has left it.
    assert starts[("r", 200)] < 100 and starts[("r", 200)] + 50 >= 100
    assert starts[("r", 400)] < 400
    idx = tl["r"].indices(max(t.end for t in tl.values()))
    assert idx[starts[("r", 200)]] == 200 and idx[-1] == 449


def test_plan_waits_while_the_arms_would_come_too_close():
    moves = [Move("l", 0, 100), Move("r", 200, 300)]
    # Too close while left is in its move (frames 0-99) and right has started (frames >= 200).
    too_close = lambda arm, own, other: bool(np.any((own < 100) & (other >= 200)) if arm == "l"  # noqa: E731
                                            else np.any((other < 99) & (own >= 200)))
    tl = _plan(moves, 300, FPS, too_close, {})
    assert tl["r"].items[0][0] >= 100


def test_timeline_holds_are_time_warped_source_holds():
    tl = _Timeline(100)
    tl.items = [(0, Move("l", 0, 10)), (15, Move("l", 50, 60))]  # a 40-frame source hold played in 5 steps
    idx = tl.indices(30)
    assert idx[:10].tolist() == list(range(10)) and idx[15:25].tolist() == list(range(50, 60))
    assert idx[10:15].tolist() == [10, 18, 26, 34, 42] and idx[-1] == 64


def test_clearance_sees_arms_apart_at_home():
    cfg = load_config()
    env = BimanualEnv(cfg)
    env.reset(seed=1)
    d_arm, d_cam = ArmClearance(cfg.sim).measure(env.data.qpos[env.arm_qadr][None])[0]
    assert d_arm > 0.1 and d_cam > 0.1


def test_recorded_actions_replay_exactly(tmp_path):
    """What retiming relies on: re-executing action/joint_pos from sim/state[0] reproduces the recorded states."""
    cfg = load_config()
    env = BimanualEnv(cfg)
    env.reset(seed=4)
    ik = BimanualIK(cfg.sim, cfg.ik, env.control_dt)
    ik.reset(env.data.qpos[env.arm_qadr])
    steps = []
    for k in range(80):
        p, q = ik.tcp_pose("l")
        ik.set_target("l", p + [0.0, 0.0, -0.002], q)
        ik.follow(env.data.qpos[env.arm_qadr])
        target, grip = ik.solve(), np.array([1.0 - k / 80, 1.0])
        obs, state, flags = env.observe(), env.get_state(), env.contact_flags()
        env.apply_action(target, grip)
        steps.append(make_step(env, obs, state, target, grip, [np.zeros(7)] * 2, False, 0, flags,
                               teleop_fields(None)))
        env.step()
    path = write_episode(tmp_path / "episode_000000.hdf5", steps, episode_meta(cfg, env, 80, False, "controllers", "t"))
    _, d = read_episode(path)
    S, A = d["sim/state"], d["action/joint_pos"]
    env.set_state(S[0])
    for t in range(len(A) - 1):
        env.apply_action(np.r_[A[t, :7], A[t, 8:15]], A[t, [7, 15]])
        env.step()
        assert np.array_equal(env.get_state(), S[t + 1]), t
    assert pytest.approx(d["action/joint_pos"][-1, 7], abs=1e-6) == 1.0 - 79 / 80
