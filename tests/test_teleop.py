import time

import numpy as np
import pytest

from questcollect.config import TeleopConfig
from questcollect.server.protocol import (
    SRC_CONTROLLER, SRC_HAND, HandInput, InputFrame, pack_input, parse_input,
)
from questcollect.teleop.arm_mapper import ArmClutch
from questcollect.teleop.filters import OneEuro, OneEuroRotation
from questcollect.teleop.frames import (
    Placement, mat_to_quat_wxyz, quat_wxyz_to_mat, wxyz_to_xyzw, xyzw_to_wxyz,
)
from questcollect.teleop.hand_mapper import HandMapper
from questcollect.teleop.teleop import Teleop


def random_rot(rng):
    q = rng.normal(size=4)
    return quat_wxyz_to_mat(q / np.linalg.norm(q))


def test_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        R = random_rot(rng)
        assert np.allclose(quat_wxyz_to_mat(mat_to_quat_wxyz(R)), R, atol=1e-9)
    q = np.array([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(wxyz_to_xyzw(xyzw_to_wxyz(q)), q)


def test_placement_axes():
    """XR forward (-z) is world +x, XR right (+x) is world -y, XR up (+y) is world +z."""
    pl = Placement(operator_x=0.0)
    for xr, world in (([0, 0, -1], [1, 0, 0]), ([1, 0, 0], [0, -1, 0]), ([0, 1, 0], [0, 0, 1])):
        assert np.allclose(pl.world_point_from_xr(xr), world)


def test_placement_recenter_roundtrip():
    pl = Placement()
    yaw = 0.7
    head_q = [0, np.sin(yaw / 2), 0, np.cos(yaw / 2)]  # xyzw, rotation about XR +y
    pl.recenter([0.4, 1.7, -0.2], head_q)
    # The head maps to the operator spot behind the arms, looking along world +x.
    p_w = pl.world_point_from_xr([0.4, 1.7, -0.2])
    assert np.allclose(p_w, [pl.operator_x, 0, 1.7], atol=1e-9)
    _, R_w = pl.world_from_xr([0.4, 1.7, -0.2], head_q)
    assert np.allclose(R_w @ [0, 0, -1], [1, 0, 0], atol=1e-9)  # head forward -> world +x
    T = pl.matrix()
    p = np.array([0.3, -0.2, 0.9])
    assert np.allclose(pl.world_point_from_xr((T @ np.append(p, 1))[:3]), p)


def test_clutch_no_jump_on_reengage():
    rng = np.random.default_rng(1)
    c = ArmClutch()
    ee_p, ee_R = np.array([0.5, 0.2, 1.0]), random_rot(rng)
    hand_p, hand_R = np.array([0.0, 0.0, 1.2]), random_rot(rng)
    c.engage(hand_p, hand_R, ee_p, ee_R)
    p, R = c.target(hand_p, hand_R)
    assert np.allclose(p, ee_p) and np.allclose(R, ee_R)
    # Move hand, release, move hand far away, re-engage at the held target: no jump.
    p1, R1 = c.target(hand_p + [0.1, 0, 0], hand_R)
    assert np.allclose(p1, ee_p + [0.1, 0, 0])
    c.release()
    c.engage(hand_p + [1.0, 1.0, 0], random_rot(rng), p1, R1)
    p2, R2 = c.target(hand_p + [1.0, 1.0, 0], c._hR0)
    assert np.allclose(p2, p1) and np.allclose(R2, R1)


def test_clutch_scale_and_rotation():
    c = ArmClutch()
    c.engage(np.zeros(3), np.eye(3), np.array([1.0, 0, 0]), np.eye(3))
    Rz = quat_wxyz_to_mat([np.cos(0.25), 0, 0, np.sin(0.25)])
    p, R = c.target(np.array([0.1, 0.0, 0.0]), Rz, scale=2.0)
    assert np.allclose(p, [1.2, 0, 0])
    assert np.allclose(R, Rz)


def test_one_euro_converges_and_smooths():
    f = OneEuro(min_cutoff=1.0, beta=0.0)
    rng = np.random.default_rng(2)
    out = [f(np.array([1.0]) + rng.normal(scale=0.01, size=1), 0.02) for _ in range(200)]
    assert abs(out[-1][0] - 1.0) < 0.01
    assert np.std([o[0] for o in out[100:]]) < 0.006  # noise attenuated

    fr = OneEuroRotation(min_cutoff=1.0, beta=0.0)
    R_target = quat_wxyz_to_mat([np.cos(0.5), np.sin(0.5), 0, 0])
    fr(np.eye(3), 0.02)
    for _ in range(300):
        R = fr(R_target, 0.02)
    assert np.allclose(R, R_target, atol=1e-3)


def hand_input(pinch_index=0.08, pinch_middle=0.08, pos=(0.0, 1.0, -0.3)) -> HandInput:
    h = HandInput(source=SRC_HAND)
    h.joints[:, 3:] = [0, 0, 0, 1]
    h.joints[:, :3] = pos
    h.pinch[:] = [pinch_index, pinch_middle]
    return h


def test_hand_mapper_clutch_toggle_and_gripper():
    m = HandMapper(TeleopConfig())
    assert m.update(hand_input())[0] is False
    assert m.update(hand_input(pinch_middle=0.01))[0] is True     # pinch -> engage
    assert m.update(hand_input(pinch_middle=0.01))[0] is True     # held: no re-toggle
    assert m.update(hand_input(pinch_middle=0.05))[0] is True     # release latch
    assert m.update(hand_input(pinch_middle=0.01))[0] is False    # second pinch -> disengage
    # Index pinch that brings the thumb near the middle tip is NOT a clutch gesture.
    m.reset()
    assert m.update(hand_input(pinch_index=0.005, pinch_middle=0.015))[0] is False
    cfg = TeleopConfig()
    assert m.gripper(hand_input(pinch_index=cfg.pinch_open_m)) == pytest.approx(1.0)
    assert m.gripper(hand_input(pinch_index=0.0)) == pytest.approx(0.0)


def controller_frame(x=0.2, y=1.1, squeeze=0.0, trigger=0.0) -> InputFrame:
    f = InputFrame(recv_time=time.monotonic())
    for side, xx in (("left", -x), ("right", x)):
        h = HandInput(source=SRC_CONTROLLER)
        h.pose = np.array([xx, y, -0.35, 0, 0, 0, 1.0])
        h.buttons[0], h.buttons[1] = trigger, squeeze
        setattr(f, side, h)
    return f


def make_teleop(mode="controllers") -> Teleop:
    t = Teleop(TeleopConfig(mode=mode))
    t.reset({"l": (np.array([0.46, 0.3, 1.15]), np.eye(3)), "r": (np.array([0.46, -0.3, 1.15]), np.eye(3))})
    return t


def test_teleop_controller_follow_and_hold():
    t = make_teleop()
    for _ in range(5):
        out = t.update(controller_frame(squeeze=1.0), time.monotonic(), 0.02)
    p0 = out.targets["r"][0].copy()
    for k in range(1, 121):  # move 10 cm right + 10 cm down over 0.4 s, then hold still
        a = min(k / 20, 1.0)
        out = t.update(controller_frame(x=0.2 + 0.1 * a, y=1.1 - 0.1 * a, squeeze=1.0, trigger=1.0), time.monotonic(), 0.02)
    assert out.engaged["r"]
    assert np.allclose(out.targets["r"][0] - p0, [0, -0.1, -0.1], atol=1e-3)
    assert out.grippers[1] == pytest.approx(0.0)
    # Release and move: target holds.
    out = t.update(controller_frame(x=0.3, y=1.0), time.monotonic(), 0.02)
    held = out.targets["r"][0].copy()
    for k in range(1, 31):
        out = t.update(controller_frame(x=0.3 + 0.01 * k, y=1.0 - 0.01 * k), time.monotonic(), 0.02)
    assert not out.engaged["r"] and np.allclose(out.targets["r"][0], held)


def test_teleop_stale_input_disengages():
    t = make_teleop()
    f = controller_frame(squeeze=1.0)
    out = t.update(f, time.monotonic(), 0.02)
    assert out.engaged["r"]
    out = t.update(f, f.recv_time + 1.0, 0.02)  # no new input for 1 s
    assert not out.engaged["r"] and not out.tracked["r"]


def test_teleop_jump_rejection():
    t = make_teleop()
    for _ in range(3):
        out = t.update(controller_frame(squeeze=1.0), time.monotonic(), 0.02)
    p0 = out.targets["r"][0].copy()
    out = t.update(controller_frame(x=0.8, squeeze=1.0), time.monotonic(), 0.02)  # 60 cm in one tick
    assert not out.engaged["r"]
    assert np.allclose(out.targets["r"][0], p0)


def test_teleop_mode_mismatch():
    t = make_teleop("hands")
    out = t.update(controller_frame(squeeze=1.0), time.monotonic(), 0.02)
    assert out.mode_mismatch and not out.tracked["r"] and not out.engaged["r"]


def test_workspace_clamp():
    t = make_teleop()
    for _ in range(3):
        t.update(controller_frame(squeeze=1.0), time.monotonic(), 0.02)
    y = 1.1
    for _ in range(200):  # drag down slowly well below the table
        y -= 0.01
        out = t.update(controller_frame(y=y, squeeze=1.0), time.monotonic(), 0.02)
    assert out.targets["r"][0][2] >= TeleopConfig().workspace_min[2] - 1e-9


def test_protocol_roundtrip():
    f = controller_frame(squeeze=0.7, trigger=0.3)
    f.seq, f.t_client, f.t_echo = 42, 1234.5, 999.25
    f.left = hand_input(pinch_index=0.02)
    g = parse_input(pack_input(f))
    assert g.seq == 42 and g.t_client == 1234.5 and g.t_echo == 999.25
    assert g.left.source == SRC_HAND and g.right.source == SRC_CONTROLLER
    assert np.allclose(g.right.buttons[:2], [0.3, 0.7], atol=1e-6)
    assert np.allclose(g.left.pinch[0], 0.02, atol=1e-6)
    with pytest.raises(ValueError):
        parse_input(b"\x01" * 10)
