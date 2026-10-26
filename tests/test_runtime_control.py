"""Control arbitration in the runtime: a policy drives, the operator takes over (shared mode / DAgger)."""

import time

import numpy as np
import pytest

from questcollect.config import load_config
from questcollect.protocol import SRC_CONTROLLER, HandInput, InputFrame
from questcollect.recorder import SOURCE_OPERATOR, SOURCE_POLICY
from questcollect.runtime import Runtime
from questcollect.sim import HOME_Q


class StubPolicy:
    """Stands in for learn.policy.PolicyRunner: always asks for the same joint targets."""
    name = "stub"

    def __init__(self, q):
        self.q, self.resets = q, 0

    def reset(self, env=None):
        self.resets += 1

    def act(self, env):
        return self.q.copy(), np.ones(2)


def _controllers(squeeze_left=0.0) -> InputFrame:
    f = InputFrame()
    for side, x in (("left", -0.2), ("right", 0.2)):
        h = HandInput(source=SRC_CONTROLLER)
        h.pose = np.array([x, 1.1, -0.35, 0, 0, 0, 1.0])
        setattr(f, side, h)
    f.left.buttons[1] = squeeze_left
    return f


def _ticks(rt, n, frame=None):
    for _ in range(n):
        if frame is not None:
            rt.submit_input(frame)
        rt._tick(time.monotonic())


@pytest.fixture
def rt():
    cfg = load_config().model_copy(deep=True)
    rt = Runtime(cfg)
    rt.policy = StubPolicy(np.tile(HOME_Q, 2) + np.r_[0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    return rt


def test_policy_drives_through_the_safety_filter(rt):
    rt.control_mode = "policy"
    _ticks(rt, 60)
    assert rt._driver == "policy" and rt.policy.resets == 1
    assert np.allclose(rt.env.data.qpos[rt.env.arm_qadr], rt.policy.q, atol=0.02)  # free motion: unchanged


def test_shared_takeover_does_not_jump_and_is_recorded(rt):
    rt.control_mode = "shared"
    rt.recorder.cfg.record.countdown_s = 0.0
    rt._handle("record_toggle", {}, time.monotonic())
    _ticks(rt, 60, _controllers())
    assert rt._driver == "policy"
    tcp = rt.env.data.site_xpos[rt.env.tcp_sites].copy()
    _ticks(rt, 3, _controllers(squeeze_left=1.0))  # Grip: the operator takes the left arm
    _ticks(rt, 20, _controllers())                  # released button, arm still engaged (toggle)
    assert rt._driver == "operator"
    assert np.linalg.norm(rt.env.data.site_xpos[rt.env.tcp_sites] - tcp, axis=1).max() < 0.005
    sources = [s["control"]["source"] for s in rt.recorder.steps]
    assert sources[0] == SOURCE_POLICY and sources[-1] == SOURCE_OPERATOR
    _ticks(rt, 3, _controllers(squeeze_left=1.0))  # Grip again: released, the policy resumes with a fresh chunk
    _ticks(rt, 2, _controllers())
    assert rt._driver == "policy" and rt.policy.resets == 2


def test_policy_modes_need_a_policy(rt):
    rt.policy = None
    rt._handle("set_control_mode", {"mode": "policy"}, 0.0)
    assert rt.control_mode == "teleop"
