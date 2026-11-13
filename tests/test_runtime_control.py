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


def test_takeover_keeps_the_policy_grippers_until_the_trigger_matches(rt):
    rt.control_mode = "shared"
    rt.policy.act = lambda env: (rt.policy.q.copy(), np.zeros(2))  # the policy holds both grippers closed
    _ticks(rt, 120, _controllers())  # the finger targets move at 50 mm/s: fully closed after ~1 s
    closed = rt.env.data.ctrl[rt.env.grip_act].copy()
    assert np.allclose(closed, 0)
    _ticks(rt, 3, _controllers(squeeze_left=1.0))  # takeover with both Triggers released (= open)
    _ticks(rt, 30, _controllers())
    assert rt._driver == "operator"
    assert np.allclose(rt.env.data.ctrl[rt.env.grip_act], closed)  # nothing dropped
    f = _controllers()
    f.left.buttons[0] = 1.0  # left Trigger pulled: matches the closed gripper, so it follows the Trigger again
    _ticks(rt, 2, f)
    _ticks(rt, 30, _controllers())  # ... and opens when released
    assert rt._grip_latch[0] is None and rt._grip_latch[1] is not None
    assert rt.env.data.ctrl[rt.env.grip_act][0] > closed[0] and np.isclose(rt.env.data.ctrl[rt.env.grip_act][1], closed[1])


def test_policy_modes_need_a_policy(rt):
    rt.policy = None
    rt._handle("set_control_mode", {"mode": "policy"}, 0.0)
    assert rt.control_mode == "teleop"


def test_operator_drives_through_the_safety_filter(rt):
    """Steering the left hand through the right arm: the safety filter stops it short instead of colliding."""
    _ticks(rt, 5, _controllers(squeeze_left=1.0))  # Grip press engages the left arm
    _ticks(rt, 5, _controllers())
    assert rt._engaged["l"]
    blocked = []
    for k in range(150):
        f = _controllers()
        f.left.pose[0] = -0.2 + min(k, 100) * 0.006  # 0.6 m towards the right arm (XR +x = world -y) in 2 s
        _ticks(rt, 1, f)
        blocked.append(rt._safety_rad)
        _, arms_touch, _ = rt.env.contact_flags()
        assert not arms_touch
    assert max(blocked) > 0.05  # it did have to step in


def test_operator_fingers_stay_off_the_table(rt):
    """Pushing the hand straight down past the table top: the fingers stop safety.finger_table_margin above it."""
    import mujoco
    m, d = rt.env.model, rt.env.data
    fingers = [g for g in range(m.ngeom) if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name.startswith("l_")
               and m.body(m.geom_bodyid[g]).name.endswith("finger")]
    table = m.geom("table_top").id
    _ticks(rt, 5, _controllers(squeeze_left=1.0))
    _ticks(rt, 5, _controllers())
    gap = []
    for k in range(200):
        f = _controllers()
        f.left.pose[1] = 1.1 - min(k, 120) * 0.006  # down 0.72 m: the target ends far below the table top
        _ticks(rt, 1, f)
        assert not rt.env.contact_flags()[2]
        gap.append(min(mujoco.mj_geomDistance(m, d, g, table, 0.05, None) for g in fingers))
    assert min(gap) > 0.5 * rt.cfg.safety.finger_table_margin
    assert gap[-1] < 2 * rt.cfg.safety.finger_table_margin  # it did get down to the margin


def test_review_summary(rt):
    """After a take stops, status.review judges it with the curation rules."""
    rt.cfg.record.countdown_s = 0
    rt.command("record_toggle")
    _ticks(rt, 80, _controllers())
    assert rt.recorder.recording
    rt.command("record_toggle")
    _ticks(rt, 10, _controllers())
    st = rt.status()
    assert st["rec_state"] == "review"
    review = st["review"]
    assert review["duration_s"] == pytest.approx(len(rt.recorder.steps) / rt.cfg.sim.control_hz, abs=0.1)
    assert review["rejected"] == "failed take" and review["used_s"] == 0  # the tower was not built
    rt.command("discard")
    _ticks(rt, 10)
    assert rt.status()["review"] is None


def test_replay_pause_seek_and_hold(rt, tmp_path):
    """A replay can start paused, seek and resume; it holds the last frame; recording ends it."""
    rt.cfg.record.countdown_s = 0
    rt.recorder.out_dir = tmp_path
    rt.command("record_toggle")
    _ticks(rt, 30, _controllers())
    rt.command("save", success=True)
    _ticks(rt, 1)
    rt.recorder.flush()
    path = str(next(tmp_path.glob("episode_*.hdf5")))
    rt.command("replay", path=path, paused=True, i=5)
    _ticks(rt, 5)
    assert rt.status()["replay"] == {"path": path, "i": 5, "n": 30, "paused": True, "speed": 1.0, "loop": False}
    rt.command("replay_options", speed=0.5)
    rt.command("replay_pause")
    _ticks(rt, 4)
    assert rt._replay["i"] == 7  # half speed: one recorded step every other tick
    rt.command("replay_options", speed=2.0, loop=True)
    _ticks(rt, 13)
    assert rt._replay["i"] == 2 and not rt._replay["paused"]  # 7 + 2 * 11 = 29 (last), wraps to 0, then 2
    rt.command("replay_options", speed=1.0, loop=False)
    rt.command("replay_pause", paused=True)
    _ticks(rt, 1)
    rt.command("replay_seek", i=26)
    rt.command("replay_pause")
    _ticks(rt, 10)
    assert rt.status()["replay"]["i"] == 29 and rt.status()["replay"]["paused"]  # held on the last frame
    rt.command("record_toggle")
    _ticks(rt, 2)
    assert rt._replay is None and rt.recorder.recording


def test_take_over_from_a_replayed_frame(rt, tmp_path):
    """"Take over here": the new take starts at exactly the replayed state, the scene holds still through the
    countdown, and the file says where it came from."""
    from questcollect.recorder import read_episode

    rt.recorder.out_dir = tmp_path
    rt.cfg.record.countdown_s = 0
    rt.control_mode = "policy"  # something that moves the arms
    rt.command("record_toggle")
    _ticks(rt, 40)
    rt.command("save", success=False)
    _ticks(rt, 1)
    rt.recorder.flush()
    src = str(next(tmp_path.glob("episode_*.hdf5")))
    _, data = read_episode(src, keys=["sim/state"])

    rt.control_mode = "teleop"
    rt.command("replay", path=src, paused=True, i=20)
    _ticks(rt, 2)
    rt._replay["meta"]["scene_hash"] = "another scene"  # recorded elsewhere: may be replayed, not continued
    rt.command("replay_branch")
    _ticks(rt, 2)
    assert rt._replay is not None and rt._branch is None and rt.recorder.state.value == "idle"
    rt._replay["meta"]["scene_hash"] = rt.env.scene_hash

    rt.cfg.record.countdown_s = 0.5
    t = time.monotonic()
    rt.command("replay_branch")
    for k in range(10):  # 0.2 s of the countdown: the scene holds at the frame
        rt.submit_input(_controllers())
        rt._tick(t + 0.02 * k)
    assert rt._replay is None and rt.recorder.state.value == "countdown"
    assert rt.status()["branch"] == {"of": src, "step": 20, "frozen": True}
    assert np.array_equal(rt.env.get_state(), data["sim/state"][20])
    for k in range(30):
        rt.submit_input(_controllers())
        rt._tick(t + 0.6 + 0.02 * k)
    assert rt.recorder.recording and not rt._branch["frozen"]
    rt.command("save", success=False)
    _ticks(rt, 1)
    rt.recorder.flush()
    new = sorted(tmp_path.glob("episode_*.hdf5"))[-1]
    meta, out = read_episode(new, keys=["sim/state"])
    assert meta["branch_of"] == src and meta["branch_step"] == 20
    assert np.array_equal(out["sim/state"][0], data["sim/state"][20])
    assert rt._branch is None  # the next take is a fresh one


def test_policy_rollouts_are_listed_read_only(rt, tmp_path):
    """Saved eval rollouts (outputs/<run>/eval/<time>) show up as datasets to replay and take over from; labels
    can't be changed there and nothing is written into the folder."""
    from fastapi.testclient import TestClient

    from questcollect.server import create_app

    rec, runs = tmp_path / "data" / "episodes", tmp_path / "outputs"
    rec.mkdir(parents=True)
    rt.recorder.out_dir = rec
    rt.cfg.record.out_dir, rt.cfg.learn.runs_dir = str(rec), str(runs)
    rt.cfg.record.countdown_s = 0
    rt.command("record_toggle")
    _ticks(rt, 20, _controllers())
    rt.command("save", success=False)
    _ticks(rt, 1)
    rt.recorder.flush()
    ev = runs / "run1" / "eval" / "20261113-120000"
    ev.mkdir(parents=True)
    next(rec.glob("episode_*.hdf5")).rename(ev / "episode_000000.hdf5")
    (ev / "metrics.json").write_text('{"summary": {"success_rate": 0.0, "stage_rates": {"base_in_area": 0.5}}}')
    before = sorted(p.name for p in ev.iterdir())

    client = TestClient(create_app(rt))
    rows = client.get("/api/datasets").json()
    ds = next(d for d in rows if d["readonly"])
    assert ds["name"] == "eval/run1/20261113-120000" and ds["count"] == 1 and "base in area 50%" in ds["description"]
    eps = client.get("/api/episodes", params={"dataset": ds["name"]}).json()
    assert [e["file"] for e in eps] == ["episode_000000.hdf5"]
    assert client.get("/api/audit", params={"dataset": ds["name"]}).status_code == 200
    r = client.patch("/api/episodes/episode_000000.hdf5", params={"dataset": ds["name"]}, json={"note": "x"})
    assert r.status_code == 403
    assert client.get("/api/episodes", params={"dataset": "eval/../x"}).status_code == 404
    assert sorted(p.name for p in ev.iterdir()) == before  # no audit cache, no curation.json
