import mujoco
import numpy as np
import pytest

from questcollect.config import load_config
from questcollect.ik import COLLISION_BODIES, BimanualIK, collision_geoms
from questcollect.safety import FINGER_BODIES, TABLE_BODIES, SafetyFilter

DT = 0.02


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _unsafe_path(cfg, targets: dict, steps: int = 300) -> list[np.ndarray]:
    """Joint targets from the plain IK (no collision avoidance) driving the TCPs to `targets`."""
    ik = BimanualIK(cfg.sim, cfg.ik.model_copy(update={"avoid_arm_collisions": False}), DT)
    for a, p in targets.items():
        ik.set_target(a, np.asarray(p, float), ik.tcp_pose(a)[1])
    return [ik.solve() for _ in range(steps)]


def _run(f: SafetyFilter, path) -> tuple[np.ndarray, list[float]]:
    """Feed the path through the filter with perfect tracking (measured = last safe target)."""
    q = f.configuration.q[f.arm_qadr].copy()
    changes = []
    for q_prop in path:
        q, change = f.filter(q_prop, q)
        changes.append(change)
    return q, changes


def _min_distance(f: SafetyFilter, geoms_a, geoms_b) -> float:
    d = f.configuration.data
    return min(mujoco.mj_geomDistance(f.model, d, g1, g2, 1.0, None) for g1 in geoms_a for g2 in geoms_b)


def test_free_motion_is_untouched(cfg):
    f = SafetyFilter(cfg.sim, cfg.ik, cfg.safety, DT)
    path = _unsafe_path(cfg, {"l": [0.45, 0.3, 1.0], "r": [0.45, -0.3, 1.0]})
    _, changes = _run(f, path)
    assert max(changes[5:]) < 1e-3, max(changes)


def test_arms_kept_apart(cfg):
    f = SafetyFilter(cfg.sim, cfg.ik, cfg.safety, DT)
    path = _unsafe_path(cfg, {"l": [0.5, 0.0, 1.0], "r": [0.5, 0.0, 1.0]})
    _run(f, path)
    m = f.model
    left, right = collision_geoms(m, "l", COLLISION_BODIES), collision_geoms(m, "r", COLLISION_BODIES)
    # The linearized limit stops ~5 mm short of the margin; the point is that the arms never meet.
    assert _min_distance(f, left, right) > cfg.safety.arm_margin / 2
    unsafe = SafetyFilter(cfg.sim, cfg.ik, cfg.safety.model_copy(update={"enabled": False}), DT)
    _run(unsafe, path)
    assert _min_distance(unsafe, left, right) < 0.0  # without the filter the same targets intersect


def test_arm_kept_above_table(cfg):
    f = SafetyFilter(cfg.sim, cfg.ik, cfg.safety, DT)
    path = _unsafe_path(cfg, {"l": [0.45, 0.3, cfg.sim.table_height - 0.08]})
    _run(f, path)
    m, table = f.model, [f.model.geom("table_top").id]
    links = collision_geoms(m, "l", TABLE_BODIES)
    fingers = collision_geoms(m, "l", FINGER_BODIES)
    assert _min_distance(f, links, table) > cfg.safety.table_margin - 0.005
    assert _min_distance(f, fingers, table) > -0.002
