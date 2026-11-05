import numpy as np
import pytest

from questcollect.learn.features import EE_NAMES, FeatureSpec, ee_to_pose, pose_to_ee


def _random_poses(n: int, rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.concatenate([rng.uniform(-1, 1, (n, 3)), q], axis=1)


def _same_rotation(q1, q2) -> bool:
    return abs(abs(float(np.dot(q1, q2))) - 1) < 1e-9


def test_ee_rotation_roundtrip_and_sign_invariance():
    rng = np.random.default_rng(0)
    for pose in _random_poses(200, rng):
        ee = pose_to_ee(pose)
        back = ee_to_pose(ee)
        assert np.allclose(back[:3], pose[:3])
        assert _same_rotation(back[3:], pose[3:])
        flipped = np.concatenate([pose[:3], -pose[3:]])
        assert np.allclose(pose_to_ee(flipped), ee)  # q and -q: the same features


def test_ee_action_layout_roundtrip():
    rng = np.random.default_rng(1)
    poses = _random_poses(2, rng)
    grips = np.array([0.2, 0.9])
    data = {"action/ee_pose": np.concatenate([poses[0], poses[1], grips])[None]}
    spec = FeatureSpec((), 240, 320, "ee", False)
    a = spec.recorded_action(data)[0]
    assert a.shape == (len(EE_NAMES),)
    arm, g = spec.split_action(a)
    assert arm.shape == (2, 7) and np.allclose(g, grips)
    for got, want in zip(arm, poses):
        assert np.allclose(got[:3], want[:3]) and _same_rotation(got[3:], want[3:])


def test_ee_features_continuous_when_the_quaternion_sign_flips():
    """A down-pointing gripper (w ~ 0) whose quaternion sign flips between steps gives smooth features."""
    yaw = np.linspace(-1.0, 1.0, 50)
    q = np.stack([np.zeros_like(yaw), np.cos(yaw / 2), np.sin(yaw / 2), np.zeros_like(yaw)], 1)  # 180 deg about x, then yaw
    q[::2] *= -1  # sign flips every step, as in the recordings
    ee = pose_to_ee(np.concatenate([np.zeros((50, 3)), q], 1))
    assert np.abs(np.diff(ee, axis=0)).max() < 0.1


def test_ee_wrench_in_the_state():
    """The wrench goes in log-compressed, sign kept, after the joint velocities; zero stays zero."""
    from questcollect.learn.features import WRENCH_NAMES
    spec = FeatureSpec((), 240, 320, "joint", False, joint_vel=True, ee_wrench=True)
    w = np.zeros((2, 6)); w[0, 2], w[1, 0] = -0.1, 100.0
    obs = {"joint_pos": np.zeros(14), "gripper_width": np.zeros(2), "joint_vel": np.zeros(14), "ee_wrench": w}
    st = spec.state(obs)
    assert len(st) == len(spec.state_names) == 16 + 14 + 12 and spec.state_names[-12:] == WRENCH_NAMES
    tail = st[-12:]
    assert tail[2] == pytest.approx(-np.log1p(0.1)) and tail[6] == pytest.approx(np.log1p(100.0))
    assert np.count_nonzero(tail) == 2
