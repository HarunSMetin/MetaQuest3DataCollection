import numpy as np
import pytest

from questcollect.audit import audit_dir, audit_episode, select, set_curation, true_runs
from questcollect.config import CurateConfig, load_config
from questcollect.recorder import SOURCE_POLICY, episode_meta, make_step, teleop_fields, write_episode
from questcollect.sim import BimanualEnv


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def env(cfg):
    e = BimanualEnv(cfg)
    e.reset(seed=3)
    return e


def _episode(env, cfg, path, n=100, moving=(20, 80), link_table=(40, 45), policy=None, with_contacts=True):
    """n steps of a still scene, with joint motion, arm-on-table contact and policy steps where asked."""
    steps = []
    for i in range(n):
        obs = env.observe()
        obs["joint_vel"] = obs["joint_vel"] + (0.5 if moving[0] <= i < moving[1] else 0.0)
        flags = np.array([link_table[0] <= i < link_table[1], False, False])
        source = SOURCE_POLICY if policy and policy[0] <= i < policy[1] else 0
        step = make_step(env, obs, env.get_state(), obs["joint_pos"], np.ones(2),
                         [obs["ee_pose_left"], obs["ee_pose_right"]], False, 0, flags, teleop_fields(None), source)
        if not with_contacts:
            del step["contacts"]
        steps.append(step)
    write_episode(path, steps, episode_meta(cfg, env, n, True, "controllers", "test"))
    return path


def test_true_runs():
    assert true_runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert true_runs(np.zeros(3, bool)) == []


def test_audit_measures_contacts_idle_and_trim(env, cfg, tmp_path):
    a = audit_episode(_episode(env, cfg, tmp_path / "episode_000000.hdf5"))
    assert a.active == (20, 80)
    assert a.idle_frac == pytest.approx(0.4)
    assert a.contact_frac["link_table"] == pytest.approx(0.05) and a.contact_events["link_table"] == 1
    assert a.level_times == [None, None, None] and a.operator_runs == [(0, 100)]


def test_select_applies_rules_and_curation(env, cfg, tmp_path):
    _episode(env, cfg, tmp_path / "episode_000000.hdf5")
    audits = audit_dir(tmp_path)
    assert (tmp_path / "audit.json").exists() and audit_dir(tmp_path) == audits  # second call: from the cache
    segs, rejected = select(tmp_path, audits, CurateConfig())
    assert not segs and "arm link on the table" in rejected["episode_000000.hdf5"]
    lenient = CurateConfig(max_link_table_frac=0.1)
    segs, _ = select(tmp_path, audits, lenient)
    assert [(s.start, s.stop) for s in segs] == [(20, 80)]
    set_curation(tmp_path, "episode_000000.hdf5", True, "blurry")
    _, rejected = select(tmp_path, audits, lenient)
    assert rejected["episode_000000.hdf5"] == "excluded by hand: blurry"
    set_curation(tmp_path, "episode_000000.hdf5", False)
    assert select(tmp_path, audits, lenient)[0]


def test_dagger_keeps_only_operator_steps(env, cfg, tmp_path):
    _episode(env, cfg, tmp_path / "episode_000000.hdf5", n=300, moving=(20, 280), link_table=(0, 0),
             policy=(100, 150))
    segs, _ = select(tmp_path, audit_dir(tmp_path), CurateConfig())
    assert [(s.start, s.stop) for s in segs] == [(20, 100), (150, 280)]
    segs, _ = select(tmp_path, audit_dir(tmp_path), CurateConfig(dagger_frames="all"))
    assert [(s.start, s.stop) for s in segs] == [(20, 280)]


def test_old_episodes_are_replayed_for_contacts(env, cfg, tmp_path):
    """Episodes from before schema v5 have no `contacts`: the audit replays sim/state instead."""
    a = audit_episode(_episode(env, cfg, tmp_path / "episode_000000.hdf5", with_contacts=False))
    assert a.contact_frac == {"link_table": 0.0, "arm_arm": 0.0, "tip_table": 0.0}  # arms at home, clear of all
