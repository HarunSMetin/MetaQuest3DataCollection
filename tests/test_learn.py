"""The learning pipeline end to end on tiny data: episodes -> audit/select -> export -> train -> eval.

Needs the `learn` extra (lerobot, torch) and offscreen rendering: MUJOCO_GL=egl uv run --extra learn --group dev
pytest tests/test_learn.py
"""

import json

import numpy as np
import pytest

pytest.importorskip("lerobot")

import torch  # noqa: E402

from questcollect.audit import audit_dir, select  # noqa: E402
from questcollect.config import load_config  # noqa: E402
from questcollect.learn.evaluate import evaluate  # noqa: E402
from questcollect.learn.export import MANIFEST, export  # noqa: E402
from questcollect.learn.features import FeatureSpec  # noqa: E402
from questcollect.learn.policy import spec_of  # noqa: E402
from questcollect.learn.train import checkpoint_dir, train  # noqa: E402
from questcollect.recorder import episode_meta, make_step, read_episode, teleop_fields, write_episode  # noqa: E402
from questcollect.sim import BimanualEnv  # noqa: E402

TINY_ACT = {"policy.type": "act", "policy.device": "cuda" if torch.cuda.is_available() else "cpu",
            "policy.dim_model": 32, "policy.n_heads": 2, "policy.dim_feedforward": 64, "policy.n_encoder_layers": 1,
            "policy.n_decoder_layers": 1, "policy.latent_dim": 8, "policy.chunk_size": 10,
            "policy.n_action_steps": 10, "policy.pretrained_backbone_weights": None, "policy.push_to_hub": False,
            "batch_size": 2, "num_workers": 0, "steps": 2, "save_freq": 2, "log_freq": 1}


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config().model_copy(deep=True)
    cfg.learn.datasets_dir, cfg.learn.runs_dir = str(tmp_path / "datasets"), str(tmp_path / "runs")
    cfg.learn.cameras, cfg.learn.image_size = ["cam_high", "cam_left_wrist"], [48, 64]
    cfg.learn.presets["tiny"] = TINY_ACT
    cfg.curate.trim_idle = False
    return cfg


def _record(cfg, folder, episodes=2, n=60):
    env = BimanualEnv(cfg)
    for ep in range(episodes):
        env.reset(seed=ep)
        steps = []
        for _ in range(n):
            obs = env.observe()
            q = obs["joint_pos"] + 0.002
            steps.append(make_step(env, obs, env.get_state(), q, np.ones(2), [obs["ee_pose_left"], obs["ee_pose_right"]],
                                   False, 0, env.contact_flags(), teleop_fields(None)))
            env.apply_action(q, np.ones(2))
            env.step()
        write_episode(folder / f"episode_{ep:06d}.hdf5", steps, episode_meta(cfg, env, n, True, "controllers", "t"))


def test_export_train_eval(cfg, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw = tmp_path / "raw"
    _record(cfg, raw)
    segments, rejected = select(raw, audit_dir(raw), cfg.curate)
    assert len(segments) == 2 and not rejected
    spec = FeatureSpec.from_config(cfg)
    out = cfg.datasets_dir / "tiny"
    assert export(segments, out, spec, "local/tiny") == 2
    with pytest.raises(FileExistsError):
        export(segments, out, spec, "local/tiny")

    ds = LeRobotDataset("local/tiny", root=out)
    assert ds.num_episodes == 2 and ds.num_frames == 120
    item = ds[0]
    assert item["observation.state"].shape == (16,) and item["action"].shape == (16,)
    assert item["observation.images.cam_high"].shape == (3, 48, 64)
    assert "observation.images.cam_low" not in item and "observation.environment_state" not in item
    _, data = read_episode(raw / "episode_000000.hdf5")
    assert np.allclose(item["action"].numpy(), data["action/joint_pos"][0], atol=1e-5)
    assert json.loads((out / MANIFEST).read_text())["features"] == spec.to_dict()

    run = train(cfg, "tiny", "tiny", name="tiny_run")
    assert checkpoint_dir(run).exists() and spec_of(checkpoint_dir(run)) == spec  # deployed = trained layout

    summary, results = evaluate(run, cfg, n=1, max_steps=15)
    assert summary["rollouts"] == 1 and (results / "metrics.json").exists()
    meta, data = read_episode(results / "episode_000000.hdf5")
    assert meta["source"] == "policy" and data["control/source"].tolist() == [1] * 15
    assert data["action/joint_pos"].shape == (15, 16)
