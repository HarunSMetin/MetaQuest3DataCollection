"""Raw HDF5 episodes -> LeRobotDataset.

Camera frames are re-rendered offline from the recorded sim states, so any resolution works
and recording stays light. All lerobot API usage lives in this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from questcollect.config import load_config
from questcollect.record.hdf5_io import read_episode, read_meta
from questcollect.record.render import CameraRenderer
from questcollect.record.schema import ACTION_NAMES
from questcollect.sim.env import BimanualEnv

log = logging.getLogger(__name__)

MAX_GRIPPER_WIDTH = 0.08
EE_NAMES = (
    [f"left_ee_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"right_ee_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + ["left_gripper", "right_gripper"]
)


def _select(raw: Path, task: str | None, include_failures: bool) -> list[Path]:
    files = sorted(Path(raw).glob("**/episode_*.hdf5"))
    out = []
    for f in files:
        m = read_meta(f)
        if task and m.get("task") != task:
            continue
        if not include_failures and not m.get("success"):
            continue
        out.append(f)
    return out


def _state_and_action(data: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    grip = np.clip(data["obs/gripper_width"] / MAX_GRIPPER_WIDTH, 0.0, 1.0)
    if mode == "joint":
        q = data["obs/joint_pos"]
        state = np.concatenate([q[:, :7], grip[:, :1], q[:, 7:], grip[:, 1:]], axis=1)
        return state, data["action/joint_pos"]
    state = np.concatenate([data["obs/ee_pose_left"], data["obs/ee_pose_right"], grip], axis=1)
    return state, data["action/ee_pose"]


def export(raw: Path, out: Path, repo_id: str, task: str | None = None, action: str = "joint",
           height: int = 480, width: int = 640, include_failures: bool = False) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = _select(raw, task, include_failures)
    if not files:
        log.warning("no episodes matched in %s", raw)
        return 0
    metas = [read_meta(f) for f in files]
    tasks = {m["task"] for m in metas}
    fps = int(metas[0]["fps"])
    cameras = list(metas[0]["camera_names"])
    # Object poses differ in size between tasks; include them only for single-task datasets.
    env_dim = None
    if len(tasks) == 1:
        _, d0 = read_episode(files[0], keys=["obs/objects"])
        env_dim = int(np.prod(d0["obs/objects"].shape[1:]))

    names = {"joint": ACTION_NAMES, "ee": EE_NAMES}.get(action)
    if names is None:
        raise ValueError(f"unknown action mode {action!r} (joint | ee)")
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    if env_dim:
        features["observation.environment_state"] = {"dtype": "float32", "shape": (env_dim,), "names": None}
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}

    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=out,
                               robot_type="bimanual_franka_panda", use_videos=True, image_writer_threads=4)
    cfg = load_config()
    envs: dict[str, tuple[BimanualEnv, CameraRenderer]] = {}
    for f, meta in zip(files, metas):
        _, data = read_episode(f)
        if meta["task"] not in envs:
            env = BimanualEnv(cfg, meta["task"])
            envs[meta["task"]] = (env, CameraRenderer(env.model, height, width, cameras))
        env, renderer = envs[meta["task"]]
        states = data["sim/state"]
        if states.shape[1] != env.state_size:
            raise RuntimeError(f"{f}: recorded state size {states.shape[1]} != scene {env.state_size}; "
                               "the scene changed since recording")
        state, act = _state_and_action(data, action)
        objects = data["obs/objects"].reshape(len(states), -1)
        for t in range(len(states)):
            env.set_state(states[t])
            frame = {
                "observation.state": state[t].astype(np.float32),
                "action": act[t].astype(np.float32),
                "task": meta["instruction"],
            }
            if env_dim:
                frame["observation.environment_state"] = objects[t].astype(np.float32)
            for cam, img in renderer.render(env.data).items():
                frame[f"observation.images.{cam}"] = img
            ds.add_frame(frame)
        ds.save_episode()
        log.info("exported %s (%d frames)", f.name, len(states))
    ds.finalize()
    for _, r in envs.values():
        r.close()
    return len(files)
