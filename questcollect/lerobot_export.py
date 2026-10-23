"""Raw HDF5 episodes -> LeRobotDataset.

Camera frames are re-rendered offline from the recorded sim states, so any resolution works
and recording stays light. All lerobot API usage lives in this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from questcollect.config import load_config
from questcollect.recorder import episode_files, read_episode, read_meta
from questcollect.sim import AppearanceRandomizer, BimanualEnv, CameraRenderer

log = logging.getLogger(__name__)

MAX_GRIPPER_WIDTH = 0.08
JOINT_NAMES = (
    [f"left_joint{i}" for i in range(1, 8)] + ["left_gripper"]
    + [f"right_joint{i}" for i in range(1, 8)] + ["right_gripper"]
)
EE_NAMES = (
    [f"left_ee_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"right_ee_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + ["left_gripper", "right_gripper"]
)


def _state_and_action(data: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    grip = np.clip(data["obs/gripper_width"] / MAX_GRIPPER_WIDTH, 0.0, 1.0)
    if mode == "joint":
        q = data["obs/joint_pos"]
        state = np.concatenate([q[:, :7], grip[:, :1], q[:, 7:], grip[:, 1:]], axis=1)
        return state, data["action/joint_pos"]
    state = np.concatenate([data["obs/ee_pose_left"], data["obs/ee_pose_right"], grip], axis=1)
    return state, data["action/ee_pose"]


def export(raw: Path, out: Path, repo_id: str, action: str = "joint",
           height: int = 480, width: int = 640, include_failures: bool = False,
           randomize_visuals: bool = False) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = {"joint": JOINT_NAMES, "ee": EE_NAMES}.get(action)
    if names is None:
        raise ValueError(f"unknown action mode {action!r} (joint | ee)")
    files = [f for f in episode_files(raw) if include_failures or read_meta(f).get("success")]
    if not files:
        log.warning("no episodes matched in %s", raw)
        return 0
    meta0 = read_meta(files[0])
    cameras = list(meta0["camera_names"])
    env = BimanualEnv(load_config())
    renderer = CameraRenderer(env.model, height, width, cameras)
    looks = AppearanceRandomizer(env.model) if randomize_visuals else None

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.environment_state": {"dtype": "float32", "shape": (7 * len(meta0["objects"]),), "names": None},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id=repo_id, fps=int(meta0["fps"]), features=features, root=out,
                               robot_type="bimanual_franka_panda", use_videos=True, image_writer_threads=4)

    for k, f in enumerate(files):
        meta, data = read_episode(f)
        if looks is not None:
            looks.apply(np.random.default_rng([k, int(meta.get("seed", 0))]))
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
                "observation.environment_state": objects[t].astype(np.float32),
                "task": meta["instruction"],
            }
            for cam, img in renderer.render(env.data).items():
                frame[f"observation.images.{cam}"] = img
            ds.add_frame(frame)
        ds.save_episode()
        log.info("exported %s (%d frames)", f.name, len(states))
    ds.finalize()
    renderer.close()
    return len(files)
