"""What a policy sees and outputs: the one definition shared by the exporter (export.py) and the policy runner
(policy.py), so a policy is deployed with exactly the observations it was trained on.

The spec is saved in every dataset manifest; the runner reads it back from there, not from today's config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from questcollect.config import Config
from questcollect.sim import FINGER_TRAVEL

GRIPPER_MAX_WIDTH = 2 * FINGER_TRAVEL  # m, fully open Franka Hand
JOINT_NAMES = (
    [f"left_joint{i}" for i in range(1, 8)] + ["left_gripper"]
    + [f"right_joint{i}" for i in range(1, 8)] + ["right_gripper"]
)
EE_NAMES = (
    [f"left_ee_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + [f"right_ee_{k}" for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    + ["left_gripper", "right_gripper"]
)
ROBOT_TYPE = "bimanual_franka_panda"


@dataclass(frozen=True)
class FeatureSpec:
    cameras: tuple[str, ...]
    height: int
    width: int
    action: str            # joint: [7 joints, gripper] per arm | ee: [xyz qwxyz] per arm + 2 grippers
    env_state: bool        # brick poses as observation.environment_state (privileged, sim only)

    @classmethod
    def from_config(cls, cfg: Config) -> FeatureSpec:
        h, w = cfg.learn.image_size
        return cls(tuple(cfg.learn.cameras), int(h), int(w), cfg.learn.action, cfg.learn.env_state)

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSpec:
        return cls(tuple(d["cameras"]), int(d["height"]), int(d["width"]), d["action"], bool(d["env_state"]))

    def to_dict(self) -> dict:
        return {**asdict(self), "cameras": list(self.cameras)}

    @property
    def names(self) -> list[str]:
        return JOINT_NAMES if self.action == "joint" else EE_NAMES

    def lerobot_features(self) -> dict:
        n = len(self.names)
        f = {
            "observation.state": {"dtype": "float32", "shape": (n,), "names": self.names},
            "action": {"dtype": "float32", "shape": (n,), "names": self.names},
        }
        if self.env_state:
            f["observation.environment_state"] = {"dtype": "float32", "shape": (28,), "names": None}
        for cam in self.cameras:
            f[f"observation.images.{cam}"] = {"dtype": "video", "shape": (self.height, self.width, 3),
                                              "names": ["height", "width", "channels"]}
        return f

    def state(self, obs: dict) -> np.ndarray:
        """observation.state from BimanualEnv.observe() (one step) or the episode's obs/* arrays (leading T)."""
        grip = np.clip(np.asarray(obs["gripper_width"]) / GRIPPER_MAX_WIDTH, 0.0, 1.0)
        if self.action == "joint":
            q = np.asarray(obs["joint_pos"])
            return np.concatenate([q[..., :7], grip[..., :1], q[..., 7:], grip[..., 1:]], axis=-1)
        return np.concatenate([obs["ee_pose_left"], obs["ee_pose_right"], grip], axis=-1)

    def environment_state(self, obs: dict) -> np.ndarray:
        objects = np.asarray(obs["objects"])
        return objects.reshape(*objects.shape[:-2], -1)

    def recorded_action(self, data: dict) -> np.ndarray:
        """(T, n) action column of an episode (read_episode data)."""
        return data["action/joint_pos"] if self.action == "joint" else data["action/ee_pose"]

    def split_action(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Policy output -> (arm part, grippers): (14,) joint targets or (2, 7) TCP poses, and (2,) grippers."""
        a = np.asarray(a, float)
        if self.action == "joint":
            return np.concatenate([a[:7], a[8:15]]), np.array([a[7], a[15]])
        return a[:14].reshape(2, 7), a[14:16]


def episode_obs(data: dict) -> dict:
    """read_episode data -> the obs dict layout of BimanualEnv.observe() (with a leading time axis)."""
    return {k[4:]: v for k, v in data.items() if k.startswith("obs/")}
