"""What a policy sees and outputs: the one definition shared by the exporter (export.py) and the policy runner
(policy.py), so a policy is deployed with exactly the observations it was trained on.

The spec is saved in every dataset manifest; the runner reads it back from there, not from today's config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from questcollect.config import Config
from questcollect.sim import FINGER_TRAVEL

GRIPPER_MAX_WIDTH = 2 * FINGER_TRAVEL  # m, fully open Franka Hand
JOINT_NAMES = (
    [f"left_joint{i}" for i in range(1, 8)] + ["left_gripper"]
    + [f"right_joint{i}" for i in range(1, 8)] + ["right_gripper"]
)
# TCP pose per arm: position + the first two columns of the rotation matrix ("6D rotation", Zhou et al. 2019).
# Quaternions don't work here: the gripper points down, so w ~ 0 and q / -q flip from step to step, and a
# network would average the two. The 6D form is continuous and the same for q and -q.
EE_KEYS = ("x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32")
EE_NAMES = [f"{a}_ee_{k}" for a in ("left", "right") for k in EE_KEYS] + ["left_gripper", "right_gripper"]
VEL_NAMES = [f"{a}_joint{i}_vel" for a in ("left", "right") for i in range(1, 8)]
WRENCH_NAMES = [f"{a}_ee_{k}" for a in ("left", "right") for k in ("fx", "fy", "fz", "tx", "ty", "tz")]
# The hand wrench spans 0.1 N (a carried brick) to ~250 N (pressing one on), so it goes in log-compressed:
# sign(f) log(1 + |f| / scale). Min-max normalized as is, everything below ~10 N would look like zero.
WRENCH_SCALE = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # N, Nm
ROBOT_TYPE = "bimanual_franka_panda"


@dataclass(frozen=True)
class FeatureSpec:
    cameras: tuple[str, ...]
    height: int
    width: int
    action: str            # joint: [7 joints, gripper] per arm | ee: [xyz, 6D rotation] per arm + 2 grippers
    env_state: bool        # brick poses as observation.environment_state (privileged, sim only)
    joint_vel: bool = False  # observation.state also has the 14 joint velocities (rad/s)
    ee_wrench: bool = False  # ... and the external wrench on each hand (sim.BimanualEnv.ee_wrench, log-compressed)
    # Action labels: "recorded" = action/* as saved (the operator's target before the DART drift); "executed" = the
    # targets actually sent (drift included), taken from sim/state. The executed ones are what seated the bricks.
    action_label: str = "recorded"

    @classmethod
    def from_config(cls, cfg: Config) -> FeatureSpec:
        h, w = cfg.learn.image_size
        return cls(tuple(cfg.learn.cameras), int(h), int(w), cfg.learn.action, cfg.learn.env_state,
                   cfg.learn.joint_vel, cfg.learn.ee_wrench, cfg.learn.action_label)

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSpec:
        return cls(tuple(d["cameras"]), int(d["height"]), int(d["width"]), d["action"], bool(d["env_state"]),
                   bool(d.get("joint_vel", False)), bool(d.get("ee_wrench", False)),  # older datasets: without
                   d.get("action_label", "recorded"))

    def to_dict(self) -> dict:
        return {**asdict(self), "cameras": list(self.cameras)}

    @property
    def names(self) -> list[str]:
        """Action components (and the first part of the state, which mirrors the action layout)."""
        return JOINT_NAMES if self.action == "joint" else EE_NAMES

    @property
    def state_names(self) -> list[str]:
        return self.names + (VEL_NAMES if self.joint_vel else []) + (WRENCH_NAMES if self.ee_wrench else [])

    def lerobot_features(self) -> dict:
        f = {
            "observation.state": {"dtype": "float32", "shape": (len(self.state_names),), "names": self.state_names},
            "action": {"dtype": "float32", "shape": (len(self.names),), "names": self.names},
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
            parts = [q[..., :7], grip[..., :1], q[..., 7:], grip[..., 1:]]
        else:
            parts = [pose_to_ee(obs["ee_pose_left"]), pose_to_ee(obs["ee_pose_right"]), grip]
        if self.joint_vel:
            parts.append(np.asarray(obs["joint_vel"]))
        if self.ee_wrench:
            w = np.asarray(obs["ee_wrench"])  # (..., 2, 6)
            w = np.sign(w) * np.log1p(np.abs(w) / WRENCH_SCALE)
            parts.append(w.reshape(*w.shape[:-2], 12))
        return np.concatenate(parts, axis=-1)

    def environment_state(self, obs: dict) -> np.ndarray:
        objects = np.asarray(obs["objects"])
        return objects.reshape(*objects.shape[:-2], -1)

    def recorded_action(self, data: dict) -> np.ndarray:
        """(T, n) action column of an episode (read_episode data)."""
        if self.action == "joint":
            return data["action/joint_pos"]
        a = data["action/ee_pose"]  # [L xyz qwxyz, R xyz qwxyz, grip L, grip R]
        return np.concatenate([pose_to_ee(a[..., :7]), pose_to_ee(a[..., 7:14]), a[..., 14:16]], axis=-1)

    def split_action(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Policy output -> (arm part, grippers): (14,) joint targets or (2, 7) TCP poses [xyz qwxyz], and
        (2,) grippers."""
        a = np.asarray(a, float)
        if self.action == "joint":
            return np.concatenate([a[:7], a[8:15]]), np.array([a[7], a[15]])
        n = len(EE_KEYS)
        return np.stack([ee_to_pose(a[:n]), ee_to_pose(a[n:2 * n])]), a[2 * n:2 * n + 2]


def pose_to_ee(pose: np.ndarray) -> np.ndarray:
    """(..., 7) [xyz qwxyz] -> (..., 9) [xyz, first two rotation-matrix columns]."""
    pose = np.asarray(pose, float)
    w, x, y, z = np.moveaxis(pose[..., 3:7] / np.linalg.norm(pose[..., 3:7], axis=-1, keepdims=True), -1, 0)
    col1 = [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)]
    col2 = [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)]
    return np.concatenate([pose[..., :3], np.stack(col1 + col2, axis=-1)], axis=-1)


def ee_to_pose(ee: np.ndarray) -> np.ndarray:
    """(9,) [xyz, two rotation columns] -> (7,) [xyz qwxyz]; the columns are re-orthonormalized (Gram-Schmidt)."""
    ee = np.asarray(ee, float)
    b1 = ee[3:6] / np.linalg.norm(ee[3:6])
    b2 = ee[6:9] - b1 @ ee[6:9] * b1
    b2 /= np.linalg.norm(b2)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.column_stack([b1, b2, np.cross(b1, b2)]).ravel())
    return np.concatenate([ee[:3], q])


def episode_obs(data: dict) -> dict:
    """read_episode data -> the obs dict layout of BimanualEnv.observe() (with a leading time axis)."""
    return {k[4:]: v for k, v in data.items() if k.startswith("obs/")}
