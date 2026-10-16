"""Task interface: scene objects, randomized reset, success check."""

from __future__ import annotations

import mujoco
import numpy as np

TASKS: dict[str, type["Task"]] = {}


def register(cls):
    TASKS[cls.name] = cls
    return cls


def make_task(name: str) -> "Task":
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]()


class Task:
    name: str = "base"
    instruction: str = ""
    objects: tuple[str, ...] = ()  # free-floating bodies whose poses are recorded

    table_height: float = 0.75

    def build(self, spec: mujoco.MjSpec, layout) -> None:
        self.table_height = layout.table_height

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
        pass

    def success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        return False

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def add_free_body(spec, name, pos, geoms: list[dict]):
        body = spec.worldbody.add_body(name=name, pos=list(pos))
        body.add_freejoint(name=f"{name}_free")
        for g in geoms:
            body.add_geom(**g)
        return body

    @staticmethod
    def set_free_pose(data, name, pos, yaw=0.0):
        q = data.joint(f"{name}_free").qpos
        q[:3] = pos
        q[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        data.joint(f"{name}_free").qvel[:] = 0

    def object_poses(self, data) -> np.ndarray:
        """(k, 7) array of [x y z qw qx qy qz] for each tracked object."""
        if not self.objects:
            return np.zeros((0, 7))
        return np.stack([data.joint(f"{o}_free").qpos.copy() for o in self.objects])
