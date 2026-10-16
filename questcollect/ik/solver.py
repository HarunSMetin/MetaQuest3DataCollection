"""Differential IK for both arms with mink (one QP, limits + arm-arm collision avoidance)."""

from __future__ import annotations

import mink
import mujoco
import numpy as np

from questcollect.config import IKConfig, SimConfig
from questcollect.sim.scene import ARM_JOINTS, ARMS, HOME_Q, build_spec

COLLISION_BODIES = ("link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")


def _collision_geoms(model: mujoco.MjModel, arm: str) -> list[int]:
    ids = []
    for b in COLLISION_BODIES:
        bid = model.body(f"{arm}_{b}").id
        ids += [g for g in mink.get_body_geom_ids(model, bid) if model.geom_contype[g] or model.geom_conaffinity[g]]
    return ids


class BimanualIK:
    """Keeps its own kinematic configuration (robots only) that tracks EE targets smoothly."""

    def __init__(self, sim_cfg: SimConfig, cfg: IKConfig, dt: float):
        self.cfg = cfg
        self.dt = dt
        self.model = build_spec(sim_cfg).compile()
        self.configuration = mink.Configuration(self.model)
        m = self.model
        self.arm_qadr = np.array([m.joint(f"{a}_{j}").qposadr[0] for a in ARMS for j in ARM_JOINTS])

        self.frame_tasks = {
            a: mink.FrameTask(f"{a}_tcp", "site", position_cost=cfg.position_cost,
                              orientation_cost=cfg.orientation_cost, lm_damping=1.0)
            for a in ARMS
        }
        self.posture = mink.PostureTask(m, cost=cfg.posture_cost)
        self.tasks = [*self.frame_tasks.values(), self.posture]
        vel = {f"{a}_{j}": cfg.max_joint_vel for a in ARMS for j in ARM_JOINTS}
        self.limits = [
            mink.ConfigurationLimit(m),
            mink.VelocityLimit(m, vel),
            mink.CollisionAvoidanceLimit(
                m, geom_pairs=[(_collision_geoms(m, "l"), _collision_geoms(m, "r"))],
                minimum_distance_from_collisions=cfg.collision_margin,
                collision_detection_distance=cfg.collision_margin + 0.08,
            ),
        ]
        self.reset(np.tile(HOME_Q, 2))

    def reset(self, arm_q: np.ndarray) -> None:
        q = self.model.qpos0.copy()
        q[self.arm_qadr] = arm_q
        self.configuration.update(q)
        self.posture.set_target(q)
        for a in ARMS:
            self.frame_tasks[a].set_target(self.configuration.get_transform_frame_to_world(f"{a}_tcp", "site"))

    def tcp_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        T = self.configuration.get_transform_frame_to_world(f"{arm}_tcp", "site")
        return T.translation().copy(), T.rotation().wxyz.copy()

    def set_target(self, arm: str, pos: np.ndarray, wxyz: np.ndarray) -> None:
        T = mink.SE3.from_rotation_and_translation(mink.SO3(np.asarray(wxyz, float)), np.asarray(pos, float))
        self.frame_tasks[arm].set_target(T)

    def solve(self) -> np.ndarray:
        """Advance the IK configuration one control step; returns the (14,) joint targets."""
        sub_dt = self.dt / self.cfg.iterations
        for _ in range(self.cfg.iterations):
            v = mink.solve_ik(self.configuration, self.tasks, sub_dt, solver="daqp",
                              damping=1e-3, limits=self.limits)
            self.configuration.integrate_inplace(v, sub_dt)
        return self.configuration.q[self.arm_qadr].copy()

    def errors(self) -> dict[str, tuple[float, float]]:
        """Per-arm (position error m, orientation error rad) to the current target."""
        out = {}
        for a, t in self.frame_tasks.items():
            e = t.compute_error(self.configuration)
            out[a] = (float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])))
        return out
