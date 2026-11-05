"""Safety filter for policy actions: change the proposed joint targets as little as possible so the arms keep
apart from each other and from the table, within joint position and speed limits.

One QP per control step with mink, on the robot-only scene (like BimanualIK): a posture task pulls towards the
policy's joint targets, and the limits keep it safe. In free motion the output equals the input; near an
obstacle the arms slow down and stop at the margin instead of pushing in. Grippers are not filtered.
"""

from __future__ import annotations

import mink
import numpy as np

from questcollect.config import IKConfig, SafetyConfig, SimConfig
from questcollect.ik import collision_geoms
from questcollect.sim import ARM_JOINTS, ARMS, HOME_Q, build_spec

TABLE_BODIES = ("link4", "link5", "link6", "link7", "hand")  # kept safety.table_margin above the table
FINGER_BODIES = ("left_finger", "right_finger")  # kept safety.finger_table_margin above the table (0: may touch)
DETECTION_EXTRA = 0.08  # m beyond the margin at which an approaching obstacle starts to slow the arm
# QP steps per control step. The collision limits are linearized: one 20 ms step let fingertips sink 7.6 mm into
# the table and arms close to 3 mm; 8 substeps keep them at ~0 mm and ~15 mm, for ~1.5 ms per control step.
SUBSTEPS = 8


class SafetyFilter:
    def __init__(self, sim_cfg: SimConfig, ik_cfg: IKConfig, cfg: SafetyConfig, dt: float):
        self.cfg, self.dt, self.max_lead = cfg, dt, ik_cfg.max_lead
        self.model = build_spec(sim_cfg, with_task=False).compile()
        m = self.model
        self.configuration = mink.Configuration(m)
        self._fk = mink.Configuration(m)  # tcp_poses(q) for other joint angles than the filter's own
        self.arm_qadr = np.array([m.joint(f"{a}_{j}").qposadr[0] for a in ARMS for j in ARM_JOINTS])
        self.finger_qadr = np.array([m.joint(f"{a}_finger_joint{i}").qposadr[0] for a in ARMS for i in (1, 2)])
        # The fingers are only there to be in the right place: the QP may not move them to clear an obstacle.
        self._finger_range = m.jnt_range[[m.joint(f"{a}_finger_joint{i}").id for a in ARMS for i in (1, 2)]].T
        self.freeze = [mink.DofFreezingTask(m, [int(m.joint(f"{a}_finger_joint{i}").dofadr[0])
                                                for a in ARMS for i in (1, 2)])]
        self.posture = mink.PostureTask(m, cost=1.0)
        vel = {f"{a}_{j}": ik_cfg.max_joint_vel for a in ARMS for j in ARM_JOINTS}
        table = [m.geom("table_top").id]
        self.limits = [
            mink.ConfigurationLimit(m), mink.VelocityLimit(m, vel),
            mink.CollisionAvoidanceLimit(m, geom_pairs=[(collision_geoms(m, "l"), collision_geoms(m, "r"))],
                                         minimum_distance_from_collisions=cfg.arm_margin,
                                         collision_detection_distance=cfg.arm_margin + DETECTION_EXTRA),
            mink.CollisionAvoidanceLimit(m, geom_pairs=[(collision_geoms(m, "l", TABLE_BODIES)
                                                         + collision_geoms(m, "r", TABLE_BODIES), table)],
                                         minimum_distance_from_collisions=cfg.table_margin,
                                         collision_detection_distance=cfg.table_margin + DETECTION_EXTRA),
            mink.CollisionAvoidanceLimit(m, geom_pairs=[(collision_geoms(m, "l", FINGER_BODIES)
                                                         + collision_geoms(m, "r", FINGER_BODIES), table)],
                                         minimum_distance_from_collisions=cfg.finger_table_margin,
                                         collision_detection_distance=cfg.finger_table_margin + 0.02),
        ]
        self.reset(np.tile(HOME_Q, 2))

    def reset(self, arm_q: np.ndarray) -> None:
        q = self.model.qpos0.copy()
        q[self.arm_qadr] = arm_q
        self.configuration.update(q)

    def filter(self, q_proposed: np.ndarray, q_measured: np.ndarray,
               fingers: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """(14,) proposed joint targets -> (safe targets, largest change in rad). Like BimanualIK.follow, the
        filter's own configuration stays within ik.max_lead of the measured joints, so it can't wind up.
        fingers: the measured finger joints [l1, l2, r1, r2] (m). An open finger sits 4 cm off the hand's centre,
        so with the hand tilted it hangs lower than a closed one; without them the fingers count as closed."""
        q = self.configuration.q.copy()
        if fingers is not None:  # just inside the joint range: frozen at the limit, the QP is infeasible
            lo, hi = self._finger_range
            q[self.finger_qadr] = np.clip(np.ravel(fingers), lo + 1e-6, hi - 1e-6)
        if not self.cfg.enabled:
            q[self.arm_qadr] = q_proposed
            self.configuration.update(q)
            return np.asarray(q_proposed, float).copy(), 0.0
        lead = q[self.arm_qadr] - q_measured
        q[self.arm_qadr] = q_measured + np.clip(lead, -self.max_lead, self.max_lead)
        self.configuration.update(q)
        target = q.copy()
        target[self.arm_qadr] = q_proposed
        self.posture.set_target(target)
        dt = self.dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            v = mink.solve_ik(self.configuration, [self.posture], dt, solver="daqp", damping=1e-6,
                              limits=self.limits, constraints=self.freeze)
            self.configuration.integrate_inplace(v, dt)
        q_safe = self.configuration.q[self.arm_qadr].copy()
        return q_safe, float(np.abs(q_safe - q_proposed).max())

    def tcp_poses(self, arm_q: np.ndarray | None = None) -> list[np.ndarray]:
        """TCP pose [x y z qw qx qy qz] of each arm at the last filtered targets, or at the (14,) joint angles
        given (recorded as action/ee_pose)."""
        conf = self.configuration
        if arm_q is not None:
            q = self.model.qpos0.copy()
            q[self.arm_qadr] = arm_q
            self._fk.update(q)
            conf = self._fk
        out = []
        for a in ARMS:
            T = conf.get_transform_frame_to_world(f"{a}_tcp", "site")
            out.append(np.concatenate([T.translation(), T.rotation().wxyz]))
        return out
