"""Bimanual Franka MuJoCo environment (joint-position control)."""

from __future__ import annotations

import mujoco
import numpy as np

from questcollect.config import Config
from questcollect.sim.scene import ARM_JOINTS, ARMS, HOME_Q, build_spec, set_home
from questcollect.sim.tasks import make_task

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
GRIPPER_CTRL_MAX = 255.0


class BimanualEnv:
    def __init__(self, cfg: Config, task_name: str | None = None):
        self.cfg = cfg
        self.task = make_task(task_name or cfg.task)
        self.spec = build_spec(cfg.sim, self.task)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.xml = self.spec.to_xml()
        self.control_dt = 1.0 / cfg.sim.control_hz
        self.n_substeps = max(1, round(self.control_dt / self.model.opt.timestep))

        m = self.model
        self.arm_qadr = np.array([m.joint(f"{a}_{j}").qposadr[0] for a in ARMS for j in ARM_JOINTS])
        self.arm_vadr = np.array([m.joint(f"{a}_{j}").dofadr[0] for a in ARMS for j in ARM_JOINTS])
        self.finger_qadr = np.array(
            [[m.joint(f"{a}_finger_joint{i}").qposadr[0] for i in (1, 2)] for a in ARMS]
        )
        self.arm_act = np.array([m.actuator(f"{a}_actuator{i}").id for a in ARMS for i in range(1, 8)])
        self.grip_act = np.array([m.actuator(f"{a}_actuator8").id for a in ARMS])
        self.tcp_sites = [m.site(f"{a}_tcp").id for a in ARMS]
        self.state_size = mujoco.mj_stateSize(m, STATE_SPEC)
        self.rng = np.random.default_rng()
        self.seed: int | None = None

    # -- lifecycle -------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict:
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2**31))
        self.rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        set_home(self.model, self.data)
        self.task.reset(self.model, self.data, self.rng)
        self.apply_action(np.tile(HOME_Q, 2), np.ones(2))
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def apply_action(self, joint_targets: np.ndarray, grippers: np.ndarray) -> None:
        """joint_targets: (14,) radians [left 7, right 7]; grippers: (2,) in [0=closed, 1=open]."""
        self.data.ctrl[self.arm_act] = joint_targets
        self.data.ctrl[self.grip_act] = np.clip(grippers, 0.0, 1.0) * GRIPPER_CTRL_MAX

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # -- observation -------------------------------------------------------------
    def observe(self) -> dict:
        d = self.data
        ee = []
        for sid in self.tcp_sites:
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, d.site_xmat[sid])
            ee.append(np.concatenate([d.site_xpos[sid], q]))
        return {
            "joint_pos": d.qpos[self.arm_qadr].copy(),
            "joint_vel": d.qvel[self.arm_vadr].copy(),
            "gripper_width": d.qpos[self.finger_qadr].sum(axis=1),
            "ee_pose_left": ee[0],
            "ee_pose_right": ee[1],
            "objects": self.task.object_poses(d),
        }

    def success(self) -> bool:
        return self.task.success(self.model, self.data)

    # -- state (exact replay / re-rendering) -------------------------------------------
    def get_state(self) -> np.ndarray:
        s = np.empty(self.state_size)
        mujoco.mj_getState(self.model, self.data, s, STATE_SPEC)
        return s

    def set_state(self, state: np.ndarray) -> None:
        mujoco.mj_setState(self.model, self.data, np.asarray(state, dtype=np.float64), STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)
