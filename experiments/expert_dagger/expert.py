"""A scripted expert for the two-brick task (red base into the build area, blue clicked on top of it).

It "cheats": it reads the true brick poses from the simulation, so it needs no learning. Each arm runs its own
little program (a Python generator) that decides from the *current* state, every tick:

  left  (red):  pick red -> carry it into the build area -> set it down -> release -> back out of the zone
  right (blue): pick blue (at the same time) -> carry it to a waiting point beside the build area -> wait until
                the left arm has left the zone -> hover over the tower -> descend while servoing the brick's true
                position onto its goal -> press until it clicks -> release -> back out

Grasps are re-tried if the brick is not held, the placement uses the brick's true pose in the hand (so an
off-centre grasp is compensated), and the tower goal is computed from where the red brick actually lies now.

The arms are driven exactly like the operator's: TCP targets -> BimanualIK (follow the measured joints) ->
SafetyFilter -> apply. Steps are recorded with recorder.make_step (control/source = operator), so the files are
ordinary episodes for audit / export / check_pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from questcollect.config import Config
from questcollect.frames import mat_to_quat_wxyz, quat_wxyz_to_mat
from questcollect.ik import BimanualIK
from questcollect.recorder import SOURCE_OPERATOR, make_step, teleop_fields
from questcollect.safety import SafetyFilter
from questcollect.sim import (ARMS, BRICKS, BUILD_AREA_CENTER, BimanualEnv, _quat_yaw, _rot_z, tower_poses)

RED, BLUE = 0, 1
DOWN = np.diag([1.0, -1.0, -1.0])  # TCP pointing at the table
ZONE_R = 0.11            # m: xy radius around the build area that only one arm may be in
WAIT_Y = 0.21            # m: the right arm waits this far to the side of the build area (0.15 let the arms touch)
TRANSIT_Z = 0.10         # m above the table for carrying
HOVER_Z = 0.025          # m: brick bottom this far above its goal before the slow final descent
SLOW = 0.03              # m/s final descent
PRESS = 0.006            # m: push the TCP target this far below the seated height
CLICK_TIMEOUT_S = 1.5
GRASP_SETTLE_S = 0.6     # fingers need ~0.5 s from open to a 32 mm brick at 50 mm/s per finger
MAX_TRIES = 3
STUD_CLEAR = 0.007     # m: brick bottom this far above its goal (just over the 4.5 mm studs) for the last alignment
ALIGN_TOL = 0.0004     # m: brick xy within this of its goal before the final descent


def pose(p, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def body_T(env: BimanualEnv, name: str) -> np.ndarray:
    b = env.data.body(name)
    return pose(b.xpos, b.xmat.reshape(3, 3))


def tcp_T(env: BimanualEnv, arm: str) -> np.ndarray:
    sid = env.tcp_sites[ARMS.index(arm)]
    return pose(env.data.site_xpos[sid], env.data.site_xmat[sid].reshape(3, 3))


def grasp_R(yaw: float) -> np.ndarray:
    """TCP down, fingers closing across the brick's y axis, with the shortest wrist turn (yaw mod pi)."""
    return _rot_z((yaw + np.pi / 2) % np.pi - np.pi / 2) @ DOWN


def rot_angle(Ra, Rb) -> float:
    return float(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def slerp_R(Ra, Rb, s) -> np.ndarray:
    """Rotation s of the way from Ra to Rb (via the axis-angle of Ra^T Rb)."""
    Rd = Ra.T @ Rb
    ang = rot_angle(np.eye(3), Rd)
    if ang < 1e-9:
        return Rb.copy()
    w = np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]]) / (2 * np.sin(ang))
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    a = s * ang
    return Ra @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K)


@dataclass
class Params:
    """Per-episode variation, so the demos are not all identical."""
    speed: float = 0.22           # m/s average transit speed
    base_xy: tuple = (0.0, 0.0)   # offset of the red brick's goal from the build-area centre
    base_yaw: float = np.pi / 2  # the red base turned across (long side along y), like the operator's demos
    wait_xy: tuple = (0.0, 0.0)   # offset of the right arm's waiting point
    start_delay: tuple = (0.0, 0.0)  # s, per arm

    @staticmethod
    def sample(rng: np.random.Generator) -> "Params":
        return Params(speed=rng.uniform(0.18, 0.28), base_xy=tuple(rng.uniform(-0.012, 0.012, 2)),
                      base_yaw=np.pi / 2 + rng.uniform(-np.deg2rad(10), np.deg2rad(10)),
                      wait_xy=tuple(rng.uniform(-0.02, 0.02, 2)),
                      start_delay=tuple(rng.uniform(0.0, 0.4, 2)))


@dataclass
class ArmState:
    T: np.ndarray                 # commanded TCP pose
    grip: float = 1.0
    phase: str = "start"
    in_zone: bool = False
    done: bool = False
    log: list = field(default_factory=list)


class Expert:
    def __init__(self, env: BimanualEnv, cfg: Config, params: Params):
        self.env, self.cfg, self.p = env, cfg, params
        self.dt = env.control_dt
        self.th = env.table_height
        self.ik = BimanualIK(cfg.sim, cfg.ik, self.dt)
        self.safety = SafetyFilter(cfg.sim, cfg.ik, cfg.safety, self.dt)
        q0 = env.data.qpos[env.arm_qadr].copy()
        self.ik.reset(q0)
        self.safety.reset(q0)
        self.arm = {}
        for a in ARMS:
            p, q = self.ik.tcp_pose(a)
            self.arm[a] = ArmState(T=pose(p, quat_wxyz_to_mat(q)))
        self.home = {a: self.arm[a].T.copy() for a in ARMS}
        self.zone = np.array(BUILD_AREA_CENTER)
        self.t = 0.0
        self.programs = {"l": self._left(), "r": self._right()}
        self.failed: str | None = None

    # -- primitives (generators: one yield per control tick) ---------------------------------------------
    def wait(self, a: str, seconds: float):
        for _ in range(max(1, round(seconds / self.dt))):
            yield

    def move(self, a: str, goal, speed: float | None = None, min_time: float = 0.3):
        """Min-jerk from the commanded pose to goal (a 4x4 or a function returning one, re-read every tick)."""
        st = self.arm[a]
        T0 = st.T.copy()
        G = goal() if callable(goal) else goal
        dist = np.linalg.norm(G[:3, 3] - T0[:3, 3])
        ang = rot_angle(T0[:3, :3], G[:3, :3])
        dur = max(min_time, 1.875 * dist / (speed or self.p.speed), ang / 1.2)  # 1.875: min-jerk peak/avg
        n = max(1, round(dur / self.dt))
        for k in range(1, n + 1):
            G = goal() if callable(goal) else goal
            s = k / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5
            st.T = pose(T0[:3, 3] + s * (G[:3, 3] - T0[:3, 3]), slerp_R(T0[:3, :3], G[:3, :3], s))
            yield

    def above(self, T: np.ndarray, z: float) -> np.ndarray:
        U = T.copy()
        U[2, 3] = z
        return U

    def zone_dist(self, a: str) -> float:
        return float(np.linalg.norm(tcp_T(self.env, a)[:2, 3] - self.zone))

    def pick(self, a: str, b: int):
        """Grasp brick b (re-tried); afterwards self.arm[a] holds and returns the in-hand transform."""
        st, brick, env = self.arm[a], BRICKS[b], self.env
        for attempt in range(MAX_TRIES):
            st.phase = f"pick {brick.name} #{attempt + 1}"

            def grasp_pose():
                B = body_T(env, brick.name)
                return pose([B[0, 3], B[1, 3], self.th + brick.h / 2 + 0.002], grasp_R(_quat_yaw(
                    env.data.body(brick.name).xquat)))
            yield from self.move(a, self.above(grasp_pose(), self.th + TRANSIT_Z))
            yield from self.move(a, lambda: self.above(grasp_pose(), self.th + brick.h + 0.03))
            yield from self.move(a, grasp_pose, speed=0.08)
            yield from self.wait(a, 0.15)
            st.grip = 0.0
            yield from self.wait(a, GRASP_SETTLE_S)
            if env.held()[a] == b:
                H = np.linalg.inv(tcp_T(env, a)) @ body_T(env, brick.name)
                yield from self.move(a, self.above(st.T, self.th + TRANSIT_Z), speed=0.15)
                if env.held()[a] == b:
                    return H
            st.grip = 1.0
            yield from self.wait(a, 0.5)
            yield from self.move(a, self.above(st.T, self.th + TRANSIT_Z))
        self.failed = f"{a}: could not grasp {brick.name}"
        return None

    def place(self, a: str, b: int, goal_fn, press: bool):
        """Put held brick b at goal_fn() (4x4 brick pose); servo on its true pose while descending."""
        st, brick, env = self.arm[a], BRICKS[b], self.env
        H = np.linalg.inv(tcp_T(env, a)) @ body_T(env, brick.name)

        def tcp_for(Gb, dz=0.0):
            T = Gb @ np.linalg.inv(H)
            T[2, 3] += dz
            return T
        st.phase = f"place {brick.name}"
        yield from self.move(a, lambda: self.above(tcp_for(goal_fn()), self.th + TRANSIT_Z))
        yield from self.move(a, lambda: tcp_for(goal_fn(), HOVER_Z))
        corr = np.zeros(2)

        def servo(dz_from, dz_to, seconds, settle):
            """Move the brick from dz_from to dz_to above its goal while servoing its true xy onto the goal (the
            hand trails its target and sags a little under load); with settle, end only once it is aligned."""
            nonlocal H, corr
            n, good = max(1, round(seconds / self.dt)), 0
            for k in range(n + (round(2.0 / self.dt) if settle else 0)):
                G, Bt = goal_fn(), body_T(env, brick.name)
                H = np.linalg.inv(tcp_T(env, a)) @ Bt
                err = G[:2, 3] - Bt[:2, 3]
                corr = np.clip(corr + 0.25 * err, -0.01, 0.01)
                s = min(1.0, (k + 1) / n)
                T = tcp_for(G, dz_from + s * (dz_to - dz_from))
                T[:2, 3] += corr
                st.T = T
                yield
                good = good + 1 if np.linalg.norm(err) < ALIGN_TOL else 0
                if k + 1 >= n and (not settle or good >= 8):
                    break

        for attempt in range(MAX_TRIES if press else 1):
            st.phase = f"align {brick.name} #{attempt + 1}"
            yield from servo(HOVER_Z, HOVER_Z, 0.1, settle=True)
            yield from servo(HOVER_Z, STUD_CLEAR, (HOVER_Z - STUD_CLEAR) / SLOW, settle=True)
            st.phase = f"descend {brick.name} #{attempt + 1}"
            # last few mm onto the studs: straight down, xy held
            T0 = st.T.copy()
            n = round(STUD_CLEAR / 0.012 / self.dt)
            for k in range(n):
                T = T0.copy()
                T[2, 3] -= STUD_CLEAR * (k + 1) / n
                st.T = T
                yield
            if not press:
                break
            st.phase = f"press {brick.name} #{attempt + 1}"
            T0 = st.T.copy()
            for k in range(round(CLICK_TIMEOUT_S / self.dt)):
                T = T0.copy()
                T[2, 3] -= PRESS * min(1.0, (k + 1) / 15)
                st.T = T
                yield
                if env.progress() >= 1:
                    break
            if env.progress() >= 1:
                yield from self.wait(a, 0.15)
                break
            # not clicked (resting on the stud tops): lift back to the hover height and try again
            st.phase = f"retry {brick.name}"
            T = st.T.copy()
            T[2, 3] = T0[2, 3] + HOVER_Z
            yield from self.move(a, T, speed=0.05)
        if not press:
            T = st.T.copy()
            T[2, 3] -= 0.003  # set it down firmly
            st.T = T
            yield from self.wait(a, 0.3)
        st.grip = 1.0
        yield from self.wait(a, 0.5)
        yield from self.move(a, self.above(st.T, st.T[2, 3] + 0.06), speed=0.12)

    def leave_zone(self, a: str):
        st = self.arm[a]
        st.phase = "leave"
        T = self.home[a].copy()
        T[2, 3] = self.th + TRANSIT_Z + 0.05
        yield from self.move(a, T)
        st.in_zone = False

    # -- the two programs -------------------------------------------------------------------------------
    def _base_goal(self) -> np.ndarray:
        """Red's goal in the build area. The base looks the same turned 180 deg: the first call picks the turn
        that needs the least wrist rotation, later calls keep it."""
        c = self.zone + np.array(self.p.base_xy)
        if getattr(self, "_base_turn", None) is None:
            red = self.env.data.body(BRICKS[RED].name)
            H = np.linalg.inv(tcp_T(self.env, "l")) @ body_T(self.env, BRICKS[RED].name)
            cur = tcp_T(self.env, "l")[:3, :3]
            self._base_turn = min((0.0, np.pi), key=lambda turn: rot_angle(
                cur, _rot_z(self.p.base_yaw + turn) @ np.linalg.inv(H)[:3, :3]))
        return pose([c[0], c[1], self.th], _rot_z(self.p.base_yaw + self._base_turn))

    def _blue_goal(self) -> np.ndarray:
        """Where blue goes on the red brick *as it lies now*. Of the equivalent goals (tower turned 180 deg, blue
        turned 180 deg) the first call picks the one needing the least wrist turn and travel; later calls keep
        that choice (and only follow the red brick)."""
        env = self.env
        red = env.data.body(BRICKS[RED].name)
        if getattr(self, "_blue_choice", None) is None:
            cur = tcp_T(env, "r")
            best_cost = np.inf
            for flip in (0.0, np.pi):
                gp, gyaw = tower_poses(red.xpos, _quat_yaw(red.xquat) + flip)[1]
                for turn in (0.0, np.pi):
                    Rt = _rot_z(gyaw + turn) @ np.linalg.inv(self._blue_H)[:3, :3]
                    cost = rot_angle(cur[:3, :3], Rt) + 5.0 * np.linalg.norm(gp[:2] - cur[:2, 3])
                    if cost < best_cost:
                        self._blue_choice, best_cost = (flip, turn), cost
        flip, turn = self._blue_choice
        gp, gyaw = tower_poses(red.xpos, _quat_yaw(red.xquat) + flip)[1]
        return pose(gp, _rot_z(gyaw + turn))

    def _left(self):
        st = self.arm["l"]
        yield from self.wait("l", self.p.start_delay[0])
        H = yield from self.pick("l", RED)
        if H is None:
            return
        while self.arm["r"].in_zone:  # (the right arm never enters first, but be safe)
            st.phase = "wait zone"
            yield
        st.in_zone = True
        yield from self.place("l", RED, self._base_goal, press=False)
        yield from self.leave_zone("l")
        st.phase, st.done = "done", True
        while True:
            yield

    def _right(self):
        st = self.arm["r"]
        yield from self.wait("r", self.p.start_delay[1])
        H = yield from self.pick("r", BLUE)
        if H is None:
            return
        self._blue_H = H
        # wait beside the zone, on the right arm's side, until the base is down and the left arm is out
        W = self.above(st.T, self.th + TRANSIT_Z + 0.03)
        W[:2, 3] = self.zone + np.array([-0.02, -WAIT_Y]) + np.array(self.p.wait_xy)
        st.phase = "to wait"
        yield from self.move("r", W)
        st.phase = "wait"
        while not (self.arm["l"].done or (self.arm["l"].phase == "leave" and self.zone_dist("l") > ZONE_R)):
            yield
        st.in_zone = True
        yield from self.place("r", BLUE, self._blue_goal, press=True)
        yield from self.leave_zone("r")
        st.phase, st.done = "done", True
        while True:
            yield

    # -- one control tick --------------------------------------------------------------------------------
    def act(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance both programs one tick; returns the safe joint targets (14,) and grippers (2,)."""
        for a in ARMS:
            next(self.programs[a], None)
            T = self.arm[a].T
            self.ik.set_target(a, T[:3, 3], mat_to_quat_wxyz(T[:3, :3]))
        env = self.env
        q_meas = env.data.qpos[env.arm_qadr].copy()
        self.ik.follow(q_meas)
        q = self.ik.solve()
        q_safe, self.last_change = self.safety.filter(q, q_meas, env.data.qpos[env.finger_qadr])
        self.t += self.dt
        return q_safe, np.array([self.arm[a].grip for a in ARMS])


def run_episode(cfg: Config, seed: int, max_s: float = 40.0, tail_s: float = 1.5, record: bool = True,
                params: Params | None = None):
    """One expert take from env.reset(seed). Returns (info dict, recorded steps)."""
    env = BimanualEnv(cfg)
    env.reset(seed)
    rng = np.random.default_rng(seed + 7)
    p = params or Params.sample(rng)
    ex = Expert(env, cfg, p)
    steps, clicked_at, changes = [], None, []
    n = round(max_s / env.control_dt)
    for t in range(n):
        obs, state, flags = env.observe(), env.get_state(), env.contact_flags()
        q, grippers = ex.act()
        changes.append(ex.last_change)
        env.apply_action(q, grippers)
        success, prog = env.success(), env.progress()
        if record:
            steps.append(make_step(env, obs, state, q, grippers, ex.safety.tcp_poses(), success, prog, flags,
                                   teleop_fields(None), SOURCE_OPERATOR))
        env.step()
        if prog >= 1 and clicked_at is None:
            clicked_at = t
        if ex.failed or (ex.arm["r"].done and clicked_at is not None and
                         (t - clicked_at) * env.control_dt > tail_s):
            break
    info = {"seed": seed, "clicked": clicked_at is not None, "level": env.progress(), "in_area": env.in_area(),
            "click_s": None if clicked_at is None else clicked_at * env.control_dt, "steps": len(steps) or t + 1,
            "failed": ex.failed, "phases": {a: ex.arm[a].phase for a in ARMS},
            "safety_max": float(np.max(changes)), "params": p.__dict__}
    return info, steps, env
