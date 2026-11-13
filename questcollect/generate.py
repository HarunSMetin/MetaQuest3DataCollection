"""Demo generation (experimental augmentation): replay an operator's take from its TCP targets, adapted.

`replay` drives the arms through the source's recorded TCP targets (action/ee_pose) with the IK, like the live
teleop, and adapts them to the scene as it actually is:
- pick: near a brick the targets move with the brick from where it lay in the source to where it lies now;
- place: an upper brick goes where the base brick really is now (a tower slides a few mm when bricks are pressed
  in; the operator saw that, a blind replay would miss), and the hand is corrected for how the brick sits in it
  (H_src·H_new⁻¹, followed until FREEZE_Z above the seat): the brick, not the hand, follows the source's path;
- in between, the corrections blend along the source hand's path (so where the operator paused, the arm pauses).
Optionally the bricks start elsewhere (`spawn`), the tower is turned/shifted (`d_tower`, `d_base`), the arms
start elsewhere (`arm_start`), and each arm plays its source frames on its own time map (`frames`, from
retime.py). `check` accepts a take: tower built, no new arm-arm / arm-on-table contacts, and the arms and wrist
cameras never closer than ARM_CLEAR_M / CAM_CLEAR_M (or than the source came, where it came closer).

`generate` samples a new layout for a recorded take and keeps the result if it passes `check`. Source files are
never changed; results are new episode files with `source = "generated"`, `generated_from` and `generation`.
`generate_set` (`qc generate set`) and `generate_grasps` (`qc generate grasps`) run it over a folder of takes.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from questcollect.audit import CONTACT_KINDS, load_curation
from questcollect.ik import BimanualIK
from questcollect.recorder import (AUTO_STOP_TAIL_S, control_step, episode_files, episode_meta, read_episode,
                                   snapshot, teleop_fields, write_episode)
from questcollect.sim import (ARM_CLEAR_M, ARMS, BRICKS, CAM_CLEAR_M, SPAWN, SPAWN_JITTER_M, SPAWN_YAW, ArmClearance,
                              BimanualEnv, _quat_yaw, env_from_episode, executed_actions, held_bricks,
                              set_brick_pose)

PRE_S = 1.5        # pick window starts this long before the gripper closes (the final descent)
PLACE_M = 0.03     # place window starts when the source brick is within this of its final xy
POST_S = 0.6       # windows extend this long past the gripper opening (lift-off)
FREEZE_Z = 0.008   # the in-hand correction is followed until the brick is this close above its seat, then frozen
GRASP_MIN_FRAC = 0.4      # an off-centre grasp shift is 40-100 % of the allowed range, either way
GRASP_PAD_M = 0.012       # an off-centre grasp stays this far inside the brick's end (finger pad half width + margin)
MAX_CORRECTION_M = 0.03  # in-hand corrections larger than this are not applied
EXTRA_S = 3.0      # time allowed past the end of the commands for the tower to finish and settle
CLEAR_TOL_M = 0.005  # where the source itself came closer than ARM/CAM_CLEAR_M, a take may come this much closer
CONTACT_TOL = 5    # steps of arm-arm / arm-on-table contact allowed beyond the source's
REF = {i: i - 1 for i in range(1, len(BRICKS))}  # the brick each one is placed onto (set to 0 for the base)
KEYS = ["sim/state", "action/joint_pos", "action/ee_pose", "obs/joint_pos", "obs/ee_pose_left", "obs/ee_pose_right",
        "obs/objects", "teleop/engaged", "contacts"]


# -- poses -----------------------------------------------------------------------------------------
def planar(xy_from, yaw_from, xy_to, yaw_to):
    """(R 3x3, t 3, yaw) with p' = R p + t: maps a pose rigidly attached to `from` onto `to` (about z)."""
    d = yaw_to - yaw_from
    c, s = np.cos(d), np.sin(d)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    t = np.r_[np.asarray(xy_to) - R[:2, :2] @ np.asarray(xy_from), 0.0]
    return R, t, d


IDENT = (np.eye(3), np.zeros(3), 0.0)


def mat(pose7) -> np.ndarray:
    M = np.eye(4)
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, pose7[3:])
    M[:3, :3], M[:3, 3] = r.reshape(3, 3), pose7[:3]
    return M


def pose_of(M) -> tuple[np.ndarray, np.ndarray]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, M[:3, :3].flatten())
    return M[:3, 3].copy(), q


class Xf:
    """World planar transform X (left) and an optional hand-local correction D (right): pose -> X pose D."""

    def __init__(self, X=IDENT, D=None):
        self.X, self.D = X, D

    def __call__(self, pose7):
        R, t, d = self.X
        p = R @ pose7[:3] + t
        qz = np.array([np.cos(d / 2), 0, 0, np.sin(d / 2)])
        q = np.empty(4)
        mujoco.mju_mulQuat(q, qz, pose7[3:])
        if self.D is None:
            return p, q
        return pose_of(mat(np.r_[p, q]) @ self.D)


class _Place:
    def __init__(self, i, t_freeze):
        self.i, self.t_freeze = i, t_freeze
        self.D = None  # the in-hand correction, frozen at t_freeze


def _blend(fa, fb, pose7, s):
    pa, qa = fa(pose7)
    pb, qb = fb(pose7)
    p = (1 - s) * pa + s * pb
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = (1 - s) * qa + s * qb
    return p, q / np.linalg.norm(q)


def _lerp(a: np.ndarray, f: float) -> np.ndarray:
    i = min(int(np.floor(f)), len(a) - 1)
    j = min(i + 1, len(a) - 1)
    return a[i] + (a[j] - a[i]) * (f - i)


def _smooth(s):
    s = np.clip(s, 0, 1)
    return s * s * (3 - 2 * s)


def cycles(data: dict, fps: int) -> dict[str, list[tuple]]:
    """Per arm: (brick index, t_close, t_open, t_place, t_freeze) of every grasp that actually moved a brick."""
    a, ob = data["action/joint_pos"], data["obs/objects"]
    n = len(a)
    out = {}
    for arm, col in (("l", 7), ("r", 15)):
        closed = a[:, col] < 0.5
        edges = np.flatnonzero(np.diff(closed.astype(int)))
        res = []
        for tc in (e + 1 for e in edges if closed[e + 1]):
            to = next((e + 1 for e in edges if e + 1 > tc and not closed[e + 1]), n - 1)
            lift = ob[tc:to, :, 2] - ob[tc, :, 2]
            i = int(np.argmax(lift.max(axis=0)))
            if lift[:, i].max() < 0.01:
                continue  # a grasp that didn't pick anything up (regrasp, empty pinch)
            near = np.flatnonzero(np.linalg.norm(ob[tc:to, i, :2] - ob[to, i, :2], axis=1) < PLACE_M)
            tp = tc + int(near[0])
            low = np.flatnonzero(ob[tp:to, i, 2] - ob[to, i, 2] < FREEZE_Z)
            res.append((i, tc, to, tp, tp + int(low[0]) if len(low) else to))
        out[arm] = res
    return out


# -- replay ----------------------------------------------------------------------------------------
@dataclass
class Take:
    """A replayed take: recorded steps, and per step the contact flags, measured arm joints and progress."""
    steps: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    q: list = field(default_factory=list)
    progress: list = field(default_factory=list)
    end: int = 0          # steps until success held for AUTO_STOP_TAIL_S (0: never)


def replay(env: BimanualEnv, data: dict, fps: int, frames: dict[str, np.ndarray] | None = None,
           spawn: dict | None = None, d_tower: float = 0.0, d_base=(0.0, 0.0), arm_start: dict | None = None,
           joints: bool = False, grasp_shift: dict[int, np.ndarray] | None = None) -> Take:
    """Play a source take (data from read_episode with KEYS) from its first state, adapted (module docstring).
    frames: per arm the source frame to play at each new step (default: as recorded); spawn: brick name ->
    (xy, yaw) start; d_tower (rad) / d_base (m, xy): turn/shift of the tower about the source's base; arm_start:
    arm -> TCP pose [x y z qw qx qy qz] to start from.
    joints: play each arm's recorded joint targets at its frame instead (no IK, no adaptation). Exact for the
    source's own timing; for a retimed take (same layout, each arm's moves in order) the scene mostly evolves as
    recorded, and the IK path's small tracking errors (p99 7-40 mm while pressing) can't grow into a missed brick.
    Not with spawn / d_tower / d_base / arm_start, which need the IK.
    grasp_shift: brick index -> world xy (m) its start is moved by while the hand still reaches where the source
    hand did, so it grasps the brick that far off centre; the place then follows the brick through the in-hand
    correction (an off-centre grasp that still lands aligned)."""
    if joints and (spawn or d_tower or np.any(d_base) or arm_start or grasp_shift):
        raise ValueError("joints=True replays the recorded layout only")
    n = len(data["action/joint_pos"])
    frames = frames or {a: np.arange(n) for a in ARMS}
    length = max(len(f) for f in frames.values())
    # frames may be fractional (retime eases into and out of a staging hold): the joint targets are then
    # interpolated; everything else (IK targets, grippers, teleop fields) uses the nearest frame.
    ffr = {a: lambda t, a=a: float(frames[a][min(t, len(frames[a]) - 1)]) for a in ARMS}
    frame = {a: lambda t, a=a: int(round(ffr[a](t))) for a in ARMS}
    ob = data["obs/objects"]
    d_base = np.asarray(d_base, float)
    cyc = cycles(data, fps)

    env.set_state(data["sim/state"][0])
    if spawn:
        for b in BRICKS:
            xy, yaw = spawn[b.name]
            set_brick_pose(env.data, b.name, [*xy, env.table_height], yaw)
        mujoco.mj_forward(env.model, env.data)
    for i, dxy in (grasp_shift or {}).items():
        b = ob[0, i]
        set_brick_pose(env.data, BRICKS[i].name, [b[0] + dxy[0], b[1] + dxy[1], b[2]], _quat_yaw(b[3:]))
    if grasp_shift:
        mujoco.mj_forward(env.model, env.data)
    ee_src = {"l": data["action/ee_pose"][:, 0:7], "r": data["action/ee_pose"][:, 7:14]}
    ik = BimanualIK(env.cfg.sim, env.cfg.ik, env.control_dt)
    ik.reset(env.data.qpos[env.arm_qadr])
    T_start = {a: Xf() for a in ARMS}
    if arm_start:
        for a in ARMS:
            ik.set_target(a, arm_start[a][:3], arm_start[a][3:])
        for _ in range(150):  # 3 s of IK steps: converges from the source pose
            q_arm = ik.solve()
        env.data.qpos[env.arm_qadr] = q_arm
        env.data.qvel[env.arm_vadr] = 0
        env.apply_action(q_arm, data["action/joint_pos"][0, [7, 15]])
        env.data.ctrl[env.arm_act] = q_arm
        mujoco.mj_forward(env.model, env.data)
        ik.reset(q_arm)
        for a in ARMS:
            p, q = ik.tcp_pose(a)
            T_start[a] = Xf(IDENT, np.linalg.inv(mat(ee_src[a][0])) @ mat(np.r_[p, q]))
    T_pick = {}
    for i, b in enumerate(BRICKS):
        src_xy, src_yaw = ob[0, i, :2], _quat_yaw(ob[0, i, 3:])
        T_pick[i] = Xf(planar(src_xy, src_yaw, *spawn[b.name]) if spawn else IDENT)

    # Per arm the anchor windows (source frames): [start, end, transform or _Place].
    windows = {a: [] for a in ARMS}
    for a in ARMS:
        for i, tc, to, tp, tf in cyc[a]:
            windows[a].append([max(0, tc - round(PRE_S * fps)), tc + 1, T_pick[i]])
            windows[a].append([tp, min(n - 1, to + round(POST_S * fps)), _Place(i, tf)])
    # Blends advance with the source hand's path (m, + 0.2 m/rad), not time: where the operator paused, the arm
    # pauses too.
    path = {}
    for a in ARMS:
        e = ee_src[a]
        dq = np.abs(np.sum(e[1:, 3:] * e[:-1, 3:], axis=1)).clip(0, 1)
        path[a] = np.r_[0, np.cumsum(np.linalg.norm(np.diff(e[:, :3], axis=0), axis=1) + 0.2 * np.arccos(dq))]

    def progress_s(a, f0, f1, f):
        L = path[a][f1] - path[a][f0]
        if L < 0.01:
            return _smooth((f - f0) / max(1, f1 - f0))
        return _smooth((path[a][f] - path[a][f0]) / L)

    ee_meas = {"l": data["obs/ee_pose_left"], "r": data["obs/ee_pose_right"]}
    now = {}

    def place_T(a, i, f, D=None):
        """Base shift (upper bricks follow where the base really is, also while it slides under the press) +
        in-hand correction (D: the frozen one), so the brick, not the hand, follows the source brick's path
        relative to the tower."""
        X = IDENT
        if i == 0 and (d_tower or d_base.any()):  # move the base from where the source put it; the rest follows it
            xy, yaw = ob[-1, 0, :2], _quat_yaw(ob[-1, 0, 3:])
            X = planar(xy, yaw, xy + d_base, yaw + d_tower)
        if i > 0:  # relative to the brick it goes onto (whose studs it must meet), wherever that is now
            cur = env.data.body(BRICKS[REF[i]].name)
            X = planar(ob[f, REF[i], :2], _quat_yaw(ob[f, REF[i], 3:]), cur.xpos[:2], _quat_yaw(cur.xquat))
        if D is not None:
            return Xf(X, D)
        if now["held"].get(a) != i:  # the brick isn't in this hand (dropped, or not grasped yet): no correction,
            return Xf(X, np.eye(4))  # else hand⁻¹·brick over a long lever turns into a jump of the target
        H_src = np.linalg.inv(mat(ee_meas[a][f])) @ mat(ob[f, i])
        o = now["obs"]
        H_new = np.linalg.inv(mat(o["ee_pose_left" if a == "l" else "ee_pose_right"])) @ mat(o["objects"][i])
        D = H_src @ np.linalg.inv(H_new)
        if np.linalg.norm(D[:3, 3]) > MAX_CORRECTION_M:  # the brick sits far off how it sat in the source hand:
            D = np.eye(4)                                 # a correction this large would throw the arm
        return Xf(X, D)

    def target(a, f):
        pose = ee_src[a][f]
        prev_end, prev_T = 0, T_start[a]
        for w in windows[a]:
            pl = w[2] if isinstance(w[2], _Place) else None
            if pl and w[0] <= f <= w[1]:
                if f >= pl.t_freeze and pl.D is None:  # the brick is about to seat: keep how it sits in the hand
                    pl.D = place_T(a, pl.i, pl.t_freeze).D
                return place_T(a, pl.i, f, pl.D)(pose)
            if f < w[0]:
                Tn = place_T(a, pl.i, f) if pl else w[2]
                return _blend(prev_T, Tn, pose, progress_s(a, prev_end, w[0], f))
            if f <= w[1]:
                return w[2](pose)
            prev_end, prev_T = w[1], (place_T(a, pl.i, w[1], pl.D) if pl else w[2])
        return _blend(prev_T, Xf(), pose, progress_s(a, prev_end, n - 1, f))

    grips = data["action/joint_pos"][:, [7, 15]]
    src_q = data["action/joint_pos"]
    eng = data["teleop/engaged"]
    take = Take()
    tail, run = round(AUTO_STOP_TAIL_S * fps), 0
    for t in range(length + round(EXTRA_S * fps)):
        before = snapshot(env)
        obs = now["obs"] = before.obs
        if not joints:
            now["held"] = held_bricks(env.model, env.data)
        f = {a: frame[a](t) for a in ARMS}
        if joints:
            q_t = np.r_[_lerp(src_q[:, 0:7], ffr["l"](t)), _lerp(src_q[:, 8:15], ffr["r"](t))]
            ee = [ee_src["l"][f["l"]], ee_src["r"][f["r"]]]
        else:
            ee = []
            for a in ARMS:
                p, q = target(a, f[a])
                ik.set_target(a, p, q)
                ee.append(np.r_[p, q])
            ik.follow(env.data.qpos[env.arm_qadr])
            q_t = ik.solve()
        g = np.array([grips[f["l"], 0], grips[f["r"], 1]])
        teleop = teleop_fields(None, [eng[f[a], i] for i, a in enumerate(ARMS)], [True, True], 0.0)
        step, ok, prog = control_step(env, before, q_t, g, ee, teleop)
        take.steps.append(step)
        take.flags.append(before.contacts)
        take.q.append(obs["joint_pos"])
        take.progress.append(prog)
        run = run + 1 if ok else 0
        if run >= tail:
            take.end = t + 1
            break
    return take


# -- acceptance ------------------------------------------------------------------------------------
@dataclass
class Limits:
    """What a take derived from a source may do: contact steps (CONTACT_KINDS) and closest distances [m]."""
    contacts: dict[str, int]
    arm_m: float
    cam_m: float

    @classmethod
    def of_source(cls, data: dict, clearance: ArmClearance) -> Limits:
        f = np.asarray(data["contacts"], bool)
        src = clearance.measure(data["obs/joint_pos"][::2]).min(axis=0)
        lim = lambda fixed, own: fixed if own >= fixed else own - CLEAR_TOL_M  # noqa: E731
        return cls({k: int(f[:, i].sum()) + CONTACT_TOL for i, k in enumerate(CONTACT_KINDS)},
                   lim(ARM_CLEAR_M, src[0]), lim(CAM_CLEAR_M, src[1]))


def check(take: Take, limits: Limits, clearance: ArmClearance) -> tuple[int | None, dict]:
    """(first bad step or None, summary): a bad step is a contact beyond the limits, the arms or a wrist camera too
    close, or (tower not built) the last step."""
    flags = np.array(take.flags, bool)
    clear = clearance.measure(np.array(take.q[::2]))
    bad = []
    for i, k in enumerate(CONTACT_KINDS):
        hits = np.flatnonzero(flags[:, i])
        if k in ("arm_arm", "link_table") and len(hits) > limits.contacts[k]:
            bad.append(int(hits[limits.contacts[k]]))
    near = np.flatnonzero((clear[:, 0] < limits.arm_m) | (clear[:, 1] < limits.cam_m))
    if len(near):
        bad.append(int(near[0]) * 2)
    if not take.end:
        bad.append(len(take.steps) - 1)
    summary = {"success": bool(take.end), "progress": int(take.progress[-1]),
               "contacts": {k: int(flags[:, i].sum()) for i, k in enumerate(CONTACT_KINDS)},
               "arm_m": round(float(clear[:, 0].min()), 4), "cam_m": round(float(clear[:, 1].min()), 4)}
    return (min(bad) if bad else None), summary


# -- generation ------------------------------------------------------------------------------------
def generate(src: Path, seed: int, out_dir: Path | None, identity: bool = False, keep_failed: bool = False,
             jitter: float = SPAWN_JITTER_M, spawn_yaw: float = SPAWN_YAW, tower_yaw: float = 0.0,
             tower_frac: float = 1.0, base_xy: float = 0.0, arm_xy: float = 0.0, arm_yaw: float = 0.0,
             grasp_offset: float = 0.0, grasp_across: float = 0.0, grasp_bricks: tuple[int, ...] = (0, 1),
             layout: bool = True, min_cam_m: float | None = None) -> dict:
    """One take from src in a new layout drawn from seed; written to out_dir if it passes `check` (or keep_failed).
    jitter (m) / spawn_yaw (rad): +- range of each brick's start around its SPAWN centre; tower_yaw (rad): +- turn
    of the whole tower about the source's base, in a tower_frac share of the takes; base_xy (m): +- shift of the
    base; arm_xy (m) / arm_yaw (rad): +- offset of each arm's starting TCP (z gets half of arm_xy), which fades
    out on the way to its first brick. identity: replay the source unchanged (a check of the replay itself).
    grasp_offset / grasp_across (m): +- off-centre grasp of each brick in grasp_bricks, along its long axis
    (capped so the finger pads stay on the brick) / across it; layout=False keeps the source layout otherwise.
    min_cam_m: accept wrist cameras this close (m) instead of CAM_CLEAR_M (the IK replay of a retimed take runs a
    centimetre closer than its joint replay; nothing touches)."""
    src = Path(src)
    meta, data = read_episode(src, keys=KEYS)
    fps = int(meta["fps"])
    env = env_from_episode(meta)
    data = executed_actions(env, data)  # the targets that built the tower, not the operator's before the DART drift
    rng = np.random.default_rng(seed)
    spawn = {b.name: (np.asarray(SPAWN[b.name]) + rng.uniform(-jitter, jitter, 2), rng.uniform(-spawn_yaw, spawn_yaw))
             for b in BRICKS}
    d_tower = rng.uniform(-tower_yaw, tower_yaw) if tower_yaw > 0 and rng.random() < tower_frac else 0.0
    d_base = rng.uniform(-base_xy, base_xy, 2) if base_xy > 0 else np.zeros(2)
    arm_start = None
    if arm_xy > 0 or arm_yaw > 0:
        arm_start = {}
        for i, a in enumerate(ARMS):
            p0 = data["action/ee_pose"][0, 7 * i:7 * i + 7]
            off = rng.uniform(-1, 1, 3) * [arm_xy, arm_xy, arm_xy / 2]
            _, q = Xf(planar([0, 0], 0, [0, 0], rng.uniform(-arm_yaw, arm_yaw)))(p0)
            arm_start[a] = np.r_[p0[:3] + off, q]
    grasp_shift = {}
    if grasp_offset > 0 or grasp_across > 0:
        first_grasp = {}  # brick -> (arm, close frame) of the source's first grasp of it
        for a, cs in cycles(data, fps).items():
            for i, tc, *_ in cs:
                if i not in first_grasp or tc < first_grasp[i][1]:
                    first_grasp[i] = (a, tc)
        for i in grasp_bricks:
            if i not in first_grasp:
                continue
            a, tc = first_grasp[i]
            r = np.zeros(9)
            mujoco.mju_quat2Mat(r, data["action/ee_pose"][tc, 7 * ARMS.index(a) + 3:7 * ARMS.index(a) + 7])
            close = r.reshape(3, 3)[:2, 1]  # the fingers close along the TCP y axis
            yaw = _quat_yaw(data["obs/objects"][0, i, 3:])
            bx, by = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
            # slide along the brick axis the fingers don't close on (between the pads), capped at its end
            (along, half), (across, _) = sorted([(bx, BRICKS[i].half[0]), (by, BRICKS[i].half[1])],
                                                key=lambda e: abs(e[0] @ close))
            along_max = max(0.0, min(grasp_offset, half - GRASP_PAD_M))
            u = rng.choice([-1, 1]) * rng.uniform(GRASP_MIN_FRAC, 1.0) * along_max  # mostly clearly off centre
            grasp_shift[i] = u * along + rng.uniform(-grasp_across, grasp_across) * across
    if identity or not layout:
        spawn, d_tower, d_base, arm_start = None, 0.0, np.zeros(2), None
    if identity:
        grasp_shift = {}
    take = replay(env, data, fps, spawn=spawn, d_tower=d_tower, d_base=d_base, arm_start=arm_start,
                  grasp_shift=grasp_shift or None)
    clearance = ArmClearance(env.cfg.sim)
    limits = Limits.of_source(data, clearance)
    if min_cam_m is not None:
        limits.cam_m = min(limits.cam_m, min_cam_m)
    bad, summary = check(take, limits, clearance)
    res = {"src": src.name, "seed": seed, "ok": bad is None, **summary, "len_s": round(len(take.steps) / fps, 1),
           "tower_yaw_deg": round(float(np.rad2deg(d_tower)), 1),
           "base_shift_mm": [round(float(v) * 1000, 1) for v in d_base],
           "grasp_shift_mm": {BRICKS[i].name: [round(float(x) * 1000, 1) for x in d] for i, d in grasp_shift.items()}}
    if out_dir is not None and (bad is None or keep_failed):
        env.seed = seed
        m = episode_meta(env.cfg, env, len(take.steps), bool(take.end),
                         input_mode=meta.get("input_mode", "controllers"), operator=meta.get("operator", "operator"),
                         source="generated")
        m["generated_from"] = str(src)
        m["generation"] = json.dumps({
            "seed": seed, "spawn_jitter_m": jitter, "spawn_yaw_deg": float(np.rad2deg(spawn_yaw)),
            "tower_yaw_range_deg": float(np.rad2deg(tower_yaw)), "tower_yaw_deg": float(np.rad2deg(d_tower)),
            "tower_frac": tower_frac, "base_xy_m": base_xy, "base_shift_m": d_base.tolist(), "arm_xy_m": arm_xy,
            "arm_yaw_deg": float(np.rad2deg(arm_yaw)), "layout": layout and not identity,
            "grasp_offset_m": grasp_offset, "grasp_across_m": grasp_across, "min_cam_m": min_cam_m,
            "grasp_shift_m": {BRICKS[i].name: d.tolist() for i, d in grasp_shift.items()}})
        res["path"] = str(write_episode(Path(out_dir) / f"gen_{src.stem}_s{seed}.hdf5", take.steps, m))
    return res


# -- whole sets: `qc generate set`, `qc generate grasps` ---------------------------------------------------------
SET_SEED_BASE = 10_000    # far from the eval seeds (EVAL_SEED_BASE); each source gets its own block of 1000 seeds
GRASP_SEED_BASE = 70_000  # grasp sets: 100 seeds per source


def _set_job(src: str, seed: int, tmp: str, kw: dict) -> dict:
    return generate(Path(src), seed, Path(tmp), **kw)


def generate_set(src_dir: Path, out: Path, per: int = 10, max_tries: int = 30, seed_base: int = SET_SEED_BASE,
                 jobs: int = 6, **kw) -> dict:
    """A balanced set: `per` kept takes (new layouts, `generate` options in kw) from each source take.

    Writes episode_XXXXXX.hdf5 (what qc audit / export read) and names.txt (episode -> source, seed, layout).
    Seeds go in rounds, so every source keeps a few jobs in flight and stops once it has `per` or `max_tries`."""
    out, tmp = Path(out), Path(out) / "_tmp"
    sources = episode_files(Path(src_dir))
    kept = {s.name: [] for s in sources}
    tries = dict.fromkeys(kept, 0)
    with ProcessPoolExecutor(jobs) as pool:
        futs = {}

        def submit(src):
            if len(kept[src.name]) + sum(f_src == src.name for f_src, _ in futs.values()) >= per \
                    or tries[src.name] >= max_tries:
                return
            seed = seed_base + sources.index(src) * 1000 + tries[src.name]
            tries[src.name] += 1
            futs[pool.submit(_set_job, str(src), seed, str(tmp), kw)] = (src.name, seed)

        for _ in range(2):
            for s in sources:
                submit(s)
        while futs:
            f = next(as_completed(futs))
            name, _ = futs.pop(f)
            r = f.result()
            if r["ok"] and len(kept[name]) < per:
                kept[name].append(r)
            elif r.get("path"):
                Path(r["path"]).unlink()
            print(f"{name}: {len(kept[name])}/{per} kept, {tries[name]} tried", flush=True)
            submit(next(s for s in sources if s.name == name))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in (r for s in sources for r in kept[s.name]):
        dst = out / f"episode_{len(rows):06d}.hdf5"
        Path(r["path"]).rename(dst)
        rows.append({"episode": dst.name, **{k: r[k] for k in ("src", "seed", "tower_yaw_deg", "base_shift_mm",
                                                               "len_s", "arm_m", "cam_m")}})
    for p in tmp.glob("*.hdf5"):
        p.unlink()
    if tmp.exists():
        tmp.rmdir()
    (out / "names.txt").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"kept": len(rows), "tried": sum(tries.values()), "per_source": {k: len(v) for k, v in kept.items()}}


def _grasp_job(job: tuple) -> dict:
    src, seed, out, kw = job
    try:
        r = generate(Path(src), seed, Path(out), layout=False, **kw)
        return {k: r.get(k) for k in ("src", "seed", "ok", "progress", "grasp_shift_mm", "arm_m", "cam_m")}
    except Exception as e:  # noqa: BLE001 - one bad source shouldn't stop the set
        return {"src": Path(src).name, "seed": seed, "ok": False, "error": repr(e)}


def generate_grasps(src_dir: Path, out: Path, seeds: int = 6, seed_base: int = GRASP_SEED_BASE, jobs: int = 6,
                    **kw) -> int:
    """Off-centre grasps: each source take (minus those excluded with `qc audit --exclude`) replayed in its own
    layout with the bricks shifted between the finger pads (`generate` grasp options in kw), `seeds` tries each.
    Every take is simulated and kept only if it passes `check`. Prints one JSON line per try; returns kept."""
    Path(out).mkdir(parents=True, exist_ok=True)
    excluded = {f for f, c in load_curation(Path(src_dir)).items() if c.get("exclude")}
    srcs = [s for s in sorted(Path(src_dir).glob("episode_*.hdf5")) if s.name not in excluded]
    print(f"{len(srcs)} sources ({len(excluded)} excluded by hand: {sorted(excluded)})", flush=True)
    jobs_ = [(str(s), seed_base + 100 * n + k, str(out), kw) for n, s in enumerate(srcs) for k in range(seeds)]
    ok = 0
    with ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(_grasp_job, jobs_):
            ok += bool(r.get("ok"))
            print(json.dumps(r), flush=True)
    print(f"{ok}/{len(jobs_)} takes kept -> {out}")
    return ok
