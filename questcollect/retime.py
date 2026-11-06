"""Retiming (experimental augmentation): turn a one-arm-at-a-time take into one where both arms work at once.

With one operator, an arm mostly waits while the other works (the operator attends to one hand at a time), and a
policy trained on that learns to wait too. Retiming reschedules the take: each arm's recorded command stream is
split into moves (where its commands change) and holds (where they don't), and every move starts as early as
it may, so a waiting arm's next move overlaps the other arm's work. Staging (default): a move that goes to the
tower is split just outside the tower zone, at the slowest point of the operator's approach, so the arm carries
its brick there while the other arm still works at the tower, slows into a wait (eased: a move stopped dead
shakes the brick in the hand) and places once the other arm lifts off. Elsewhere an arm only waits where the
operator stopped too. "As early as it may":
- tower zone: only one arm at a time near the tower (TCP within ZONE_R of the base), in the recorded order; a move
  may start before the zone is free if it gets there after;
- clearance: the arms stay ARM_CLEAR_M apart and each wrist camera CAM_CLEAR_M from the other arm, or as close as
  the source take came where it came closer (generate.Limits; planned on the commanded joints with a margin, the
  arms' poses as recorded together excepted). Physics contacts alone miss a wrist camera inside the other arm;
- gripper: a hold before a move that grasps or releases, or right after a gripper command, keeps up to
  GRIP_SETTLE_S: the arm starts a grasp settled, and the fingers travel at 50 mm/s.
Holds are time-warped (resampled), not cut, so the commands stay continuous. The schedule is played back with
generate.replay, first with each arm's own recorded joint targets on its new timeline (exact motion; the IK path's
tracking errors of a few mm while pressing grew into missed bricks and a dragged tower in 25 of 71 takes), else
through the IK with each brick placed where the tower really is, and kept if it passes generate.check (tower
built, no new contacts, clearance); otherwise the moves around the failure start later and it is tried again.

The source episodes are never modified; results are new episode files (same names) in another folder, with
`source = "retimed"`, `retimed_from` and the moves that start earlier (`retime_moves`) in their attributes. Teleop input
fields follow each arm's stream (engaged); raw XR poses stay in the source file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from questcollect import generate
from questcollect.audit import true_runs
from questcollect.recorder import episode_meta, read_episode, write_episode
from questcollect.sim import ARM_CLEAR_M, ARMS, CAM_CLEAR_M, ArmClearance, env_from_episode, executed_actions

MOVING_RAD_S = 0.02     # an arm moves when a joint command changes faster than this
MIN_HOLD_S = 0.4        # stiller stretches shorter than this belong to the move around them
KEEP_S = 0.2            # a shortened hold keeps this much (the arm settles, then starts)
GRIP_SETTLE_S = 0.8     # ... or this much next to a gripper command (the arm settles; fingers travel at 50 mm/s)
ZONE_R = 0.12           # the tower zone: TCP within this of the base brick (xy)
STEP_S = 0.1            # start times are searched on this grid
CHECK_EVERY = 3         # clearance is checked every this many steps while scheduling
MARGIN_M = 0.005        # extra clearance while scheduling (the measured arm trails its commands)
PLAN_SLACK_M = 0.003    # ... and at least this much more than the limit generate.check applies
SAME_TIMING = 3         # ... except for pairs of frames this close in source time (as recorded: known to be fine)
DELAY_S = 0.5           # a move that failed the check starts this much later per retry
MAX_TRIES = 8
# Staging: a move that enters the tower zone is split just outside it, so the arm can carry its brick there early
# and wait (beside the tower, clear of the other arm) until the zone is free, instead of starting the whole move
# late. The split is at the slowest point of the approach in STAGE_WIN_S before the zone; the arm eases into and
# out of the wait over EASE_S (no stop mid-motion: that would shake the brick in the hand).
STAGE_MIN_S = 1.0       # only moves that approach the zone for at least this long
STAGE_WIN_S = 1.5       # the split is searched in this window before the zone entry ...
STAGE_LEAD_S = 0.2      # ... ending this long before it
EASE_S = 0.3


@dataclass(frozen=True)
class Move:
    """Source steps [s, e) of one arm between two holds, played back as one; [zone_s, zone_e): the part of it near
    the tower (zone_s = -1: none)."""
    arm: str
    s: int
    e: int
    zone_s: int = -1
    zone_e: int = -1
    settle: bool = False   # it ends right after a gripper command
    grips: bool = False    # it has a gripper command (a grasp or release: the arm starts it settled)
    staged: bool = False   # it ends (approach) or starts (zone part) at a staging split, not a recorded hold

    @property
    def n(self) -> int:
        return self.e - self.s

    @property
    def zone(self) -> bool:
        return self.zone_s >= 0


@dataclass
class RetimeResult:
    source: str
    fps: int
    steps_before: int
    steps_after: int
    moves: list[dict]             # {"arm", "at_s" (new timeline), "earlier_s"} for every move that starts earlier
    rejected: int                 # schedules that failed generate.check
    min_clearance: dict = field(default_factory=dict)  # {"arm_m", "cam_m"} measured in the result
    path: str | None = None
    mode: str = ""                # "joints" (recorded joint targets) or "ik" (TCP targets through the IK, adapted)

    @property
    def saved_frac(self) -> float:
        return 1.0 - self.steps_after / self.steps_before if self.steps_before else 0.0


class _Streams:
    """Each arm's recorded commands: [7 joints, gripper] and TCP pose."""

    def __init__(self, data: dict):
        a, ee = data["action/joint_pos"], data["action/ee_pose"]
        self.act = {"l": a[:, 0:8], "r": a[:, 8:16]}
        self.ee = {"l": ee[:, 0:7], "r": ee[:, 7:14]}


# -- the schedule ----------------------------------------------------------------------------------
def _moves(s: _Streams, tower_xy: np.ndarray, fps: int, stage: bool = False) -> list[Move]:
    """Every arm's moves between holds (stiller stretches shorter than MIN_HOLD_S belong to the move), in source
    order. A move is only split for staging (stage=True: see STAGE_*), elsewhere an arm only waits where the
    operator stopped too (a move stopped halfway shakes the brick in the hand)."""
    out = []
    for a in ARMS:
        act = s.act[a]
        d = np.abs(np.diff(act, axis=0))
        moving = np.r_[(d[:, :7].max(axis=1) > MOVING_RAD_S / fps) | (d[:, 7] > 1e-3), False]
        still = np.zeros(len(act), bool)
        for start, stop in true_runs(~moving):
            still[start:stop] = stop - start >= MIN_HOLD_S * fps
        zone = np.linalg.norm(s.ee[a][:, :2] - tower_xy, axis=1) < ZONE_R
        for start, stop in true_runs(~still):
            near = start + np.flatnonzero(zone[start:stop])
            end_grip = act[max(start, stop - round(fps)):stop, 7]  # a gripper command in its last second
            m = Move(a, int(start), int(stop), int(near[0]) if len(near) else -1,
                     int(near[-1]) + 1 if len(near) else -1, bool(np.ptp(end_grip) > 0.5),
                     bool(np.ptp(act[start:stop, 7]) > 0.5))
            out += _split_for_staging(m, s, fps) if stage else [m]
    return sorted(out, key=lambda m: (m.s, m.arm))


def _split_for_staging(m: Move, s: _Streams, fps: int) -> list[Move]:
    """[approach, zone part] of a move that reaches the tower zone after a long enough approach, else [m]."""
    if not m.zone or m.zone_s - m.s < STAGE_MIN_S * fps:
        return [m]
    lo = max(m.s + round(EASE_S * fps), m.zone_s - round(STAGE_WIN_S * fps))
    hi = m.zone_s - round(STAGE_LEAD_S * fps)
    if hi <= lo:
        return [m]
    speed = np.linalg.norm(np.diff(s.ee[m.arm][lo - 1:hi + 1, :3], axis=0), axis=1)
    k = lo + int(np.argmin(speed))
    grip = s.act[m.arm][:, 7]
    return [Move(m.arm, m.s, k, grips=bool(np.ptp(grip[m.s:k]) > 0.5), staged=True),
            Move(m.arm, k, m.e, m.zone_s, m.zone_e, m.settle, bool(np.ptp(grip[k:m.e]) > 0.5), staged=True)]


class _Timeline:
    """One arm's schedule: moves at new start times; between them it holds (source frame of the hold)."""

    def __init__(self, n_src: int):
        self.n_src = n_src
        self.items: list[tuple[int, Move]] = []   # (new start, move), in time order

    @property
    def end(self) -> int:
        return self.items[-1][0] + self.items[-1][1].n if self.items else 0

    def frame(self, t: int) -> int:
        """Source frame the arm shows at new time t (a hold shows the frame its move ended on)."""
        last = 0
        for start, p in self.items:
            if t < start:
                break
            if t < start + p.n:
                return p.s + (t - start)
            last = p.e - 1
        return last

    def to_new(self, f: int) -> int | None:
        """New time at which the move holding source frame f plays it (None: f is in a hold)."""
        for start, p in self.items:
            if p.s <= f < p.e:
                return start + f - p.s
        return None

    def indices(self, length: int, ease: int = 0) -> np.ndarray:
        """Source frame per new step, holds time-warped from the source's hold frames. ease > 0: around a staging
        split the frames are box-averaged over 2 ease + 1 steps (fractional: the arm slows into the wait and
        speeds up out of it, a C1 time curve; elsewhere the average equals the frames)."""
        out = np.empty(length, int)
        prev_t, prev_e = 0, 0
        for start, p in self.items + [(length, None)]:
            src_hold = np.arange(prev_e, p.s if p else self.n_src)  # the source frames between the moves
            gap = start - prev_t
            if gap > 0:
                if len(src_hold) == 0:
                    out[prev_t:start] = max(prev_e - 1, 0)
                elif p is None:  # after the last move: the hold as recorded, then its last frame
                    out[prev_t:start] = src_hold[np.minimum(np.arange(gap), len(src_hold) - 1)]
                else:
                    out[prev_t:start] = src_hold[(np.arange(gap) * len(src_hold)) // gap]
            if p is None:
                break
            out[start:start + p.n] = np.arange(p.s, p.e)
            prev_t, prev_e = start + p.n, p.e
        # the split's two sides: where an approach ends, and where its zone part starts
        kinks = [start + p.n if not p.zone else start for start, p in self.items if p.staged]
        if not ease or not kinks:
            return out
        f = out.astype(float)
        padded = np.pad(f, ease, mode="edge")
        avg = np.convolve(padded, np.ones(2 * ease + 1) / (2 * ease + 1), mode="valid")
        for t in kinks:
            sl = slice(max(0, t - ease), min(length, t + ease + 1))
            f[sl] = avg[sl]
        return f


class _Deadlock(Exception):
    def __init__(self, blocker: Move):
        self.blocker = blocker  # the move to start later


def _schedule(moves: list[Move], n_src: int, fps: int, too_close, delays: dict[Move, int]) -> dict[str, _Timeline]:
    """Each move at its earliest start (see the module docstring). Each arm's moves keep their order; of the two
    arms' next moves, the one that can start first is placed first (so an arm that has to get out of the other's
    way does so before the other is planned); visits to the tower zone in source order. When neither can go (an
    arm that moved on early now blocks the other), the move furthest ahead of its recorded time starts later and
    the plan is made again. too_close(arm, own_frames, other_frames) -> bool checks the arms' source frames along
    a stretch of the new timeline; delays: extra steps per move (not changed)."""
    delays = dict(delays)
    for _ in range(60):
        try:
            return _plan(moves, n_src, fps, too_close, delays)
        except _Deadlock as e:
            delays[e.blocker] = delays.get(e.blocker, 0) + round(DELAY_S * fps)
    return _plan(moves, n_src, fps, too_close, delays, give_up=True)


def _plan(moves, n_src, fps, too_close, delays, give_up=False) -> dict[str, _Timeline]:
    tl = {a: _Timeline(n_src) for a in ARMS}
    todo = {a: [m for m in moves if m.arm == a] for a in ARMS}
    zone_free = 0
    prev = {a: None for a in ARMS}
    step = max(1, round(STEP_S * fps))

    def lower_bound(m: Move) -> int:
        last = prev[m.arm]
        hold = m.s - (last.e if last else 0)  # source hold before this move
        keep = round((GRIP_SETTLE_S if (last and last.settle) or m.grips else KEEP_S) * fps)
        t = tl[m.arm].end + min(hold, keep)
        if m.zone:  # it reaches the zone only when the other arm has left it
            # (a staged zone part eases in over EASE_S, centred on its start: start it that much later, so it
            # doesn't creep towards the tower before the other arm is out)
            t = max(t, zone_free - (m.zone_s - m.s) + (round(EASE_S * fps) if m.staged else 0))
        return t + delays.get(m, 0)

    def earliest(m: Move, t: int, patience: int) -> int | None:
        o = "r" if m.arm == "l" else "l"
        for t in range(t, t + patience, step):
            ts = np.arange(t, max(t + m.n, tl[o].end) + 1, CHECK_EVERY)
            own = np.where(ts < t + m.n, m.s + ts - t, m.e - 1)
            if not too_close(m.arm, own, np.array([tl[o].frame(x) for x in ts])):
                return t
        return None

    while todo["l"] or todo["r"]:
        options = []
        for a in ARMS:
            if not todo[a]:
                continue
            m = todo[a][0]
            o = "r" if a == "l" else "l"
            if m.zone and any(q.zone and q.zone_s < m.zone_s for q in todo[o]):
                continue  # the other arm's earlier visit to the tower comes first
            lb = lower_bound(m)
            t = earliest(m, lb, 20 * fps)
            options.append((np.inf if t is None else t, m.s, a, lb))
        t, _, a, lb = min(options)
        if t == np.inf and not give_up:  # nobody can go without coming too close
            # The move furthest ahead of its recorded time yields: with both arms on their recorded timing, the
            # recorded take shows the way is clear.
            last = [(start, m) for x in ARMS for start, m in tl[x].items[-1:]]
            if last:
                raise _Deadlock(max(last, key=lambda x: x[1].s - x[0])[1])
        m = todo[a].pop(0)
        if t == np.inf:  # wait longer (generate.check decides)
            t = earliest(m, lb, 60 * fps) or lb + 60 * fps
        tl[a].items.append((int(t), m))
        prev[a] = m
        if m.zone:
            zone_free = int(t) + m.zone_e - m.s
    return tl


def _too_close_fn(clearance: ArmClearance, s: _Streams, bounds: tuple[float, float]):
    """too_close for _schedule, on precomputed per-frame poses of each arm's commanded joints (cached pairs)."""
    poses = {a: clearance.poses(a, s.act[a][:, :7]) for a in ARMS}
    cache: dict[tuple[int, int], tuple[float, float]] = {}
    b_arm, b_cam = bounds

    def too_close(arm, own, other):
        fl, fr = (own, other) if arm == "l" else (other, own)
        for i, j in zip(fl.tolist(), fr.tolist()):
            if abs(i - j) <= SAME_TIMING:
                continue  # the arms as they were at the same moment in the source
            if (i, j) not in cache:
                cache[(i, j)] = clearance.distance((poses["l"][0][i], poses["l"][1][i]),
                                                   (poses["r"][0][j], poses["r"][1][j]))
            d_arm, d_cam = cache[(i, j)]
            if d_arm < b_arm or d_cam < b_cam:
                return True
        return False
    return too_close


# -- retiming --------------------------------------------------------------------------------------
def retime_episode(path: Path, out_dir: Path | None = None, stage: bool = True) -> RetimeResult:
    """Retime one episode; writes the result to out_dir (if given and it got shorter)."""
    path = Path(path)
    meta, data = read_episode(path, keys=generate.KEYS)
    fps = int(meta["fps"])
    env = env_from_episode(meta)
    data = executed_actions(env, data)  # the targets that built the tower, not the operator's before the DART drift
    s = _Streams(data)
    n = len(data["action/joint_pos"])
    clearance = ArmClearance(env.cfg.sim)
    limits = generate.Limits.of_source(data, clearance)
    # Planning margin to the fixed limits; where the source came closer (its limit has CLEAR_TOL_M slack already),
    # just under the source's own closest commanded poses.
    src_cmd = clearance.measure(np.c_[s.act["l"][::2, :7], s.act["r"][::2, :7]]).min(axis=0)
    # Never looser than what generate.check then demands (+ PLAN_SLACK_M: the measured arm trails its commands),
    # or every retry fails the check the same way.
    plan_bounds = (max(min(ARM_CLEAR_M + MARGIN_M, src_cmd[0] - 0.003), limits.arm_m + PLAN_SLACK_M),
                   max(min(CAM_CLEAR_M + MARGIN_M, src_cmd[1] - 0.003), limits.cam_m + PLAN_SLACK_M))

    src = generate.replay(env, data, fps, joints=True)  # exact: the source's own physics
    if not src.end:
        raise ValueError(f"{path.name}: does not replay to a finished tower")
    moves = _moves(s, data["obs/objects"][-1, 0, :2], fps, stage=stage)
    too_close = _too_close_fn(clearance, s, plan_bounds)
    delays: dict[Move, int] = {}
    best, rejected = None, 0
    for _ in range(MAX_TRIES):
        tl = _schedule(moves, n, fps, too_close, delays)
        length = max(t.end for t in tl.values())
        frames = {a: tl[a].indices(length, ease=round(EASE_S * fps)) for a in ARMS}
        # Each arm's own recorded joint targets on its new timeline first (its motion exactly as recorded); if
        # the scene evolves differently enough to fail, the IK path that adapts the placements to the tower.
        for mode in ("joints", "ik"):
            take = generate.replay(env, data, fps, frames=frames, joints=mode == "joints")
            bad, summary = generate.check(take, limits, clearance)
            if bad is None:
                break
        if bad is None:
            best = (tl, take, {**summary, "mode": mode})
            break
        rejected += 1
        # Start later what each arm did at the failure that runs early: its move then, or the one that took it to
        # where it waits.
        hit = []
        for a in ARMS:
            started = [(start, m) for start, m in tl[a].items if start <= bad]
            if started and started[-1][0] < started[-1][1].s:
                hit.append(started[-1][1])
        early = [(start, m) for a in ARMS for start, m in tl[a].items if start < m.s and start <= bad]
        if not hit and early:  # or else the latest move before the failure that started early
            hit = [max(early, key=lambda x: x[0])[1]]
        if not hit:
            break
        for m in hit:
            delays[m] = delays.get(m, 0) + round(DELAY_S * fps)
    if best is None or best[1].end >= src.end:
        return RetimeResult(path.name, fps, n, src.end, [], rejected)
    tl, take, summary = best
    earlier = [{"arm": m.arm, "at_s": round(start / fps, 2), "earlier_s": round((m.s - start) / fps, 2)}
               for a in ARMS for start, m in tl[a].items if m.s - start >= fps // 2]
    result = RetimeResult(path.name, fps, n, take.end, sorted(earlier, key=lambda x: x["at_s"]), rejected,
                          {"arm_m": summary["arm_m"], "cam_m": summary["cam_m"]}, mode=summary["mode"])
    if out_dir is not None:
        env.seed = int(meta.get("seed", -1))
        out = episode_meta(env.cfg, env, len(take.steps), True, input_mode=meta.get("input_mode", "controllers"),
                           operator=meta.get("operator", "operator"), source="retimed")
        out["retimed_from"] = str(path)
        out["retime_moves"] = json.dumps(result.moves)
        out["retime_min_clearance"] = json.dumps(result.min_clearance)
        out["retime_mode"] = summary["mode"]
        if "generation" in meta:  # a retimed generated take keeps how it was generated
            out["generated_from"], out["generation"] = meta.get("generated_from", ""), json.dumps(meta["generation"])
        result.path = str(write_episode(Path(out_dir) / path.name, take.steps, out))
    return result
