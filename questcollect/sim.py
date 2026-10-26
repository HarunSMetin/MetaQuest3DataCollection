"""The MuJoCo side: scene (two Franka arms + table + DUPLO bricks), the build_tower task, and the environment.

World frame: z up, +x points away from the operator (towards the far table edge),
+y is the operator's left. Both arm bases sit on the near table edge facing +x.
"""

from __future__ import annotations

import io
from typing import NamedTuple

import mujoco
import numpy as np
from PIL import Image

from questcollect.config import ROOT, Config, SimConfig

PANDA_XML = ROOT / "assets" / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

ARMS = ("l", "r")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])
TCP_OFFSET = 0.1034  # flange-to-fingertip-center along the hand z axis
GRIPPER_CTRL_MAX = 255.0  # Menagerie's ctrl range: 0 = closed, 255 = fully open
# Gripper like the real Franka Hand (70 N continuous, 140 N peak, 50 mm/s per finger, 80 mm stroke):
# a stiff position servo on the finger tendon whose force is capped at the configured grasp force, so
# closing on any brick squeezes with that force (Menagerie's spring squeezed 2-4 N depending on width).
FINGER_TRAVEL = 0.04      # per finger
FINGER_SPEED = 0.05       # m/s per finger
GRIP_KP = 28000.0         # N/m on the tendon (= mean finger travel); saturates within ~5 mm of contact
GRIP_KD = 200.0           # N s/m, about critical damping for the fingers
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
TABLE_HALF = (0.55, 0.75)  # half extents x, y
TABLE_CENTER_X = 0.4
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
MESH = mujoco.mjtGeom.mjGEOM_MESH

# -- task: build_tower -----------------------------------------------------------------------------
# Four real DUPLO parts (same shapes and colors every episode, start poses randomized) that the operator
# stacks into one fixed tower. Brick body frames sit at the bottom-centre of the brick.
TASK = "build_tower"
PITCH = 0.016            # stud spacing
BRICK_H = 0.0192         # height of a 1-high brick without studs (DUPLO = 2x LEGO System)
STUD_R = 0.00468         # stud diameter 9.36 mm (estimate: ~2x System's 4.8 mm minus fit)
STUD_H = 0.0045          # matches BrickLink's 42.9 mm overall height for the 1x2x2 brick
STUD_CHAMFER = 0.0015    # 45 degree lead-in on the stud top: guides a brick that is up to ~1.8 mm off into place
PLAY = 0.0003            # stud clearance per side; larger = easier to seat, sloppier fit
TOP_T = 0.002            # top plate thickness
GAP = 0.0001             # brick outline is n * PITCH - 2 * GAP, like real bricks
WALL_T = PITCH / 2 - STUD_R - PLAY - GAP  # walls hug the studs of the brick below
TUBE_R = PITCH / np.sqrt(2) - STUD_R - PLAY  # underside tubes touch the 4 surrounding studs
BRICK_FRICTION = 0.5     # ABS on ABS: no published value; measure with a tilt test (tan of slip angle)
OPERATOR_GROUP = 5       # geom group MuJoCo cameras don't render (they show 0-2): headset/dashboard only
# Build area: a taped square on the table (seen by the cameras, like tape on a real table). The tower only
# counts when its base brick lies completely inside; bricks never start in it.
BUILD_AREA_CENTER = (0.45, 0.0)
BUILD_AREA_HALF = (0.07, 0.07)
TAPE_W = 0.006
TAPE_RGBA = (0.92, 0.92, 0.88, 1)

# Clutch power. Real studs hold by a slightly elastic press fit; MuJoCo bodies are rigid, so a weld
# constraint between two bricks emulates it: it switches on when a brick sits on the stud grid and is
# pressed down hard enough, and off when the weld has to hold the bricks together with more than the pull-off
# force (tension and shear; pressing them further together is the studs' job and doesn't count). Peeling
# needs no extra rule: the edge the brick pivots on and the weld form a couple, so the moment shows up
# as weld force (peel torque ~ CLUTCH_PULL_N x half the brick length). The forces are estimates (LEGO System is quoted at ~2-3 N to press and ~3-5 N to pull apart;
# DUPLO studs are bigger); measure real bricks with a luggage scale and put the numbers here.
CLUTCH_PRESS_N = 6.0     # downward contact force between two bricks that seats them
CLUTCH_PULL_N = 10.0     # force the lock holds before the bricks come apart
CLUTCH_ALIGN_XY = 0.001  # how far off the stud grid a brick may be and still seat
CLUTCH_ALIGN_Z = 0.0015
CLUTCH_ALIGN_YAW = np.deg2rad(4)
CLUTCH_SOLREF = [0.01, 1]  # slightly compliant lock, like flexing plastic
# Rigid contacts (bricks, arm links, table top): timeconst 0.004 s (>= 2 * physics_dt for stability) and
# impedance 0.99-0.999, so a brick pressed with 50+ N sinks < 0.4 mm into another and an arm pushing at full
# torque stays on the surface. MuJoCo's default 0.9-0.95 let bricks sink > 1 mm, more than the stud play
# (the stack felt soft and springy), and arms sink centimetres into each other. The lock stays softer than
# the contacts so the studs, not the weld, carry the press and any sideways load.
RIGID_SOLREF = [0.004, 1]
RIGID_SOLIMP = [0.99, 0.999, 0.0005, 0.5, 2]
# Collision bits: everything collides on bit 0; bit 1 links robot geoms to the brick cores only. A core is an
# invisible solid box filling a brick's hollow underside that only the robot hits: pushed hard, a finger
# could otherwise tip a brick or go through a thin wall into the cavity and stay hooked there with the gripper
# open. Bricks still meet each other (studs, sockets, clutch) with the hollow shape.
ROBOT_CONTYPE = 0b11
CORE_CONAFFINITY = 0b10


class Brick(NamedTuple):
    name: str
    nx: int              # studs along the brick's x axis
    ny: int
    nz: int              # height in bricks (1 = 19.2 mm)
    rgba: tuple[float, float, float, float]
    mass: float          # kg

    @property
    def half(self) -> tuple[float, float]:
        return self.nx * PITCH / 2 - GAP, self.ny * PITCH / 2 - GAP

    @property
    def h(self) -> float:
        return self.nz * BRICK_H

    @property
    def label(self) -> str:
        size = f"{min(self.nx, self.ny)}x{max(self.nx, self.ny)}" + (f"x{self.nz}" if self.nz > 1 else "")
        return f"{self.name.split('_')[0]} {size}"


# Bottom to top. Real parts with BrickLink masses: 3011, 87084, 3437, 4066 (DUPLO has no 1-high 1x2 brick).
BRICKS = (
    Brick("red_2x4", 4, 2, 1, (0.86, 0.1, 0.12, 1), 0.0122),
    Brick("blue_2x3", 3, 2, 1, (0.1, 0.35, 0.85, 1), 0.00959),
    Brick("yellow_2x2", 2, 2, 1, (0.98, 0.78, 0.1, 1), 0.00675),
    Brick("green_1x2x2", 2, 1, 2, (0.15, 0.65, 0.25, 1), 0.00795),
)
# Pose of each brick above the base relative to the brick below: (dx, dy) in studs in the lower
# brick's frame, and relative yaw.
TOWER = ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.0, np.pi / 2))
LEVELS = len(TOWER)
OBJECTS = tuple(b.name for b in BRICKS)  # free-floating bodies whose poses are recorded
CLUTCH_PAIRS = tuple((i, j) for i in range(len(BRICKS)) for j in range(i + 1, len(BRICKS)))
INSTRUCTION = (f"Build the tower in the taped area: {BRICKS[0].label} at the bottom, then "
               + ", ".join(b.label for b in BRICKS[1:-1]) + f", and {BRICKS[-1].label} on top. "
               "Press each brick down until it clicks.")
# Correct-placement tolerances.
TOL_XY = 0.004
TOL_Z = 0.003
TOL_YAW = np.deg2rad(8)
TOWER_STILL_MPS = 0.02


def _stud_xy(n: int) -> np.ndarray:
    return (np.arange(n) - (n - 1) / 2) * PITCH


def _brick_parts(b: Brick) -> list[tuple]:
    """Hollow brick: top plate, 4 walls, chamfered studs, and tubes between the studs underneath (2-wide bricks)."""
    hx, hy = b.half
    hz = (b.h - TOP_T) / 2
    shaft = (STUD_H - STUD_CHAMFER) / 2
    studs = [(x, y) for x in _stud_xy(b.nx) for y in _stud_xy(b.ny)]
    parts = [
        (BOX, [hx, hy, TOP_T / 2], [0, 0, b.h - TOP_T / 2]),
        *[(BOX, [WALL_T / 2, hy, hz], [s * (hx - WALL_T / 2), 0, hz]) for s in (-1, 1)],
        *[(BOX, [hx - WALL_T, WALL_T / 2, hz], [0, s * (hy - WALL_T / 2), hz]) for s in (-1, 1)],
        *[(CYL, [STUD_R, shaft], [x, y, b.h + shaft]) for x, y in studs],
        *[(MESH, [], [x, y, b.h + 2 * shaft]) for x, y in studs],  # stud_top frustum
    ]
    if b.ny == 2:
        parts += [(CYL, [TUBE_R, hz], [x, 0, hz]) for x in _stud_xy(b.nx - 1)]
    return parts


def _stud_top_verts(n: int = 16) -> list[float]:
    """Frustum from the stud radius at z=0 to STUD_R - STUD_CHAMFER at z=STUD_CHAMFER (a convex mesh)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda r, z: np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], 1)  # noqa: E731
    return np.concatenate([ring(STUD_R, 0.0), ring(STUD_R - STUD_CHAMFER, STUD_CHAMFER)]).ravel().tolist()


def _volume(gtype, size) -> float:
    if gtype == MESH:  # stud_top frustum
        r1, r2 = STUD_R, STUD_R - STUD_CHAMFER
        return np.pi * STUD_CHAMFER * (r1 * r1 + r1 * r2 + r2 * r2) / 3
    return 8 * np.prod(size) if gtype == BOX else np.pi * size[0] ** 2 * 2 * size[1]


def _add_brick_geoms(body, b: Brick, **kw) -> None:
    for gtype, size, pos in _brick_parts(b):
        if gtype == MESH:
            body.add_geom(type=MESH, meshname="stud_top", pos=pos, **kw)
        else:
            body.add_geom(type=gtype, size=size, pos=pos, **kw)


def _add_task(spec: mujoco.MjSpec, th: float) -> None:
    # Stud tops are placed by their base (z=0 of the vertices); the compiler re-centres the mesh and its geoms.
    spec.add_mesh(name="stud_top", uservert=_stud_top_verts(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)
    for i, b in enumerate(BRICKS):
        spec.add_material(name=f"{b.name}_mat", rgba=list(b.rgba), specular=0.4, shininess=0.6)
        body = spec.worldbody.add_body(name=b.name, pos=[0.3 + 0.1 * i, 0.0, th])
        body.add_freejoint(name=f"{b.name}_free")
        # Uniform density over the hollow shape gives the real mass and a realistic inertia.
        density = b.mass / sum(_volume(t, size) for t, size, _ in _brick_parts(b))
        # priority=1: brick contact params win against the table; finger pads (priority 2) win against bricks.
        _add_brick_geoms(body, b, material=f"{b.name}_mat", density=density, condim=4,
                         friction=[BRICK_FRICTION, 0.005, 0.0001], solref=RIGID_SOLREF, solimp=RIGID_SOLIMP,
                         priority=1)
        hx, hy = b.half
        hz = (b.h - TOP_T) / 2
        body.add_geom(name=f"{b.name}_core", type=BOX, size=[hx - WALL_T, hy - WALL_T, hz], pos=[0, 0, hz],
                      density=0, group=3, rgba=[0, 0, 0, 0], contype=0, conaffinity=CORE_CONAFFINITY, condim=4,
                      friction=[BRICK_FRICTION, 0.005, 0.0001], solref=RIGID_SOLREF, solimp=RIGID_SOLIMP,
                      priority=1)
    # One (initially off) weld per brick pair; its relative pose is set when the pair clicks together.
    for i, j in CLUTCH_PAIRS:
        spec.add_equality(name=f"clutch_{i}_{j}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=BRICKS[i].name, name2=BRICKS[j].name, active=False, solref=CLUTCH_SOLREF,
                          data=[0] * 10 + [1])
    _add_build_area(spec, th)


def _add_build_area(spec: mujoco.MjSpec, th: float) -> None:
    """Four tape strips (visual only) just above the table top, outlining the build area from the inside."""
    (cx, cy), (hx, hy), t = BUILD_AREA_CENTER, BUILD_AREA_HALF, TAPE_W / 2
    strips = [([cx, cy + s * (hy - t)], [hx, t]) for s in (-1, 1)] + \
             [([cx + s * (hx - t), cy], [t, hy - 2 * t]) for s in (-1, 1)]
    for i, (xy, half) in enumerate(strips):
        spec.worldbody.add_geom(name=f"build_area_{i}", type=BOX, size=[*half, 0.0002], pos=[*xy, th + 0.0002],
                                rgba=list(TAPE_RGBA), group=1, contype=0, conaffinity=0)


def in_build_area(data: mujoco.MjData) -> bool:
    """Whether the base brick's whole footprint lies inside the build area."""
    b = BRICKS[0]
    body = data.body(b.name)
    R = body.xmat.reshape(3, 3)[:2, :2]
    hx, hy = b.half
    corners = body.xpos[:2] + np.array([[sx * hx, sy * hy] for sx in (-1, 1) for sy in (-1, 1)]) @ R.T
    return bool(np.all(np.abs(corners - BUILD_AREA_CENTER) <= BUILD_AREA_HALF))


def task_spec() -> dict:
    """The mission as data, for the operator UI: bricks, where each goes (relative to the base brick's bottom
    centre and yaw, in build order), and the build area. Everything the UI shows of the goal comes from here."""
    return {
        "task": TASK,
        "instruction": INSTRUCTION,
        "pitch": PITCH, "stud_r": STUD_R, "stud_h": STUD_H,
        "bricks": [{"name": b.name, "label": b.label, "size": [*(2 * np.array(b.half)).tolist(), b.h],
                    "studs": [b.nx, b.ny], "rgba": list(b.rgba)} for b in BRICKS],
        "goal": [{"brick": i, "pos": pos.tolist(), "yaw": yaw}
                 for i, (pos, yaw) in enumerate(tower_poses([0.0, 0.0, 0.0], 0.0))],
        "area": {"center": list(BUILD_AREA_CENTER), "half": list(BUILD_AREA_HALF)},
    }


def _yaw_quat(yaw: float) -> list[float]:
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def _quat_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def tower_poses(base_pos, base_yaw: float) -> list[tuple[np.ndarray, float]]:
    """Goal (position, yaw) of every brick for a tower whose base brick is at base_pos/base_yaw."""
    pos, yaw = np.asarray(base_pos, float).copy(), float(base_yaw)
    out = [(pos.copy(), yaw)]
    for b, (dx, dy, dyaw) in zip(BRICKS, TOWER):
        pos = pos + _rot_z(yaw) @ [PITCH * dx, PITCH * dy, b.h]
        yaw += dyaw
        out.append((pos.copy(), yaw))
    return out


def set_brick_pose(data: mujoco.MjData, name: str, pos, yaw: float = 0.0) -> None:
    j = data.joint(f"{name}_free")
    j.qpos[:3] = pos
    j.qpos[3:] = _yaw_quat(yaw)
    j.qvel[:] = 0


def _randomize_task(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, th: float) -> None:
    """Same bricks every episode; only where they start on the table changes (non-overlapping, outside the
    build area)."""
    placed: list[tuple[np.ndarray, float]] = []
    for b in BRICKS:
        r = float(np.hypot(*b.half))
        for _ in range(1000):
            xy = np.array([rng.uniform(0.30, 0.60), rng.uniform(-0.35, 0.35)])
            to_area = np.maximum(np.abs(xy - BUILD_AREA_CENTER) - BUILD_AREA_HALF, 0.0)
            if np.linalg.norm(to_area) > r + 0.02 and all(np.linalg.norm(xy - p) > r + pr + 0.02 for p, pr in placed):
                break
        placed.append((xy, r))
        set_brick_pose(data, b.name, [*xy, th], yaw=rng.uniform(-np.pi / 2, np.pi / 2))


def _wrap(a: float, period: float) -> float:
    return (a + period / 2) % period - period / 2


def clutch_eq(model: mujoco.MjModel, i: int, j: int) -> int:
    """Equality id of the weld between bricks i and j (either order)."""
    i, j = min(i, j), max(i, j)
    return model.equality(f"clutch_{i}_{j}").id


def stud_seated(data: mujoco.MjData, lower: int, upper: int) -> bool:
    """Whether the upper brick sits on the lower one's studs: right height, flat, yaw a multiple of 90
    degrees and its sockets within CLUTCH_ALIGN_XY of the studs (the stud lead-ins centre it that far)."""
    lo, up = BRICKS[lower], BRICKS[upper]
    bl, bu = data.body(lo.name), data.body(up.name)
    rl = bl.xmat.reshape(3, 3)
    rel = rl.T @ bu.xmat.reshape(3, 3)
    p = rl.T @ (bu.xpos - bl.xpos)
    yaw = float(np.arctan2(rel[1, 0], rel[0, 0]))
    k = round(yaw / (np.pi / 2))
    if abs(p[2] - lo.h) > CLUTCH_ALIGN_Z or rel[2, 2] < np.cos(CLUTCH_ALIGN_YAW) \
            or abs(yaw - k * np.pi / 2) > CLUTCH_ALIGN_YAW:
        return False
    ux, uy = (up.nx, up.ny) if k % 2 == 0 else (up.ny, up.nx)
    # A brick with n studs has them at (i - (n-1)/2) * PITCH, so sockets line up with studs when the
    # centre offset minus this half-stud parity term is a whole number of studs.
    parity = np.array([(ux - lo.nx) / 2, (uy - lo.ny) / 2])
    g = p[:2] / PITCH - parity
    return bool(np.all(np.abs(g - np.round(g)) * PITCH <= CLUTCH_ALIGN_XY))


def _relpose(data: mujoco.MjData, i: int, j: int) -> list[float]:
    """Brick j's current pose in brick i's frame (a weld's relpose)."""
    bi, bj = data.body(BRICKS[i].name), data.body(BRICKS[j].name)
    pos = bi.xmat.reshape(3, 3).T @ (bj.xpos - bi.xpos)
    qi, q = np.empty(4), np.empty(4)
    mujoco.mju_negQuat(qi, bi.xquat)
    mujoco.mju_mulQuat(q, qi, bj.xquat)
    return [*pos, *q]


def tower_progress(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Number of bricks (0..LEVELS) correctly stacked *and clicked in*, counted bottom-up. 0 unless the base
    brick lies flat inside the build area.

    Compared in the base brick's frame; the whole tower may be turned 180 degrees about the base (the
    base looks the same both ways), but every level has to agree on which way.
    """
    poses = [(data.body(b.name).xpos, data.body(b.name).xquat) for b in BRICKS]
    up = [data.body(b.name).xmat[8] for b in BRICKS]  # z axis . world z
    if up[0] < np.cos(TOL_YAW) or not in_build_area(data):
        return 0
    best = 0
    for flip in (0.0, np.pi):
        goal = tower_poses(poses[0][0], _quat_yaw(poses[0][1]) + flip)
        n = 0
        for k in range(1, len(BRICKS)):
            (p, q), (gp, gyaw) = poses[k], goal[k]
            period = np.pi / 2 if BRICKS[k].nx == BRICKS[k].ny else np.pi
            if not (up[k] > np.cos(TOL_YAW) and np.linalg.norm(p[:2] - gp[:2]) < TOL_XY
                    and abs(p[2] - gp[2]) < TOL_Z and abs(_wrap(_quat_yaw(q) - gyaw, period)) < TOL_YAW
                    and data.eq_active[clutch_eq(model, k - 1, k)]):
                break
            n = k
        best = max(best, n)
    return best


def held_bricks(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, int | None]:
    """Per arm, the index of the brick both of its fingers touch (None if none)."""
    brick_of = {model.body(b.name).id: i for i, b in enumerate(BRICKS)}
    finger_of = {model.body(f"{a}_{f}").id: (a, f) for a in ARMS for f in ("left_finger", "right_finger")}
    touch: dict[tuple[str, str], set[int]] = {k: set() for k in finger_of.values()}
    gb = model.geom_bodyid
    for g1, g2 in zip(data.contact.geom1, data.contact.geom2):
        b1, b2 = gb[g1], gb[g2]
        for f, o in ((b1, b2), (b2, b1)):
            if f in finger_of and o in brick_of:
                touch[finger_of[f]].add(brick_of[o])
    out: dict[str, int | None] = {}
    for a in ARMS:
        both = touch[(a, "left_finger")] & touch[(a, "right_finger")]
        out[a] = min(both) if both else None
    return out


def _task_success(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Tower complete, clicked together and standing on its own (released, at rest)."""
    if tower_progress(model, data) < LEVELS or any(v is not None for v in held_bricks(model, data).values()):
        return False
    return all(np.linalg.norm(data.joint(f"{o}_free").qvel[:3]) < TOWER_STILL_MPS for o in OBJECTS)


# -- scene -----------------------------------------------------------------------------------------
def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """Camera xyaxes so that the camera at `pos` looks at `target`."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, f)
    return [*right, *upv]


def _add_environment(spec: mujoco.MjSpec, th: float) -> None:
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Elliptic friction cones are exact (the default pyramid approximates them and lets grasps creep);
    # impratio > 1 makes friction stiffer than normal contact, which MuJoCo recommends against slip.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = [0.45, 0.45, 0.48]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]

    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.11, 0.12, 0.15], rgb2=[0.15, 0.16, 0.2], width=512, height=512,
    )
    spec.add_material(name="floor_mat", textures=["", "grid"], texrepeat=[8, 8], reflectance=0.15)
    spec.add_material(name="table_mat", rgba=[0.32, 0.27, 0.23, 1], specular=0.2)
    spec.add_material(name="leg_mat", rgba=[0.2, 0.2, 0.22, 1])

    wb = spec.worldbody
    wb.add_light(name="key", pos=[0.4, 0.0, 3.0], dir=[0, 0, -1], diffuse=[0.45, 0.45, 0.45], castshadow=True)
    wb.add_light(name="fill", pos=[-1.0, 1.0, 2.5], dir=[0.5, -0.3, -1], diffuse=[0.3, 0.3, 0.35], castshadow=False)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05], material="floor_mat")

    hx, hy = TABLE_HALF
    table = wb.add_body(name="table", pos=[TABLE_CENTER_X, 0, 0])
    table.add_geom(name="table_top", type=BOX, size=[hx, hy, 0.02],
                   pos=[0, 0, th - 0.02], material="table_mat", friction=[1.0, 0.005, 0.0001],
                   solref=RIGID_SOLREF, solimp=RIGID_SOLIMP)
    for sx in (-1, 1):
        for sy in (-1, 1):
            table.add_geom(type=BOX, size=[0.025, 0.025, (th - 0.04) / 2],
                           pos=[sx * (hx - 0.05), sy * (hy - 0.05), (th - 0.04) / 2],
                           material="leg_mat", contype=0, conaffinity=0)

    # Scene cameras (ALOHA naming).
    hi_pos = [0.12, 0.0, th + 0.95]
    wb.add_camera(name="cam_high", pos=hi_pos, xyaxes=lookat_xyaxes(hi_pos, [0.5, 0, th]), fovy=62)
    lo_pos = [1.15, 0.0, th + 0.35]
    wb.add_camera(name="cam_low", pos=lo_pos, xyaxes=lookat_xyaxes(lo_pos, [0.35, 0, th + 0.1]), fovy=60)


def _add_arm(spec: mujoco.MjSpec, arm: str, cfg: SimConfig) -> None:
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    for body in panda.bodies:
        body.gravcomp = 1.0  # the real Franka controller compensates gravity internally
    # Gripper: force = kp * (target - travel) - kd * speed on the "split" tendon (mean finger travel). Each
    # finger gets half the tendon force, so the cap is 2x the grasp force (squeeze between the fingers).
    grip = panda.actuator("actuator8")
    grip.gainprm = [GRIP_KP * FINGER_TRAVEL / GRIPPER_CTRL_MAX, *grip.gainprm[1:]]
    grip.biasprm = [0.0, -GRIP_KP, -GRIP_KD, *grip.biasprm[3:]]
    grip.forcerange = [-2 * cfg.gripper_force, 2 * cfg.gripper_force]
    for g in panda.geoms:
        if g.contype:
            g.solref, g.solimp = RIGID_SOLREF, RIGID_SOLIMP
            g.contype = ROBOT_CONTYPE
    # Grippy finger pads; priority 2 so their friction/stiffness win over the (slippery) bricks.
    # Menagerie's fingertip is a big pad plus four small boxes standing 1.5 mm proud of it with 4-5 mm slots in
    # between; a stud or brick wall pressed into such a slot gets clamped by opposing contacts and the brick
    # stays on the open finger. One flat box over the same area (same inner face, so grasp width is unchanged).
    for f in ("left_finger", "right_finger"):
        for g in panda.body(f).geoms:
            if g.classname.name == "fingertip_pad_collision_1":
                g.size, g.pos = [0.0085, 0.00475, 0.0085], [0, 0.00475, 0.0445]
            elif g.classname.name.startswith("fingertip_pad_collision"):
                panda.delete(g)
                continue
            if g.contype:
                g.friction = [1.5, 0.01, 0.001]
                g.condim, g.priority, g.solref = 4, 2, [0.004, 1]
    hand = panda.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.006, 0.006, 0.006],
                  rgba=[0, 1, 1, 0.0], group=4)
    # Grasp-point marker for the operator (headset + dashboard only, not the recorded cameras): the
    # point midway between the finger pads that the arm is steered by. The web client colours it per arm.
    hand.add_geom(name="tcp_marker", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0], pos=[0, 0, TCP_OFFSET],
                  rgba=[1, 1, 1, 1], group=OPERATOR_GROUP, contype=0, conaffinity=0, mass=0)
    cam_pos = [0.1, 0.0, 0.0]
    hand.add_camera(name="wrist_cam", pos=cam_pos,
                    xyaxes=lookat_xyaxes(cam_pos, [0, 0, 0.16], up=[1, 0, 0]), fovy=80)
    y = cfg.arm_spacing / 2 if arm == "l" else -cfg.arm_spacing / 2
    frame = spec.worldbody.add_frame(pos=[0.0, y, cfg.table_height])
    spec.attach(panda, prefix=f"{arm}_", frame=frame)


def build_spec(cfg: SimConfig, with_task: bool = True) -> mujoco.MjSpec:
    """Scene spec with both arms and (optionally) the task objects."""
    spec = mujoco.MjSpec()
    spec.modelname = "bimanual_panda"
    _add_environment(spec, cfg.table_height)
    for arm in ARMS:
        _add_arm(spec, arm, cfg)
        spec.add_exclude(bodyname1="table", bodyname2=f"{arm}_link0")
    # Rename wrist cameras to ALOHA names.
    spec.camera("l_wrist_cam").name = "cam_left_wrist"
    spec.camera("r_wrist_cam").name = "cam_right_wrist"
    spec.option.timestep = cfg.physics_dt
    if with_task:
        _add_task(spec, cfg.table_height)
    return spec


# -- environment -----------------------------------------------------------------------------------
class BimanualEnv:
    """Joint-position-controlled bimanual Franka scene with the build_tower task."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.table_height = cfg.sim.table_height
        self.spec = build_spec(cfg.sim)
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
        self.grip_step = GRIPPER_CTRL_MAX * FINGER_SPEED * self.control_dt / FINGER_TRAVEL  # ctrl per tick
        self.clutch_eqs = np.array([clutch_eq(m, i, j) for i, j in CLUTCH_PAIRS])
        self._brick_of_body = {m.body(b.name).id: i for i, b in enumerate(BRICKS)}
        self._clutch_armed = dict.fromkeys(CLUTCH_PAIRS, True)
        self.clutch_events: list[tuple[str, int, int]] = []  # ("lock" | "unlock", lower, upper), last step
        # sim/state = MuJoCo's integration state + the clutch welds' relative poses (they live in the
        # model, which mj_getState doesn't cover; which welds are on is part of the MuJoCo state).
        self.mj_state_size = mujoco.mj_stateSize(m, STATE_SPEC)
        self.state_size = self.mj_state_size + 7 * len(CLUTCH_PAIRS)
        self.seed: int | None = None

    def reset(self, seed: int | None = None) -> dict:
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2**31))
        rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        for arm in ARMS:
            for j, q in zip(ARM_JOINTS, HOME_Q):
                self.data.joint(f"{arm}_{j}").qpos[0] = q
            for f in ("finger_joint1", "finger_joint2"):
                self.data.joint(f"{arm}_{f}").qpos[0] = 0.04
        _randomize_task(self.model, self.data, rng, self.table_height)
        self._clutch_armed = dict.fromkeys(CLUTCH_PAIRS, True)
        self.clutch_events = []
        self.data.ctrl[self.grip_act] = GRIPPER_CTRL_MAX  # fingers start open
        self.apply_action(np.tile(HOME_Q, 2), np.ones(2))
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def apply_action(self, joint_targets: np.ndarray, grippers: np.ndarray) -> None:
        """joint_targets: (14,) radians [left 7, right 7]; grippers: (2,) in [0=closed, 1=open].

        The finger target moves towards the command at the real hand's 50 mm/s per finger.
        """
        self.data.ctrl[self.arm_act] = joint_targets
        cur = self.data.ctrl[self.grip_act]
        target = np.clip(grippers, 0.0, 1.0) * GRIPPER_CTRL_MAX
        self.data.ctrl[self.grip_act] = cur + np.clip(target - cur, -self.grip_step, self.grip_step)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self._update_clutch()

    # -- clutch (stud press fit) ---------------------------------------------------------------------
    def _pair_press(self) -> dict[tuple[int, int], float]:
        """Vertical contact force between each pair of touching bricks (pairs not in contact are absent)."""
        m, d = self.model, self.data
        out: dict[tuple[int, int], float] = {}
        f6 = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            a = self._brick_of_body.get(m.geom_bodyid[con.geom1])
            b = self._brick_of_body.get(m.geom_bodyid[con.geom2])
            if a is None or b is None:
                continue
            mujoco.mj_contactForce(m, d, c, f6)
            key = (min(a, b), max(a, b))
            out[key] = out.get(key, 0.0) + f6[0] * abs(con.frame[2])  # normal force * normal's z share
        return out

    def weld_pull(self, eq: int, i: int, j: int) -> float:
        """Force (N) with which the weld between bricks i and j holds them together: tension plus shear.

        The weld's 3 translational rows carry a world-frame force that is +z along the lower brick's up
        axis when body j is pulled away. Compression (the brick being pressed further down) is the studs'
        job and is left out, otherwise pressing hard on a clicked brick would pop it off.
        """
        d = self.data
        rows = np.flatnonzero((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eq))
        f = d.efc_force[rows[:3]]
        lower, upper = self._lower_upper(i, j)
        up = d.body(BRICKS[lower].name).xmat.reshape(3, 3)[:, 2]
        axial = f @ up
        tension = max(axial if upper == j else -axial, 0.0)
        return float(np.hypot(tension, np.linalg.norm(f - axial * up)))

    def _update_clutch(self) -> None:
        """Click bricks together when pressed onto the stud grid; pull (or peel) them apart past the limit."""
        m, d = self.model, self.data
        self.clutch_events = []
        press = self._pair_press()
        for (i, j), eq in zip(CLUTCH_PAIRS, self.clutch_eqs):
            if d.eq_active[eq]:
                if self.weld_pull(eq, i, j) > CLUTCH_PULL_N:
                    d.eq_active[eq] = 0
                    self._clutch_armed[(i, j)] = False  # don't re-click until they have separated
                    self.clutch_events.append(("unlock", *self._lower_upper(i, j)))
                continue
            if (i, j) not in press:
                self._clutch_armed[(i, j)] = True
                continue
            if not self._clutch_armed[(i, j)] or press[(i, j)] < CLUTCH_PRESS_N:
                continue
            seated = [(lo, up) for lo, up in ((i, j), (j, i)) if stud_seated(d, lo, up)]
            if not seated:
                continue
            lower, upper = seated[0]
            # Lock exactly where the bricks are: the stud lead-ins already centred them, and a weld that
            # had to pull them anywhere would carry force from the first step (and could pop right off).
            m.eq_data[eq] = [0, 0, 0, *_relpose(d, i, j), 1]
            d.eq_active[eq] = 1
            self.clutch_events.append(("lock", lower, upper))

    def _lower_upper(self, i: int, j: int) -> tuple[int, int]:
        return (i, j) if self.data.body(BRICKS[j].name).xpos[2] > self.data.body(BRICKS[i].name).xpos[2] else (j, i)

    def locked_pairs(self) -> list[tuple[int, int]]:
        return [pair for pair, eq in zip(CLUTCH_PAIRS, self.clutch_eqs) if self.data.eq_active[eq]]

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
            # (k, 7) [x y z qw qx qy qz] per tracked object
            "objects": np.stack([d.joint(f"{o}_free").qpos.copy() for o in OBJECTS]),
        }

    def success(self) -> bool:
        return _task_success(self.model, self.data)

    def progress(self) -> int:
        return tower_progress(self.model, self.data)

    def in_area(self) -> bool:
        return in_build_area(self.data)

    def held(self) -> dict[str, int | None]:
        return held_bricks(self.model, self.data)

    # Full state, for exact replay and offline re-rendering.
    def get_state(self) -> np.ndarray:
        s = np.empty(self.mj_state_size)
        mujoco.mj_getState(self.model, self.data, s, STATE_SPEC)
        return np.concatenate([s, self.model.eq_data[self.clutch_eqs, 3:10].ravel()])

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        self.model.eq_data[self.clutch_eqs, 3:10] = state[self.mj_state_size:].reshape(-1, 7)
        mujoco.mj_setState(self.model, self.data, state[:self.mj_state_size], STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)


class AppearanceRandomizer:
    """Per-episode visual variation for exported camera images (domain randomization).

    Policies trained on images from one fixed look tend to fail on a real camera. This varies lighting,
    table/floor colour, brick brightness (not hue: the instruction names the colours) and the camera
    mounts a little. Rendering only: physics and the recorded data are untouched.
    """

    FIELDS = ("light_diffuse", "light_pos", "mat_rgba", "cam_pos")

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.base = {k: getattr(model, k).copy() for k in self.FIELDS}
        self.scene_cams = [model.camera(n).id for n in ("cam_high", "cam_low")]
        self.wrist_cams = [model.camera(n).id for n in ("cam_left_wrist", "cam_right_wrist")]
        self.surface_mats = [model.material(n).id for n in ("table_mat", "floor_mat")]
        self.brick_mats = [model.material(f"{b.name}_mat").id for b in BRICKS]

    def apply(self, rng: np.random.Generator) -> None:
        m = self.model
        self.restore()
        m.light_diffuse[:] *= rng.uniform(0.6, 1.4, (m.nlight, 1))
        m.light_pos[:, :2] += rng.uniform(-0.4, 0.4, (m.nlight, 2))
        m.mat_rgba[self.surface_mats, :3] *= rng.uniform(0.8, 1.2, (len(self.surface_mats), 3))
        m.mat_rgba[self.brick_mats, :3] *= rng.uniform(0.85, 1.15, (len(self.brick_mats), 1))
        np.clip(m.mat_rgba, 0.0, 1.0, out=m.mat_rgba)
        m.cam_pos[self.scene_cams] += rng.uniform(-0.015, 0.015, (len(self.scene_cams), 3))
        m.cam_pos[self.wrist_cams] += rng.uniform(-0.003, 0.003, (len(self.wrist_cams), 3))

    def restore(self) -> None:
        for k, v in self.base.items():
            getattr(self.model, k)[:] = v


class CameraRenderer:
    """Offscreen rendering of the named cameras (dashboard previews and LeRobot export)."""

    def __init__(self, model: mujoco.MjModel, height: int, width: int, cameras: list[str]):
        self.cameras = list(cameras)
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for cam in self.cameras:
            self.renderer.update_scene(data, camera=cam)
            out[cam] = self.renderer.render().copy()
        return out

    def render_jpeg(self, data: mujoco.MjData, quality: int = 70) -> dict[str, bytes]:
        out = {}
        for cam, img in self.render(data).items():
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=quality)
            out[cam] = buf.getvalue()
        return out

    def close(self) -> None:
        self.renderer.close()
