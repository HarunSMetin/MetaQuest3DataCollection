"""Builds the side-by-side bimanual Franka scene with MjSpec.

World frame: z up, +x points away from the operator (towards the far table edge),
+y is the operator's left. Both arm bases sit on the near table edge facing +x.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from questcollect.config import ROOT, SimConfig

PANDA_XML = ROOT / "assets" / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"

ARMS = ("l", "r")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785])
TCP_OFFSET = 0.1034  # flange-to-fingertip-center along the hand z axis


@dataclass
class SceneLayout:
    table_height: float
    arm_spacing: float
    table_size: tuple[float, float] = (0.55, 0.75)  # half extents x, y
    table_center_x: float = 0.4

    def base_pos(self, arm: str) -> np.ndarray:
        y = self.arm_spacing / 2 if arm == "l" else -self.arm_spacing / 2
        return np.array([0.0, y, self.table_height])


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """Camera xyaxes so that the camera at `pos` looks at `target`."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, f)
    return [*right, *upv]


def _add_environment(spec: mujoco.MjSpec, layout: SceneLayout) -> None:
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
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

    hx, hy = layout.table_size
    th = layout.table_height
    table = wb.add_body(name="table", pos=[layout.table_center_x, 0, 0])
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.02],
                   pos=[0, 0, th - 0.02], material="table_mat", friction=[1.0, 0.005, 0.0001])
    for sx in (-1, 1):
        for sy in (-1, 1):
            table.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025, 0.025, (th - 0.04) / 2],
                           pos=[sx * (hx - 0.05), sy * (hy - 0.05), (th - 0.04) / 2],
                           material="leg_mat", contype=0, conaffinity=0)

    # Scene cameras (ALOHA naming).
    hi_pos = [0.12, 0.0, th + 0.95]
    wb.add_camera(name="cam_high", pos=hi_pos, xyaxes=lookat_xyaxes(hi_pos, [0.5, 0, th]), fovy=62)
    lo_pos = [1.15, 0.0, th + 0.35]
    wb.add_camera(name="cam_low", pos=lo_pos, xyaxes=lookat_xyaxes(lo_pos, [0.35, 0, th + 0.1]), fovy=60)


def _add_arm(spec: mujoco.MjSpec, arm: str, layout: SceneLayout) -> None:
    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    for body in panda.bodies:
        body.gravcomp = 1.0  # the real Franka controller compensates gravity internally
    hand = panda.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.006, 0.006, 0.006],
                  rgba=[0, 1, 1, 0.0], group=4)
    cam_pos = [0.1, 0.0, 0.0]
    hand.add_camera(name="wrist_cam", pos=cam_pos,
                    xyaxes=lookat_xyaxes(cam_pos, [0, 0, 0.16], up=[1, 0, 0]), fovy=80)
    frame = spec.worldbody.add_frame(pos=layout.base_pos(arm).tolist())
    spec.attach(panda, prefix=f"{arm}_", frame=frame)


def build_spec(cfg: SimConfig, task=None) -> mujoco.MjSpec:
    """Scene spec with both arms and (optionally) a task's objects."""
    layout = SceneLayout(cfg.table_height, cfg.arm_spacing)
    spec = mujoco.MjSpec()
    spec.modelname = "bimanual_panda"
    _add_environment(spec, layout)
    for arm in ARMS:
        _add_arm(spec, arm, layout)
        spec.add_exclude(bodyname1="table", bodyname2=f"{arm}_link0")
    # Rename wrist cameras to ALOHA names.
    spec.camera("l_wrist_cam").name = "cam_left_wrist"
    spec.camera("r_wrist_cam").name = "cam_right_wrist"
    spec.option.timestep = cfg.physics_dt
    if task is not None:
        task.build(spec, layout)
    return spec


def set_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for arm in ARMS:
        for j, q in zip(ARM_JOINTS, HOME_Q):
            data.joint(f"{arm}_{j}").qpos[0] = q
        for f in ("finger_joint1", "finger_joint2"):
            data.joint(f"{arm}_{f}").qpos[0] = 0.04
    mujoco.mj_forward(model, data)
