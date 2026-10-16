import mujoco
import numpy as np

from questcollect.sim.tasks.base import Task, register

BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER


@register
class LiftBox(Task):
    name = "lift_box"
    instruction = "Grasp both handles of the box and lift it level, at least 15 cm above the table."
    objects = ("box",)
    half = (0.07, 0.14, 0.05)
    lift_height = 0.15

    def build(self, spec, layout):
        super().build(spec, layout)
        th = layout.table_height
        spec.add_material(name="box_mat", rgba=[0.55, 0.4, 0.85, 1], specular=0.3)
        spec.add_material(name="handle_mat", rgba=[0.85, 0.85, 0.88, 1], specular=0.6)
        hx, hy, hz = self.half
        geoms = [dict(type=BOX, size=[hx, hy, hz], material="box_mat", mass=0.4)]
        for s in (-1, 1):
            # Vertical post + horizontal bar on each side, graspable from above.
            geoms.append(dict(type=BOX, size=[0.008, 0.008, 0.03], pos=[0, s * (hy + 0.03), hz - 0.02],
                              material="handle_mat", mass=0.02))
            geoms.append(dict(type=CYL, size=[0.009, 0.045], pos=[0, s * (hy + 0.03), hz + 0.015],
                              quat=[0.7071068, 0, 0.7071068, 0], material="handle_mat", mass=0.02,
                              condim=4, friction=[1.5, 0.01, 0.001], priority=1))
            geoms.append(dict(type=BOX, size=[0.008, 0.015, 0.008], pos=[0, s * (hy + 0.015), hz - 0.02],
                              material="handle_mat", mass=0.01))
        self.add_free_body(spec, "box", [0.5, 0.0, th + hz], geoms)

    def reset(self, model, data, rng):
        self.set_free_pose(data, "box", [rng.uniform(0.42, 0.55), rng.uniform(-0.05, 0.05), self.table_height + self.half[2]],
                           yaw=rng.uniform(-0.2, 0.2))

    def success(self, model, data):
        box = data.body("box")
        level = box.xmat.reshape(3, 3)[2, 2] > np.cos(np.deg2rad(15))
        return bool(box.xpos[2] > self.table_height + self.half[2] + self.lift_height and level)
