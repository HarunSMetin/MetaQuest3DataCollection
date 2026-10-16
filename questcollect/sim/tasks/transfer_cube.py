import mujoco
import numpy as np

from questcollect.sim.tasks.base import Task, register

BOX = mujoco.mjtGeom.mjGEOM_BOX


@register
class TransferCube(Task):
    name = "transfer_cube"
    instruction = "Pick up the red cube with the left arm, hand it over to the right arm, and place it on the green target."
    objects = ("cube",)
    half = 0.02

    def build(self, spec, layout):
        super().build(spec, layout)
        th = layout.table_height
        spec.add_material(name="cube_mat", rgba=[0.9, 0.15, 0.2, 1], specular=0.3)
        self.add_free_body(spec, "cube", [0.45, 0.25, th + self.half], [
            dict(type=BOX, size=[self.half] * 3, material="cube_mat", mass=0.08,
                 condim=4, friction=[1.5, 0.01, 0.001], priority=1),
        ])
        # Mocap so the randomized target position is part of the recorded sim state.
        target = spec.worldbody.add_body(name="target", pos=[0.45, -0.25, th], mocap=True)
        target.add_geom(name="target_pad", type=BOX, size=[0.05, 0.05, 0.001], pos=[0, 0, 0.001],
                        rgba=[0.25, 0.85, 0.4, 1], contype=0, conaffinity=0)

    def reset(self, model, data, rng):
        th = self.table_height
        cube = [rng.uniform(0.35, 0.55), rng.uniform(0.12, 0.35), th + self.half]
        self.set_free_pose(data, "cube", cube, yaw=rng.uniform(-np.pi / 4, np.pi / 4))
        data.mocap_pos[model.body("target").mocapid[0]] = [rng.uniform(0.35, 0.55), rng.uniform(-0.35, -0.15), th]

    def success(self, model, data):
        cube = data.body("cube").xpos
        tgt = data.body("target").xpos
        return bool(np.linalg.norm(cube[:2] - tgt[:2]) < 0.05 and cube[2] < self.table_height + self.half + 0.01)
