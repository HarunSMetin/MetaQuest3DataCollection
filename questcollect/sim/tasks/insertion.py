import mujoco
import numpy as np

from questcollect.sim.tasks.base import Task, register

BOX = mujoco.mjtGeom.mjGEOM_BOX


@register
class Insertion(Task):
    """ALOHA-style bimanual insertion: lift peg (left) and socket (right), insert in mid-air."""

    name = "insertion"
    instruction = "Pick up the peg with the left arm and the socket with the right arm, then insert the peg into the socket in mid-air."
    objects = ("peg", "socket")
    peg_half = (0.06, 0.011, 0.011)
    inner = 0.016
    wall = 0.007
    sock_len = 0.05

    def build(self, spec, layout):
        super().build(spec, layout)
        th = layout.table_height
        spec.add_material(name="peg_mat", rgba=[0.95, 0.65, 0.1, 1])
        spec.add_material(name="sock_mat", rgba=[0.2, 0.45, 0.95, 1])
        ph = self.peg_half
        self.add_free_body(spec, "peg", [0.45, 0.25, th + ph[2]], [
            dict(type=BOX, size=list(ph), material="peg_mat", mass=0.05, condim=4,
                 friction=[1.5, 0.01, 0.001], priority=1),
        ])
        i, w, L = self.inner, self.wall, self.sock_len
        o = i + w
        walls = [
            dict(type=BOX, size=[L, o, w / 2], pos=[0, 0, -i - w / 2]),
            dict(type=BOX, size=[L, o, w / 2], pos=[0, 0, i + w / 2]),
            dict(type=BOX, size=[L, w / 2, i], pos=[0, -i - w / 2, 0]),
            dict(type=BOX, size=[L, w / 2, i], pos=[0, i + w / 2, 0]),
        ]
        for g in walls:
            g.update(material="sock_mat", mass=0.03, condim=4, friction=[1.5, 0.01, 0.001], priority=1)
        self.add_free_body(spec, "socket", [0.45, -0.25, th + o], walls)

    def reset(self, model, data, rng):
        th = self.table_height
        self.set_free_pose(data, "peg", [rng.uniform(0.35, 0.55), rng.uniform(0.15, 0.3), th + self.peg_half[2]],
                           yaw=rng.uniform(-0.3, 0.3))
        self.set_free_pose(data, "socket", [rng.uniform(0.35, 0.55), rng.uniform(-0.3, -0.15), th + self.inner + self.wall],
                           yaw=rng.uniform(-0.3, 0.3))

    def success(self, model, data):
        sock = data.body("socket")
        peg = data.body("peg")
        R = sock.xmat.reshape(3, 3)
        # Peg tip (the end pointing towards the socket) expressed in the socket frame.
        peg_R = peg.xmat.reshape(3, 3)
        tips = [peg.xpos + s * self.peg_half[0] * peg_R[:, 0] for s in (-1, 1)]
        inside = False
        for tip in tips:
            local = R.T @ (tip - sock.xpos)
            if abs(local[1]) < self.inner and abs(local[2]) < self.inner and abs(local[0]) < self.sock_len * 0.6:
                inside = True
        lifted = peg.xpos[2] > self.table_height + 0.04 and sock.xpos[2] > self.table_height + 0.04
        return bool(inside and lifted)
