"""Exports an MjModel's visual geometry so the web client can rebuild the scene in three.js.

The JSON describes bodies, visual geoms (primitive or mesh) and materials. Mesh vertex/face data
goes in one binary blob (float32 vertices then uint32 faces per mesh). Meshes shared by both arms
are deduplicated and large meshes are decimated (the Quest renders ~80k triangles instead of
~270k for the two arms). Normals are rebuilt client-side with crease-angle smoothing.
"""

from __future__ import annotations

import mujoco
import numpy as np

from questcollect.teleop.frames import wxyz_to_xyzw

GT = mujoco.mjtGeom
GEOM_TYPES = {int(k): v for k, v in {
    GT.mjGEOM_PLANE: "plane", GT.mjGEOM_SPHERE: "sphere", GT.mjGEOM_CAPSULE: "capsule",
    GT.mjGEOM_ELLIPSOID: "ellipsoid", GT.mjGEOM_CYLINDER: "cylinder", GT.mjGEOM_BOX: "box",
    GT.mjGEOM_MESH: "mesh",
}.items()}
VISIBLE_GROUPS = (0, 1, 2)
DECIMATE_ABOVE_FACES = 1500
DECIMATE_REDUCTION = 0.7


def _weld(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices (MuJoCo keeps per-corner copies along normal/UV seams)."""
    uniq, inv = np.unique(np.round(verts, 6), axis=0, return_inverse=True)
    f = inv.reshape(-1)[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return uniq, f[keep]


def _decimate(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = _weld(verts, faces)
    if len(faces) <= DECIMATE_ABOVE_FACES:
        return verts, faces
    try:
        import fast_simplification
    except ImportError:
        return verts, faces
    v, f = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int32),
                                        target_reduction=DECIMATE_REDUCTION)
    return v, f


def _dedup_key(name: str) -> str:
    return name[2:] if name[:2] in ("l_", "r_") else name


def export_scene(model: mujoco.MjModel) -> tuple[dict, bytes]:
    meshes: list[dict] = []
    mesh_index: dict[str, int] = {}
    blobs: list[bytes] = []
    offset = 0

    def mesh_ref(mid: int) -> int:
        nonlocal offset
        key = _dedup_key(model.mesh(mid).name)
        if key in mesh_index:
            return mesh_index[key]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts, faces = _decimate(model.mesh_vert[va:va + vn], model.mesh_face[fa:fa + fn])
        verts, faces = verts.astype("<f4"), faces.astype("<u4")
        vb, fb = verts.tobytes(), faces.tobytes()
        meshes.append({"name": key, "vertOffset": offset, "nvert": len(verts),
                       "faceOffset": offset + len(vb), "nface": len(faces)})
        blobs.extend((vb, fb))
        offset += len(vb) + len(fb)
        mesh_index[key] = len(meshes) - 1
        return mesh_index[key]

    geoms = []
    for g in range(model.ngeom):
        gtype = GEOM_TYPES.get(int(model.geom_type[g]))
        if int(model.geom_group[g]) not in VISIBLE_GROUPS or gtype is None:
            continue
        rgba = model.geom_rgba[g].copy()
        mat = int(model.geom_matid[g])
        material = {"specular": 0.3, "shininess": 0.3, "reflectance": 0.0}
        if mat >= 0:
            rgba = model.mat_rgba[mat].copy()
            material = {"specular": float(model.mat_specular[mat]), "shininess": float(model.mat_shininess[mat]),
                        "reflectance": float(model.mat_reflectance[mat])}
        if rgba[3] <= 0.0:
            continue
        geoms.append({
            "name": model.geom(g).name,
            "body": int(model.geom_bodyid[g]),
            "type": gtype,
            "size": model.geom_size[g].tolist(),
            "pos": model.geom_pos[g].tolist(),
            "quat": wxyz_to_xyzw(model.geom_quat[g]).tolist(),
            "rgba": rgba.tolist(),
            "material": material,
            "mesh": mesh_ref(int(model.geom_dataid[g])) if gtype == "mesh" else None,
        })

    bodies = [{"id": b, "name": model.body(b).name, "parent": int(model.body_parentid[b])}
              for b in range(model.nbody)]
    scene = {"nbody": int(model.nbody), "bodies": bodies, "geoms": geoms, "meshes": meshes, "blobSize": offset}
    return scene, b"".join(blobs)


def body_poses(data: mujoco.MjData) -> np.ndarray:
    """(nbody, 7) [x y z qx qy qz qw] world poses for the pose stream."""
    q = data.xquat
    return np.concatenate([data.xpos, q[:, 1:4], q[:, 0:1]], axis=1)
