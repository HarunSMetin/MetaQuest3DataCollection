// Holographic hand skeletons, stylized controllers and pointer rays.

import * as THREE from "three";
import { type HandSample, SRC_CONTROLLER, SRC_HAND } from "../protocol";

// Bone pairs (joint indices) for the skeleton lines.
const BONES: [number, number][] = [];
for (const base of [1, 5, 10, 15, 20]) {
  const len = base === 1 ? 4 : 5;
  BONES.push([0, base]);
  for (let k = 0; k < len - 1; k++) BONES.push([base + k, base + k + 1]);
}

export class HandViz {
  readonly group = new THREE.Group();
  private joints: THREE.InstancedMesh;
  private bones: THREE.LineSegments;
  private jointMat: THREE.MeshStandardMaterial;
  private boneMat: THREE.LineBasicMaterial;
  private controller: THREE.Group;
  private m = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private p = new THREE.Vector3();
  private s = new THREE.Vector3();

  constructor(readonly color: THREE.Color) {
    this.jointMat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.5, roughness: 0.4, transparent: true, opacity: 0.9 });
    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 8), this.jointMat, 25);
    this.joints.frustumCulled = false;
    this.boneMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BONES.length * 6), 3));
    this.bones = new THREE.LineSegments(geo, this.boneMat);
    this.bones.frustumCulled = false;
    this.controller = this.makeController();
    this.group.add(this.joints, this.bones, this.controller);
    this.group.visible = false; // until the first tracked sample
  }

  private makeController(): THREE.Group {
    const g = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.018, 0.07, 6, 16),
      new THREE.MeshStandardMaterial({ color: 0x1b2233, roughness: 0.35, metalness: 0.3 }),
    );
    body.rotation.x = Math.PI / 2 - 0.5;
    body.position.set(0, -0.01, 0.03);
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.045, 0.005, 8, 40),
      new THREE.MeshStandardMaterial({ color: this.color, emissive: this.color, emissiveIntensity: 0.8 }),
    );
    ring.rotation.x = -0.4;
    ring.position.set(0, 0.02, -0.035);
    ring.name = "ring";
    g.add(body, ring);
    return g;
  }

  update(h: HandSample, radii: Float32Array, engaged: boolean) {
    const isHand = h.source === SRC_HAND, isCtrl = h.source === SRC_CONTROLLER;
    this.group.visible = isHand || isCtrl;
    this.joints.visible = this.bones.visible = isHand;
    this.controller.visible = isCtrl;
    this.jointMat.emissiveIntensity = engaged ? 1.4 : 0.35;
    this.boneMat.opacity = engaged ? 1.0 : 0.55;
    if (isHand) {
      const j = h.joints;
      for (let i = 0; i < 25; i++) {
        const r = Math.max(radii[i] || 0.008, 0.004) * 0.85;
        this.p.set(j[i * 7], j[i * 7 + 1], j[i * 7 + 2]);
        this.s.setScalar(r);
        this.m.compose(this.p, this.q.identity(), this.s);
        this.joints.setMatrixAt(i, this.m);
      }
      this.joints.instanceMatrix.needsUpdate = true;
      const pos = this.bones.geometry.getAttribute("position") as THREE.BufferAttribute;
      BONES.forEach(([a, b], k) => {
        pos.setXYZ(2 * k, j[a * 7], j[a * 7 + 1], j[a * 7 + 2]);
        pos.setXYZ(2 * k + 1, j[b * 7], j[b * 7 + 1], j[b * 7 + 2]);
      });
      pos.needsUpdate = true;
    } else if (isCtrl) {
      const p = h.pose;
      this.controller.position.set(p[0], p[1], p[2]);
      this.controller.quaternion.set(p[3], p[4], p[5], p[6]);
      const ring = this.controller.getObjectByName("ring") as THREE.Mesh;
      (ring.material as THREE.MeshStandardMaterial).emissiveIntensity = engaged ? 2.0 : 0.5;
    }
  }
}

export class Pointer {
  readonly group = new THREE.Group();
  private line: THREE.Mesh;
  private dot: THREE.Mesh;

  constructor(color: THREE.Color) {
    const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55, depthWrite: false });
    this.line = new THREE.Mesh(new THREE.CylinderGeometry(0.0012, 0.0025, 1, 6, 1, true).translate(0, 0.5, 0).rotateX(-Math.PI / 2), mat);
    this.dot = new THREE.Mesh(new THREE.RingGeometry(0.004, 0.007, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }));
    this.dot.renderOrder = 20;
    this.group.add(this.line, this.dot);
    this.group.visible = false;
  }

  /** Show a ray from origin along quat's -Z, ending at `hit` (or a short stub). */
  set(origin: THREE.Vector3, quat: THREE.Quaternion, hit: THREE.Vector3 | null, hitNormalQuat?: THREE.Quaternion) {
    this.group.visible = true;
    const len = hit ? origin.distanceTo(hit) : 0.25;
    this.line.position.copy(origin);
    this.line.quaternion.copy(quat);
    this.line.scale.set(1, 1, len);
    (this.line.material as THREE.MeshBasicMaterial).opacity = hit ? 0.8 : 0.25;
    this.dot.visible = !!hit;
    if (hit) {
      this.dot.position.copy(hit);
      if (hitNormalQuat) this.dot.quaternion.copy(hitNormalQuat);
    }
  }

  hide() { this.group.visible = false; }
}
