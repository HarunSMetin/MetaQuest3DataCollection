// Movable world panels (HUD, goal, camera previews): point at one with a hand whose arm is not engaged, hold
// Trigger (or pinch thumb+index) and carry it; it keeps facing the head. While held, the thumbstick pushes/pulls
// (up/down) and resizes (left/right). Poses are stored in the parent (MuJoCo world) frame, so they follow the
// table on recenter, and persist in localStorage.

import * as THREE from "three";
import type { Side } from "./input";
import type { Panel } from "./Panel";

const STORE_KEY = "qc.layout", LOCK_KEY = "qc.layout.locked";
const MIN_DIST = 0.15, MAX_DIST = 4;
const MIN_SCALE = 0.35, MAX_SCALE = 3;
const PUSH_SPEED = 1.2;  // m/s at full stick
const GROW_SPEED = 0.9;  // log-scale/s at full stick
const DEADZONE = 0.2;
const FRAME_PAD = 0.008;

interface Item {
  id: string;
  panel: Panel;
  def: () => THREE.Matrix4;  // default pose in the parent frame
  frame: THREE.Mesh;         // hover/drag outline behind the panel
  handle: THREE.Mesh;        // grab bar under the panel
  frameSize: [number, number];
}

interface Drag {
  item: Item;
  dist: number;
  local: THREE.Vector3;  // grabbed point relative to the panel centre, panel axes, times the panel's own scale
  scale: number;
  point: THREE.Vector3;  // grabbed point in world, for the pointer
}

function roundedRect(w: number, h: number, r: number) {
  const s = new THREE.Shape();
  s.moveTo(-w / 2 + r, -h / 2);
  s.lineTo(w / 2 - r, -h / 2);
  s.quadraticCurveTo(w / 2, -h / 2, w / 2, -h / 2 + r);
  s.lineTo(w / 2, h / 2 - r);
  s.quadraticCurveTo(w / 2, h / 2, w / 2 - r, h / 2);
  s.lineTo(-w / 2 + r, h / 2);
  s.quadraticCurveTo(-w / 2, h / 2, -w / 2, h / 2 - r);
  s.lineTo(-w / 2, -h / 2 + r);
  s.quadraticCurveTo(-w / 2, -h / 2, -w / 2 + r, -h / 2);
  return new THREE.ShapeGeometry(s, 6);
}

const dz = (v: number) => (Math.abs(v) < DEADZONE ? 0 : (v - Math.sign(v) * DEADZONE) / (1 - DEADZONE));

export class PanelLayout {
  locked = false;
  private items: Item[] = [];
  private saved: Record<string, number[]> = {};
  private drags: Record<Side, Drag | null> = { left: null, right: null };
  private m = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private v = new THREE.Vector3();

  constructor() {
    try {
      this.saved = JSON.parse(localStorage.getItem(STORE_KEY) ?? "{}");
      this.locked = localStorage.getItem(LOCK_KEY) === "on";
    } catch { /* storage unavailable or corrupt: defaults */ }
  }

  add(id: string, panel: Panel, def: () => THREE.Matrix4) {
    const mat = (opacity: number) =>
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity, depthWrite: false, toneMapped: false });
    const frame = new THREE.Mesh(new THREE.BufferGeometry(), mat(0.5));
    frame.renderOrder = 9;
    frame.position.z = -0.001;
    const handle = new THREE.Mesh(roundedRect(0.09, 0.012, 0.006), mat(0.7));
    handle.renderOrder = 11;
    frame.visible = handle.visible = false;
    panel.mesh.add(frame, handle);
    panel.mesh.matrixAutoUpdate = false;
    const item: Item = { id, panel, def, frame, handle, frameSize: [0, 0] };
    this.items.push(item);
    this.apply(item);
  }

  private apply(item: Item) {
    const s = this.saved[item.id];
    if (s?.length === 16) item.panel.mesh.matrix.fromArray(s);
    else item.panel.mesh.matrix.copy(item.def());
    item.panel.mesh.matrixWorldNeedsUpdate = true;
  }

  private persist() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(this.saved));
      localStorage.setItem(LOCK_KEY, this.locked ? "on" : "off");
    } catch { /* storage unavailable */ }
  }

  reset() {
    this.drags.left = this.drags.right = null;
    this.saved = {};
    for (const it of this.items) this.apply(it);
    this.persist();
  }

  toggleLock() {
    this.locked = !this.locked;
    if (this.locked) this.drags.left = this.drags.right = null;
    this.persist();
  }

  /** Visible panel meshes that can be grabbed. */
  targets(): THREE.Object3D[] {
    return this.locked ? [] : this.items.filter((it) => it.panel.mesh.visible).map((it) => it.panel.mesh);
  }

  isTarget(obj: THREE.Object3D) { return this.items.some((it) => it.panel.mesh === obj); }

  dragging(side: Side) { return this.drags[side] !== null; }

  dragPoint(side: Side) { return this.drags[side]?.point ?? null; }

  begin(side: Side, obj: THREE.Object3D, origin: THREE.Vector3, hit: THREE.Vector3) {
    const item = this.items.find((it) => it.panel.mesh === obj);
    if (!item || this.locked) return;
    const other = side === "left" ? "right" : "left";
    if (this.drags[other]?.item === item) return;  // the other hand already holds it
    const mesh = item.panel.mesh;
    mesh.updateWorldMatrix(true, false);
    this.drags[side] = {
      item,
      dist: origin.distanceTo(hit),
      local: mesh.worldToLocal(hit.clone()).multiplyScalar(this.v.setFromMatrixScale(mesh.matrix).x),
      scale: this.v.setFromMatrixScale(mesh.matrix).x,
      point: hit.clone(),
    };
  }

  /** Move the held panel: grabbed point at `dist` along the ray, panel facing `head` and upright. */
  drag(side: Side, origin: THREE.Vector3, dir: THREE.Vector3, head: THREE.Vector3, stickX: number, stickY: number, dt: number) {
    const d = this.drags[side];
    if (!d) return;
    const mesh = d.item.panel.mesh, parent = mesh.parent;
    if (!parent || !mesh.visible || this.locked) { this.end(side); return; }
    d.dist = THREE.MathUtils.clamp(d.dist - dz(stickY) * PUSH_SPEED * dt, MIN_DIST, MAX_DIST);
    const grow = Math.exp(dz(stickX) * GROW_SPEED * dt);
    const prevScale = d.scale;
    d.scale = THREE.MathUtils.clamp(d.scale * grow, MIN_SCALE, MAX_SCALE);
    d.local.multiplyScalar(d.scale / prevScale);  // grow about the grabbed point
    d.point.copy(origin).addScaledVector(dir, d.dist);

    parent.updateWorldMatrix(true, false);
    const parentScale = this.v.setFromMatrixScale(parent.matrixWorld).x;
    this.m.lookAt(head, d.point, THREE.Object3D.DEFAULT_UP);  // +z (the face) toward the head
    this.q.setFromRotationMatrix(this.m);
    const pos = this.v.copy(d.local).multiplyScalar(parentScale).applyQuaternion(this.q).negate().add(d.point);
    this.m.compose(pos, this.q, new THREE.Vector3().setScalar(d.scale * parentScale));
    mesh.matrix.copy(parent.matrixWorld).invert().multiply(this.m);
    mesh.matrixWorldNeedsUpdate = true;
  }

  end(side: Side) {
    const d = this.drags[side];
    if (!d) return;
    this.drags[side] = null;
    this.saved[d.item.id] = d.item.panel.mesh.matrix.toArray();
    this.persist();
  }

  /** Outline panels that are hovered (side colour) or held (brighter, with the grab bar). */
  highlight(hover: Map<THREE.Object3D, THREE.Color>) {
    for (const it of this.items) {
      const held = this.drags.left?.item === it ? "left" : this.drags.right?.item === it ? "right" : null;
      const color = hover.get(it.panel.mesh);
      const on = !!(held || color) && it.panel.mesh.visible;
      it.frame.visible = it.handle.visible = on;
      if (!on) continue;
      const { widthM: w, heightM: h } = it.panel;
      if (it.frameSize[0] !== w || it.frameSize[1] !== h) {  // the preview panel resizes with its grid
        it.frame.geometry.dispose();
        it.frame.geometry = roundedRect(w + 2 * FRAME_PAD, h + 2 * FRAME_PAD, 0.02);
        it.frameSize = [w, h];
      }
      it.handle.position.set(0, -h / 2 - 0.022, 0);
      const fm = it.frame.material as THREE.MeshBasicMaterial, hm = it.handle.material as THREE.MeshBasicMaterial;
      if (color) { fm.color.copy(color); hm.color.copy(color); }
      fm.opacity = held ? 0.85 : 0.45;
      hm.opacity = held ? 1 : 0.6;
    }
  }
}
