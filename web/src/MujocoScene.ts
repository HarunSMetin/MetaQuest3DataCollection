// Rebuilds the MuJoCo scene in three.js from /api/scene and animates it from the pose stream.
// Shared by the WebXR app and the desktop dashboard.

import * as THREE from "three";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { FLAG_ENGAGED_L, FLAG_ENGAGED_R, type PoseFrame } from "./protocol";

export const ARM_COLORS = { l: new THREE.Color("#22d3ee"), r: new THREE.Color("#f472b6") };

interface GeomDesc {
  name: string;
  body: number;
  type: "plane" | "sphere" | "capsule" | "ellipsoid" | "cylinder" | "box" | "mesh";
  size: number[];
  pos: number[];
  quat: number[];
  rgba: number[];
  material: { specular: number; shininess: number; reflectance: number };
  mesh: number | null;
}
interface MeshDesc { name: string; vertOffset: number; nvert: number; faceOffset: number; nface: number }
/** A MuJoCo camera: looks along its -z with +y up (like three.js), vertical field of view in degrees. */
interface CameraDesc { name: string; body: number; pos: number[]; quat: number[]; fovy: number }
export interface SceneDesc {
  version: number;
  nbody: number;
  bodies: { id: number; name: string; parent: number }[];
  geoms: GeomDesc[];
  meshes: MeshDesc[];
  cameras?: CameraDesc[];
}

interface BufferedFrame { t: number; bodies: Float32Array; flags: number }

/** Grasp-point marker (the server's `<arm>_tcp_marker` geom) plus a drop line to the table under it. */
interface TcpMarker { arm: "l" | "r"; body: THREE.Object3D; sphere: THREE.Mesh; line: THREE.Line; dot: THREE.Mesh }

/** A wrist camera's view: its frustum (on the hand) and the patch of table it sees (outline + faint fill). */
interface CamViz { body: THREE.Object3D; cam: THREE.Object3D; frustum: THREE.LineSegments; outline: THREE.LineLoop; fill: THREE.Mesh }

const INTERP_DELAY_MS = 30;
const CREASE_ANGLE = THREE.MathUtils.degToRad(35);
const OUT_OF_REACH = new THREE.Color("#ef4444");
const CAM_ASPECT = 4 / 3;      // the recorded images are 320 x 240
const FRUSTUM_LEN = 0.18;      // m drawn in front of the camera
const MAX_RAY = 1.5;           // m: a corner ray that doesn't reach the table within this hides the footprint

function gridTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 512;
  const g = c.getContext("2d")!;
  g.fillStyle = "#0d1117";
  g.fillRect(0, 0, 512, 512);
  g.strokeStyle = "rgba(120,160,255,0.10)";
  g.lineWidth = 2;
  for (let i = 0; i <= 512; i += 64) {
    g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 512); g.stroke();
    g.beginPath(); g.moveTo(0, i); g.lineTo(512, i); g.stroke();
  }
  g.strokeStyle = "rgba(120,160,255,0.22)";
  g.lineWidth = 3;
  g.strokeRect(0, 0, 512, 512);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

export class MujocoScene {
  readonly root = new THREE.Group();          // world -> xr placement applied here
  readonly environment = new THREE.Group();   // floor + backdrop (hidden in passthrough)
  desc: SceneDesc | null = null;
  engaged = { l: false, r: false };
  private outOfReach = { l: false, r: false };
  private bodies: THREE.Object3D[] = [];
  private markers: TcpMarker[] = [];
  private camViz: CamViz[] = [];
  private camVizOn = false;
  private highlighted: { name: string; meshes: THREE.Mesh[]; originals: THREE.Material[]; mat: THREE.MeshStandardMaterial } | null = null;
  private tableZ = 0;
  private frames: BufferedFrame[] = [];
  private clockOffset = Number.POSITIVE_INFINITY;
  private materials = new Map<string, THREE.Material>();
  private geometries: THREE.BufferGeometry[] = [];
  private tmpQa = new THREE.Quaternion();
  private tmpQb = new THREE.Quaternion();
  private tmpV = new THREE.Vector3();

  constructor(private opts: { shadows: boolean }) {
    this.root.matrixAutoUpdate = false;
    this.addLights();
  }

  private addLights() {
    const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x20232b, 1.4);
    hemi.position.set(0, 0, 3);
    hemi.up.set(0, 0, 1);
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(0.8, 0.6, 3.0);
    key.target.position.set(0.4, 0, 0.75);
    if (this.opts.shadows) {
      key.castShadow = true;
      key.shadow.mapSize.set(1024, 1024);
      const cam = key.shadow.camera as THREE.OrthographicCamera;
      cam.left = -1.0; cam.right = 1.0; cam.top = 1.0; cam.bottom = -1.0; cam.near = 0.5; cam.far = 4;
      key.shadow.bias = -0.0005;
      key.shadow.normalBias = 0.01;
    }
    const rim = new THREE.DirectionalLight(0x8fb4ff, 0.8);
    rim.position.set(2.0, -1.5, 1.8);
    this.root.add(hemi, key, key.target, rim);
  }

  async load() {
    const [desc, blob] = await Promise.all([
      fetch("/api/scene").then((r) => r.json() as Promise<SceneDesc>),
      fetch("/api/scene.bin").then((r) => r.arrayBuffer()),
    ]);
    this.build(desc, blob);
  }

  private material(g: GeomDesc): THREE.Material {
    const key = `${g.rgba.join(",")}|${g.material.shininess}|${g.material.specular}`;
    let m = this.materials.get(key);
    if (!m) {
      const [r, gg, b, a] = g.rgba;
      m = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setRGB(r, gg, b, THREE.SRGBColorSpace),
        roughness: THREE.MathUtils.clamp(0.85 - g.material.shininess * 0.6 - g.material.specular * 0.2, 0.15, 0.95),
        metalness: 0.05,
        transparent: a < 1,
        opacity: a,
      });
      this.materials.set(key, m);
    }
    return m;
  }

  private build(desc: SceneDesc, blob: ArrayBuffer) {
    this.clear();
    this.desc = desc;
    const meshGeoms: THREE.BufferGeometry[] = desc.meshes.map((m) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(blob, m.vertOffset, m.nvert * 3), 3));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(blob, m.faceOffset, m.nface * 3), 1));
      const creased = toCreasedNormals(g, CREASE_ANGLE);
      g.dispose();
      this.geometries.push(creased);
      return creased;
    });

    for (let b = 0; b < desc.nbody; b++) {
      const o = new THREE.Object3D();
      o.name = desc.bodies[b].name;
      o.matrixAutoUpdate = true;
      this.bodies.push(o);
      this.root.add(o);
    }

    for (const g of desc.geoms) {
      let geo: THREE.BufferGeometry;
      const s = g.size;
      let mat = this.material(g);
      switch (g.type) {
        case "box": geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
        case "sphere": geo = new THREE.SphereGeometry(s[0], 24, 16); break;
        case "ellipsoid": geo = new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]); break;
        case "capsule": geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2); break;
        case "cylinder": geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2); break;
        case "plane": {
          const w = s[0] > 0 ? 2 * s[0] : 20, h = s[1] > 0 ? 2 * s[1] : 20;
          geo = new THREE.PlaneGeometry(w, h);
          if (g.name === "floor") {
            const tex = gridTexture();
            tex.repeat.set(w / 0.5, h / 0.5);
            mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9, metalness: 0 });
          }
          break;
        }
        case "mesh": geo = meshGeoms[g.mesh!]; break;
      }
      if (g.type !== "mesh") this.geometries.push(geo);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = g.name;
      mesh.position.fromArray(g.pos);
      mesh.quaternion.fromArray(g.quat);
      const isFloor = g.name === "floor";
      mesh.castShadow = this.opts.shadows && !isFloor && g.type !== "plane";
      mesh.receiveShadow = this.opts.shadows;
      if (g.name === "table_top") this.tableZ = g.pos[2] + s[2];
      if (isFloor) this.environment.add(mesh);
      else this.bodies[g.body].add(mesh);
      if (g.name.endsWith("_tcp_marker")) this.addMarker(g.name.startsWith("l_") ? "l" : "r", mesh);
    }
    this.bodies[0].add(this.environment);
    for (const c of desc.cameras ?? []) {
      if (c.name.includes("wrist")) this.addCamViz(c, desc.bodies[c.body].name.startsWith("l_") ? ARM_COLORS.l : ARM_COLORS.r);
    }
    this.frames = [];
    const want = this.highlighted?.name ?? null;  // the old meshes are gone: highlight the same body again
    this.highlighted?.mat.dispose();
    this.highlighted = null;
    this.highlightBody(want);
  }

  private addCamViz(c: CameraDesc, color: THREE.Color) {
    const cam = new THREE.Object3D();
    cam.position.fromArray(c.pos);
    cam.quaternion.fromArray(c.quat);
    const hh = Math.tan(THREE.MathUtils.degToRad(c.fovy) / 2), hw = hh * CAM_ASPECT;
    cam.userData.corners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => new THREE.Vector3(x, y, -1));
    const L = FRUSTUM_LEN, k = (x: number, y: number) => [x * L, y * L, -L];
    const [a, b, cc, d] = [k(-hw, -hh), k(hw, -hh), k(hw, hh), k(-hw, hh)];
    const pts = [[0, 0, 0], a, [0, 0, 0], b, [0, 0, 0], cc, [0, 0, 0], d, a, b, b, cc, cc, d, d, a].flat();
    const frustumGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    const frustum = new THREE.LineSegments(frustumGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }));
    cam.add(frustum);
    const outlineGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
    const outline = new THREE.LineLoop(outlineGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }));
    const fillGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
    fillGeo.setIndex([0, 1, 2, 0, 2, 3]);
    const fill = new THREE.Mesh(fillGeo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }));
    for (const o of [outline, fill]) { o.frustumCulled = false; this.root.add(o); }
    this.geometries.push(frustumGeo, outlineGeo, fillGeo);
    const body = this.bodies[c.body];
    body.add(cam);
    this.camViz.push({ body, cam, frustum, outline, fill });
    this.setCameraViz(this.camVizOn);
  }

  /** Show where the wrist cameras look: frustum on the hand and the patch of table in view. */
  setCameraViz(on: boolean) {
    this.camVizOn = on;
    for (const v of this.camViz) v.frustum.visible = v.outline.visible = v.fill.visible = on;
  }

  /** Make one body (e.g. the brick to take next) glow; null clears it. */
  highlightBody(name: string | null) {
    if (this.highlighted?.name === name) return;
    if (this.highlighted) {
      this.highlighted.meshes.forEach((m, i) => (m.material = this.highlighted!.originals[i]));
      this.highlighted.mat.dispose();
      this.highlighted = null;
    }
    const body = name ? this.bodies.find((b) => b.name === name) : undefined;
    const meshes = (body?.children.filter((o) => o instanceof THREE.Mesh) ?? []) as THREE.Mesh[];
    if (!name || !meshes.length) {
      if (name) this.highlighted = { name, meshes: [], originals: [], mat: new THREE.MeshStandardMaterial() };  // retried after a reload
      return;
    }
    const mat = (meshes[0].material as THREE.MeshStandardMaterial).clone();
    mat.emissive.copy(mat.color);
    this.highlighted = { name, meshes, originals: meshes.map((m) => m.material as THREE.Material), mat };
    for (const m of meshes) m.material = mat;
  }

  private updateCamViz() {
    const p = this.tmpV, q = this.tmpQa, d = new THREE.Vector3(), z = this.tableZ + 0.001;
    for (const v of this.camViz) {
      if (!this.camVizOn) continue;
      p.copy(v.cam.position).applyQuaternion(v.body.quaternion).add(v.body.position);  // bodies are children of root
      q.copy(v.body.quaternion).multiply(v.cam.quaternion);
      const hits: THREE.Vector3[] = [];
      for (const c of v.cam.userData.corners as THREE.Vector3[]) {
        d.copy(c).applyQuaternion(q);
        const t = d.z < -1e-6 ? (z - p.z) / d.z : Infinity;
        if (t * d.length() > MAX_RAY) break;
        hits.push(p.clone().addScaledVector(d, t));
      }
      const show = hits.length === 4;
      v.outline.visible = v.fill.visible = show;
      if (!show) continue;
      for (const o of [v.outline, v.fill]) {
        const pos = o.geometry.getAttribute("position") as THREE.BufferAttribute;
        hits.forEach((h, i) => pos.setXYZ(i, h.x, h.y, h.z));
        pos.needsUpdate = true;
      }
    }
  }

  private addMarker(arm: "l" | "r", sphere: THREE.Mesh) {
    const color = ARM_COLORS[arm];
    // Drawn on top of the fingers so the grasp point stays visible between them.
    sphere.material = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true });
    sphere.renderOrder = 10;
    sphere.castShadow = sphere.receiveShadow = false;
    const lineGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 }));
    line.frustumCulled = false;
    const dot = new THREE.Mesh(new THREE.RingGeometry(0.004, 0.007, 24), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 }));
    this.geometries.push(lineGeo, dot.geometry);
    this.root.add(line, dot);
    this.markers.push({ arm, body: sphere.parent!, sphere, line, dot });
  }

  /** Markers turn red while the IK can't get that gripper to its target. */
  setOutOfReach(outOfReach: { l: boolean; r: boolean }) {
    this.outOfReach = { ...outOfReach };
  }

  private clear() {
    for (const b of this.bodies) this.root.remove(b);
    for (const v of this.camViz) {
      this.root.remove(v.outline, v.fill);
      for (const o of [v.frustum, v.outline, v.fill]) (o.material as THREE.Material).dispose();
    }
    this.camViz = [];
    for (const mk of this.markers) {
      this.root.remove(mk.line, mk.dot);
      for (const o of [mk.sphere, mk.line, mk.dot]) (o.material as THREE.Material).dispose();
    }
    this.markers = [];
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials.values()) m.dispose();
    this.bodies = [];
    this.geometries = [];
    this.materials.clear();
    this.environment.clear();
  }

  setPlacement(colMajor: number[]) {
    this.root.matrix.fromArray(colMajor);
    this.root.matrixWorldNeedsUpdate = true;
  }

  /** Buffer a frame; timestamps are mapped onto the local clock for jitter-free interpolation. */
  pushFrame(f: PoseFrame, recvMs: number) {
    if (!this.desc || f.nbody !== this.desc.nbody) return;
    this.clockOffset = Math.min(this.clockOffset + 0.05, recvMs - f.tServer);
    this.frames.push({ t: f.tServer + this.clockOffset, bodies: f.bodies.slice(), flags: f.flags });
    if (this.frames.length > 8) this.frames.shift();
  }

  update(nowMs: number) {
    const n = this.frames.length;
    if (!n) return;
    const t = nowMs - INTERP_DELAY_MS;
    let a = this.frames[n - 1], b = a, alpha = 0;
    for (let i = n - 1; i > 0; i--) {
      if (this.frames[i - 1].t <= t) {
        a = this.frames[i - 1];
        b = this.frames[i];
        alpha = THREE.MathUtils.clamp((t - a.t) / Math.max(b.t - a.t, 1e-3), 0, 1);
        break;
      }
    }
    if (t < this.frames[0].t) { a = b = this.frames[0]; alpha = 0; }

    for (let i = 1; i < this.bodies.length; i++) {
      const o = this.bodies[i], k = i * 7, A = a.bodies, B = b.bodies;
      o.position.set(A[k] + (B[k] - A[k]) * alpha, A[k + 1] + (B[k + 1] - A[k + 1]) * alpha, A[k + 2] + (B[k + 2] - A[k + 2]) * alpha);
      this.tmpQa.set(A[k + 3], A[k + 4], A[k + 5], A[k + 6]);
      this.tmpQb.set(B[k + 3], B[k + 4], B[k + 5], B[k + 6]);
      o.quaternion.slerpQuaternions(this.tmpQa, this.tmpQb, alpha);
    }

    const latest = this.frames[n - 1];
    this.engaged = { l: !!(latest.flags & FLAG_ENGAGED_L), r: !!(latest.flags & FLAG_ENGAGED_R) };
    if (this.highlighted?.meshes.length) this.highlighted.mat.emissiveIntensity = 0.25 + 0.2 * Math.sin(nowMs / 220);
    this.updateCamViz();
    for (const mk of this.markers) {
      const color = this.outOfReach[mk.arm] ? OUT_OF_REACH : ARM_COLORS[mk.arm];
      const engaged = this.engaged[mk.arm];
      for (const o of [mk.sphere, mk.line, mk.dot]) (o.material as THREE.MeshBasicMaterial).color.copy(color);
      (mk.sphere.material as THREE.MeshBasicMaterial).opacity = engaged ? 1 : 0.6;
      // Grasp point in world coordinates (bodies are direct children of root).
      const p = this.tmpV.copy(mk.sphere.position).applyQuaternion(mk.body.quaternion).add(mk.body.position);
      const z = this.tableZ + 0.0005, above = p.z > z + 0.003;
      mk.line.visible = mk.dot.visible = above;
      if (!above) continue;
      const pos = mk.line.geometry.getAttribute("position") as THREE.BufferAttribute;
      pos.setXYZ(0, p.x, p.y, p.z);
      pos.setXYZ(1, p.x, p.y, z);
      pos.needsUpdate = true;
      mk.dot.position.set(p.x, p.y, z);
    }
  }
}
