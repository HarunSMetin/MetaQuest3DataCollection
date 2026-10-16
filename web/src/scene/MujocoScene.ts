// Rebuilds the MuJoCo scene in three.js from /api/scene and animates it from the pose stream.
// Shared by the WebXR app and the desktop dashboard.

import * as THREE from "three";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { FLAG_ENGAGED_L, FLAG_ENGAGED_R, type PoseFrame } from "../protocol";

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
export interface SceneDesc {
  version: number;
  nbody: number;
  bodies: { id: number; name: string; parent: number }[];
  geoms: GeomDesc[];
  meshes: MeshDesc[];
}

interface BufferedFrame { t: number; bodies: Float32Array; ghosts: Float32Array; grippers: Float32Array; flags: number }

const INTERP_DELAY_MS = 30;
const CREASE_ANGLE = THREE.MathUtils.degToRad(35);

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

function makeGhost(color: THREE.Color): THREE.Group {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.6, transparent: true, opacity: 0.35, depthWrite: false });
  const palm = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.17, 0.02), mat);
  palm.position.z = -0.06;
  const f1 = new THREE.Mesh(new THREE.BoxGeometry(0.016, 0.01, 0.05), mat);
  const f2 = f1.clone();
  f1.name = "f1"; f2.name = "f2";
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.008, 12, 8), mat);
  g.add(palm, f1, f2, tip);
  g.userData.mat = mat;
  return g;
}

export class MujocoScene {
  readonly root = new THREE.Group();          // world -> xr placement applied here
  readonly environment = new THREE.Group();   // floor + backdrop (hidden in passthrough)
  readonly ghosts: [THREE.Group, THREE.Group];
  desc: SceneDesc | null = null;
  engaged = { l: false, r: false };
  private bodies: THREE.Object3D[] = [];
  private frames: BufferedFrame[] = [];
  private clockOffset = Number.POSITIVE_INFINITY;
  private materials = new Map<string, THREE.Material>();
  private geometries: THREE.BufferGeometry[] = [];
  private tmpQa = new THREE.Quaternion();
  private tmpQb = new THREE.Quaternion();

  constructor(private opts: { shadows: boolean }) {
    this.root.matrixAutoUpdate = false;
    this.ghosts = [makeGhost(ARM_COLORS.l), makeGhost(ARM_COLORS.r)];
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
      if (isFloor) this.environment.add(mesh);
      else this.bodies[g.body].add(mesh);
    }
    this.bodies[0].add(this.environment);
    for (const gh of this.ghosts) this.root.add(gh);
    this.frames = [];
  }

  private clear() {
    for (const b of this.bodies) this.root.remove(b);
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
    this.frames.push({ t: f.tServer + this.clockOffset, bodies: f.bodies.slice(), ghosts: f.ghosts.slice(), grippers: f.grippers.slice(), flags: f.flags });
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
    this.ghosts.forEach((gh, i) => {
      const k = i * 7, G = latest.ghosts;
      gh.position.set(G[k], G[k + 1], G[k + 2]);
      gh.quaternion.set(G[k + 3], G[k + 4], G[k + 5], G[k + 6]);
      const w = 0.01 + 0.08 * latest.grippers[i];
      gh.getObjectByName("f1")!.position.set(0, w / 2 + 0.005, -0.025);
      gh.getObjectByName("f2")!.position.set(0, -w / 2 - 0.005, -0.025);
      const engaged = i === 0 ? this.engaged.l : this.engaged.r;
      const mat = gh.userData.mat as THREE.MeshStandardMaterial;
      mat.opacity = engaged ? 0.55 : 0.18;
      mat.emissiveIntensity = engaged ? 1.2 : 0.4;
    });
  }
}
