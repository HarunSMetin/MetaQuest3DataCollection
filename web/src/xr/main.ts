// QuestCollect WebXR client: renders the MuJoCo mirror, streams head/hands/controllers, in-VR UI.

import * as THREE from "three";
import "./xr.css";
import { Link } from "../net";
import { levelStates } from "../goal";
import {
  B_SECONDARY, B_STICK_X, B_STICK_Y, B_TRIGGER, fetchTask, InputPacker, type ServerEvent, SRC_CONTROLLER, SRC_HAND, SRC_NONE,
  type Status, type TaskSpec,
} from "../protocol";
import { ARM_COLORS, MujocoScene } from "../MujocoScene";
import { type Side, XRInputSampler } from "./input";
import { CamPreviewPanel } from "./CamPreviewPanel";
import { GoalPanel } from "./GoalPanel";
import { Hud } from "./Hud";
import { PanelLayout } from "./Layout";
import { Sounds } from "./sounds";
import { type PanelId, WristMenu } from "./WristMenu";
import { HandViz, Pointer } from "./visuals";

// ---------------------------------------------------------------------------------------------
// Renderer + scene

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.xr.enabled = true;
renderer.xr.setReferenceSpaceType("local-floor");
renderer.xr.setFoveation(0.4);
document.getElementById("stage")!.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const BG = new THREE.Color("#060912");
scene.background = BG;
scene.fog = new THREE.Fog(BG, 5, 16);

const sky = new THREE.Mesh(
  new THREE.SphereGeometry(30, 32, 16),
  new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: { top: { value: new THREE.Color("#0b1226") }, mid: { value: new THREE.Color("#141c38") }, bottom: { value: new THREE.Color("#04060c") } },
    vertexShader: "varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }",
    fragmentShader: `uniform vec3 top; uniform vec3 mid; uniform vec3 bottom; varying vec3 vP;
      void main(){ float h = vP.y; vec3 c = h > 0.0 ? mix(mid, top, smoothstep(0.0, 0.6, h)) : mix(mid, bottom, smoothstep(0.0, 0.25, -h));
      gl_FragColor = vec4(c, 1.0); }`,
  }),
);
scene.add(sky);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.02, 60);
camera.position.set(0.9, 1.75, 1.0);

const mj = new MujocoScene({ shadows: true });
scene.add(mj.root);

// World panels live in the MuJoCo world frame (so they follow the table on recenter). These are their default
// poses; the operator can move them (layout).
const layout = new PanelLayout();

// HUD above the far table edge, facing the operator.
const hud = new Hud();
mj.root.add(hud.panel.mesh);
layout.add("hud", hud.panel, () => new THREE.Matrix4()
  .makeBasis(new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0))
  .multiply(new THREE.Matrix4().makeRotationX(-0.18))
  .setPosition(1.02, 0, 0.75 + 0.6));

// Goal panel beside the table on the operator's left, low and near, facing the operator's head (where the
// arms don't reach in front of it). MuJoCo frame: the operator stands at x = -0.5.
const OPERATOR_HEAD = new THREE.Vector3(-0.5, 0, 1.6);
function facingOperator(at: THREE.Vector3) {
  return new THREE.Matrix4().lookAt(OPERATOR_HEAD, at, new THREE.Vector3(0, 0, 1)).setPosition(at);  // +z (the face) to the head
}
const goal = new GoalPanel();
mj.root.add(goal.panel.mesh);
layout.add("goal", goal.panel, () => facingOperator(new THREE.Vector3(0.3, 0.92, 0.75 + 0.3)));

// Camera previews: the mirror image on the right (menu: Previews), so neither covers the work area.
const previews = new CamPreviewPanel();
mj.root.add(previews.panel.mesh);
layout.add("previews", previews.panel, () => facingOperator(new THREE.Vector3(0.3, -0.92, 0.75 + 0.42)));

const sounds = new Sounds();

const sampler = new XRInputSampler();
const handViz: Record<Side, HandViz> = { left: new HandViz(ARM_COLORS.l, "l"), right: new HandViz(ARM_COLORS.r, "r") };
const HINTS_KEY = "qc.hints";
function setHints(on: boolean) {
  for (const s of ["left", "right"] as Side[]) handViz[s].showHints = on;
  try { localStorage.setItem(HINTS_KEY, on ? "on" : "off"); } catch { /* storage unavailable */ }
}
try { if (localStorage.getItem(HINTS_KEY) === "off") setHints(false); } catch { /* storage unavailable */ }
// Persisted view toggles (wrist menu).
function stored(key: string, def: boolean): boolean {
  try { const v = localStorage.getItem(key); return v === null ? def : v === "on"; } catch { return def; }
}
function store(key: string, on: boolean) {
  try { localStorage.setItem(key, on ? "on" : "off"); } catch { /* storage unavailable */ }
}
const CAMVIZ_KEY = "qc.camviz", PREVIEW_KEY = "qc.preview", HUD_KEY = "qc.hud", GOAL_KEY = "qc.goal";
let camViz = false;
function setCamViz(on: boolean) { camViz = on; mj.setCameraViz(on); store(CAMVIZ_KEY, on); }
function setPreview(on: boolean) { previews.setVisible(on); store(PREVIEW_KEY, on); }
setCamViz(stored(CAMVIZ_KEY, true));
setPreview(stored(PREVIEW_KEY, false));
const panelMeshes: Record<PanelId, THREE.Object3D> = { hud: hud.panel.mesh, goal: goal.panel.mesh, previews: previews.panel.mesh };
function setPanel(id: PanelId, on: boolean) {
  if (id === "previews") { setPreview(on); return; }
  panelMeshes[id].visible = on;
  store(id === "hud" ? HUD_KEY : GOAL_KEY, on);
}
setPanel("hud", stored(HUD_KEY, true));
setPanel("goal", stored(GOAL_KEY, true));
const pointers: Record<Side, Pointer> = { left: new Pointer(ARM_COLORS.l), right: new Pointer(ARM_COLORS.r) };
for (const s of ["left", "right"] as Side[]) scene.add(handViz[s].group, pointers[s].group);

// ---------------------------------------------------------------------------------------------
// Networking

let status: Status | null = null;
let task: TaskSpec | null = null;
let lastServerT = 0;

/** Data-collection guide: which brick comes next and which arm should take it (the one on its side of the
 *  table, or the one holding it). The brick glows; that arm's status chip names it. Follows the hints toggle. */
function updateGuide(s: Status) {
  const k = task && handViz.left.showHints ? levelStates(task, s.progress, s.in_area).indexOf("next") : -1;
  const b = task && k >= 0 ? task.bricks[task.goal[k].brick] : null;
  const holder = b ? (["l", "r"] as const).find((a) => s.held[a] === b.name) : undefined;
  const body = b ? mj.root.getObjectByName(b.name) : undefined;
  const arm = holder ?? (body ? (body.position.y > 0 ? "l" : "r") : undefined);  // +y: the operator's left
  mj.highlightBody(b && !holder ? b.name : null);
  handViz.left.next = b && arm === "l" && !holder ? b.label : null;
  handViz.right.next = b && arm === "r" && !holder ? b.label : null;
}
const packer = new InputPacker();

const link = new Link("/ws/xr", {
  onPose: (f) => {
    mj.pushFrame(f, performance.now());
    lastServerT = f.tServer;
  },
  onStatus: (s) => {
    status = s;
    mj.setPlacement(s.placement);
    mj.setOutOfReach(s.out_of_reach);
    updateGuide(s);
    hud.update(s);
    goal.update(s);
    menu.update(s);
    handViz.left.setStatus(s);
    handViz.right.setStatus(s);
    updateLanding();
  },
  onEvent: (e: ServerEvent) => {
    if (e.type === "haptic") {
      const side = e.arm === "l" ? "left" : "right";
      for (let i = 0; i < (e.repeat ?? 1); i++) {
        setTimeout(() => sampler.pulse(side, e.intensity, e.ms), i * (e.ms + (e.gap_ms ?? 0)));
      }
    }
    else if (e.type === "toast") { hud.showToast(e.text, e.kind); toastDom(e.text, e.kind); }
    else if (e.type === "sound") sounds.play(e.name, e.arm);
  },
  // (Re)load the scene on every connect, so a restarted server with a changed scene is picked up.
  onOpen: () => { mj.load(); fetchTask().then((t) => { task = t; goal.setTask(t); }).catch(() => {}); updateLanding(); },
  onClose: () => { status = null; handViz.left.setStatus(null); handViz.right.setStatus(null); updateLanding(); },
});

// ---------------------------------------------------------------------------------------------
// Wrist menu + interaction

const menu = new WristMenu({
  command: (name, args) => link.command(name, args),
  exit: () => renderer.xr.getSession()?.end(),
  toggleSound: () => sounds.toggle(),
  soundOn: () => sounds.enabled,
  toggleHints: () => setHints(!handViz.left.showHints),
  hintsOn: () => handViz.left.showHints,
  toggleCamViz: () => setCamViz(!camViz),
  camVizOn: () => camViz,
  togglePanel: (id) => setPanel(id, !panelMeshes[id].visible),
  panelOn: (id) => panelMeshes[id].visible,
  previewLayout: () => previews.layout,
  setPreviewLayout: (l) => previews.setLayout(l),
  toggleCam: (name) => previews.toggleCam(name),
  camOn: (name) => previews.camOn(name),
  toggleLock: () => layout.toggleLock(),
  locked: () => layout.locked,
  resetLayout: () => layout.reset(),
});
scene.add(menu.panel.mesh);

const raycaster = new THREE.Raycaster();
const tmpV = new THREE.Vector3(), tmpV2 = new THREE.Vector3(), tmpV3 = new THREE.Vector3();
const tmpQ = new THREE.Quaternion(), tmpQ2 = new THREE.Quaternion();
const headPos = new THREE.Vector3();
const UP_OFFSET = new THREE.Vector3(0, 0.13, 0);
const selecting: Record<Side, boolean> = { left: false, right: false };
const sentGrip: Record<Side, { trigger: number; pinch: number }> = { left: { trigger: 0, pinch: 0.1 }, right: { trigger: 0, pinch: 0.1 } };
let lastFrameMs = 0;
let prevMenuBtn = 0;
let menuToggled = false;
let pokePrevZ = 1;
let palmShown = false;

function placeMenu() {
  const left = sampler.hands.left;
  headPos.set(sampler.head[0], sampler.head[1], sampler.head[2]);
  let anchor: THREE.Vector3 | null = null;

  if (left.source === SRC_CONTROLLER) {
    const btn = left.buttons[B_SECONDARY];
    if (btn > 0.5 && prevMenuBtn <= 0.5) menuToggled = !menuToggled;
    prevMenuBtn = btn;
    if (menuToggled) anchor = tmpV.set(left.pose[0], left.pose[1] + 0.16, left.pose[2]);
  } else if (left.source === SRC_HAND) {
    // Palm normal = -Y of the wrist joint; show the menu while the palm faces the head.
    const j = left.joints;
    tmpQ.set(j[3], j[4], j[5], j[6]);
    const normal = tmpV2.set(0, -1, 0).applyQuaternion(tmpQ);
    const palm = tmpV.set(j[70], j[71], j[72]); // middle-finger metacarpal
    const facing = normal.dot(tmpV3.copy(headPos).sub(palm).normalize());
    palmShown = palmShown ? facing > 0.3 : facing > 0.6;
    if (palmShown) anchor = palm.addScaledVector(normal, 0.05).add(UP_OFFSET);
  }

  if (anchor) {
    menu.panel.mesh.position.lerp(anchor, menu.visible ? 0.35 : 1);
    menu.panel.mesh.lookAt(headPos);
    if (!menu.visible) menu.setVisible(true);
  } else if (menu.visible) menu.setVisible(false);
}

/** Select = Trigger (controllers) or thumb-index pinch (hands), with hysteresis. */
function selectDown(side: Side): boolean {
  const h = sampler.hands[side], was = selecting[side];
  selecting[side] = h.source === SRC_CONTROLLER ? h.buttons[B_TRIGGER] > (was ? 0.5 : 0.7)
    : h.source === SRC_HAND ? h.pinch[0] < (was ? 0.03 : 0.015) : false;
  return selecting[side];
}

function interact(dt: number) {
  const uiCapture: Record<Side, boolean> = { left: false, right: false };
  let hovered: string | null = null;
  const panelHover = new Map<THREE.Object3D, THREE.Color>();

  for (const side of ["left", "right"] as Side[]) {
    const h = sampler.hands[side];
    const ray = sampler.rays[side];
    const wasDown = selecting[side], down = selectDown(side);
    pointers[side].hide();
    if (!ray || h.source === SRC_NONE) { layout.end(side); continue; }
    // Panels can only be grabbed with a hand whose arm is released, so moving one never moves the robot.
    const armFree = !mj.engaged[side === "left" ? "l" : "r"];
    const origin = tmpV.fromArray(ray.origin);
    const q = tmpQ.set(...ray.quat);
    const dir = tmpV2.set(0, 0, -1).applyQuaternion(q);

    if (layout.dragging(side)) {
      if (down && armFree) {
        layout.drag(side, origin, dir, headPos, h.buttons[B_STICK_X], h.buttons[B_STICK_Y], dt);
        pointers[side].set(origin, q, layout.dragPoint(side));
        uiCapture[side] = true;
        continue;
      }
      layout.end(side);
    }

    const menuRay = menu.visible && side === "right" && h.source === SRC_CONTROLLER;
    const targets = armFree ? layout.targets() : [];
    if (menuRay) targets.push(menu.panel.mesh);
    if (!targets.length) continue;
    raycaster.set(origin, dir);
    const hit = raycaster.intersectObjects(targets, false)[0];
    if (hit && hit.object === menu.panel.mesh) {
      uiCapture[side] = true;
      const local = menu.panel.mesh.worldToLocal(hit.point.clone());
      const w = menu.panel.widgetAt(local);
      hovered = w?.id ?? null;
      pointers[side].set(origin, q, hit.point, menu.panel.mesh.quaternion);
      if (w && down && !wasDown && menu.panel.press(w)) sampler.pulse(side, 0.4, 25);
    } else if (hit) {
      uiCapture[side] = true;
      panelHover.set(hit.object, side === "left" ? ARM_COLORS.l : ARM_COLORS.r);
      pointers[side].set(origin, q, hit.point, hit.object.getWorldQuaternion(tmpQ2));
      if (down && !wasDown) {
        layout.begin(side, hit.object, origin, hit.point);
        sampler.pulse(side, 0.5, 30);
      }
    } else if (menuRay) {
      pointers[side].set(origin, q, null);
    }
  }
  layout.highlight(panelHover);

  // Poke with the right index fingertip (hand tracking).
  const right = sampler.hands.right;
  if (menu.visible && right.source === SRC_HAND) {
    const local = menu.panel.mesh.worldToLocal(tmpV.set(right.joints[63], right.joints[64], right.joints[65])); // index tip (joint 9)
    const inside = Math.abs(local.x) < menu.panel.widthM / 2 && Math.abs(local.y) < menu.panel.heightM / 2;
    if (inside && local.z < 0.04 && local.z > -0.03) {
      const w = menu.panel.widgetAt(local);
      hovered = w?.id ?? null;
      if (w && pokePrevZ > 0.008 && local.z <= 0.008) menu.panel.press(w);
    }
    pokePrevZ = inside ? local.z : 1;
  }
  menu.panel.setHover(hovered);

  // Pointing at UI must not move the gripper (holding what was sent before, so a held brick isn't dropped)
  // or send the stick that is moving a panel.
  for (const side of ["left", "right"] as Side[]) {
    const h = sampler.hands[side], last = sentGrip[side];
    if (uiCapture[side]) {
      h.buttons[B_TRIGGER] = last.trigger;
      h.pinch[0] = last.pinch;
      h.buttons[B_STICK_X] = h.buttons[B_STICK_Y] = 0;
    }
    last.trigger = h.buttons[B_TRIGGER];
    last.pinch = h.pinch[0];
  }
}

// ---------------------------------------------------------------------------------------------
// Render loop

renderer.setAnimationLoop((_t, frame?: XRFrame) => {
  const now = performance.now();
  if (frame) {
    const ref = renderer.xr.getReferenceSpace()!;
    sampler.sample(frame, ref);
    placeMenu();
    interact(Math.min((now - (lastFrameMs || now)) / 1000, 0.1));
    lastFrameMs = now;
    link.sendBinary(packer.pack(sampler.head, sampler.hands.left, sampler.hands.right, lastServerT));
    tmpQ.fromArray(sampler.head, 3);
    handViz.left.update(sampler.hands.left, sampler.radii.left, mj.engaged.l, sampler.sources.left, tmpQ);
    handViz.right.update(sampler.hands.right, sampler.radii.right, mj.engaged.r, sampler.sources.right, tmpQ);
  } else {
    // Landing-page preview: slow orbit around the robots.
    const a = now / 9000;
    camera.position.set(0.0 + Math.sin(a) * 1.3, 1.65, -0.9 + Math.cos(a) * 1.3);
    camera.lookAt(0, 0.95, -0.85);
  }
  mj.update(now);
  hud.tick();
  goal.tick();
  previews.tick();
  if (menu.visible) menu.panel.redraw();
  renderer.render(scene, camera);
});

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------------------------------------------------------------------------------------------
// Landing page (2D, before entering XR)

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const FOG = scene.fog;

function setPassthrough(on: boolean) {
  scene.background = on ? null : BG;
  scene.fog = on ? null : FOG;
  sky.visible = mj.environment.visible = !on;
}

async function enterXR(mode: "vr" | "ar") {
  if (!navigator.xr) return;
  sounds.unlock(); // audio may only start from this click
  const session = await navigator.xr.requestSession(mode === "vr" ? "immersive-vr" : "immersive-ar", {
    requiredFeatures: ["local-floor"],
    optionalFeatures: ["hand-tracking"],
  });
  setPassthrough(mode === "ar");
  await renderer.xr.setSession(session);
  document.body.classList.add("in-xr");
  session.addEventListener("end", () => {
    document.body.classList.remove("in-xr");
    setPassthrough(false);
    menu.setVisible(false);
    for (const s of ["left", "right"] as Side[]) { pointers[s].hide(); handViz[s].group.visible = false; layout.end(s); }
  });
  // Align the virtual workcell with where the operator is standing.
  setTimeout(() => link.command("recenter"), 1200);
}

async function initLanding() {
  const vrBtn = $<HTMLButtonElement>("enter-vr"), arBtn = $<HTMLButtonElement>("enter-ar");
  const vr = (await navigator.xr?.isSessionSupported("immersive-vr").catch(() => false)) ?? false;
  const ar = (await navigator.xr?.isSessionSupported("immersive-ar").catch(() => false)) ?? false;
  vrBtn.disabled = !vr;
  arBtn.disabled = !ar;
  if (!vr) {
    $("xr-hint").innerHTML = window.isSecureContext
      ? "WebXR not available in this browser. Open this page in the <b>Quest Browser</b>."
      : `Not a secure context. Use <b>http://localhost</b> over USB (<code>qc setup-quest</code>) or <b>https://</b> over Wi-Fi (<code>qc certs</code>).`;
  }
  vrBtn.onclick = () => enterXR("vr");
  arBtn.onclick = () => enterXR("ar");
  for (const el of document.querySelectorAll<HTMLButtonElement>("[data-mode]")) {
    el.onclick = () => link.command("set_mode", { mode: el.dataset.mode });
  }
}

function updateLanding() {
  const s = status;
  $("conn").className = `pill ${link.open ? (s?.connected ? "ok" : "warn") : "bad"}`;
  $("conn").textContent = link.open ? (s ? "Server connected" : "Connecting…") : "Server offline";
  if (!s) return;
  for (const el of document.querySelectorAll<HTMLButtonElement>("[data-mode]")) el.classList.toggle("active", el.dataset.mode === s.mode);
  $("instruction").textContent = s.instruction;
  $("saved").textContent = String(s.saved_count);
}

function toastDom(text: string, kind: string) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = text;
  $("toasts").appendChild(el);
  setTimeout(() => el.classList.add("out"), 2200);
  setTimeout(() => el.remove(), 2700);
}

initLanding();
updateLanding();

// Debug handle for automated tests and the browser console.
(window as unknown as { __qc: unknown }).__qc = { sampler, mj, link, renderer, hud, goal, previews, menu, sounds, layout };
