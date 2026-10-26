// Draws the mission's goal structure (from TaskSpec) as a small 3D-looking picture on a 2D canvas.
// Shared by the headset's goal panel and the dashboard. Nothing here knows the tower's shape: it draws
// whatever bricks and placements the server describes, so a new mission shows up without client changes.

import type { TaskSpec } from "./protocol";

export type LevelState = "placed" | "next" | "todo";

/** Per goal entry: placed (counted by the server), the next one to place, or still to do. */
export function levelStates(task: TaskSpec, progress: number, inArea: boolean): LevelState[] {
  const placed = task.goal.map((_, k) => (k === 0 ? inArea : inArea && k <= progress));
  const next = placed.indexOf(false);
  return placed.map((p, k) => (p ? "placed" : k === next ? "next" : "todo"));
}

/** Bricks placed so far (the base counts once it is in the area), the same count as levelStates. */
export function bricksPlaced(progress: number, inArea: boolean): number {
  return inArea ? 1 + progress : 0;
}

type V3 = [number, number, number];
const EL = 0.6, AZ = -0.65; // view: from above (34°) and from the operator's right
const D: V3 = [Math.cos(EL) * Math.cos(AZ), Math.cos(EL) * Math.sin(AZ), -Math.sin(EL)]; // view direction
const R: V3 = norm([D[1], -D[0], 0]); // screen right = d x z
const U: V3 = cross(R, D); // screen up

function norm(v: V3): V3 { const n = Math.hypot(...v); return [v[0] / n, v[1] / n, v[2] / n]; }
function cross(a: V3, b: V3): V3 { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function shade(rgba: number[], k: number, alpha: number) {
  const c = (i: number) => Math.round(255 * Math.min(1, rgba[i] * k));
  return `rgba(${c(0)},${c(1)},${c(2)},${alpha})`;
}

interface Placed { k: number; size: V3; studs: [number, number]; rgba: number[]; pos: V3; yaw: number }

/** Draw the goal into the box (x, y, w, h) of a canvas context. */
export function drawGoal(g: CanvasRenderingContext2D, task: TaskSpec, states: LevelState[],
  box: { x: number; y: number; w: number; h: number }, pulse = 1) {
  const items: Placed[] = task.goal.map((e, k) => {
    const b = task.bricks[e.brick];
    return { k, size: b.size, studs: b.studs, rgba: b.rgba, pos: e.pos, yaw: e.yaw };
  });
  const world = (it: Placed, lx: number, ly: number, lz: number): V3 => {
    const c = Math.cos(it.yaw), s = Math.sin(it.yaw);
    return [it.pos[0] + c * lx - s * ly, it.pos[1] + s * lx + c * ly, it.pos[2] + lz];
  };
  const corners = (it: Placed, z: number): V3[] => {
    const [sx, sy] = [it.size[0] / 2, it.size[1] / 2];
    return [[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]].map(([a, b]) => world(it, a, b, z));
  };

  // Fit the structure into the box. The build area is drawn as a taped square hugging the base (the real
  // one is larger; fitting it would shrink the bricks to nothing).
  const base = items[0], m = 0.35 * Math.max(base.size[0], base.size[1]);
  const [ax, ay] = [Math.min(task.area.half[0], base.size[0] / 2 + m), Math.min(task.area.half[1], base.size[1] / 2 + m)];
  const floor: V3[] = [[-ax, -ay, 0], [ax, -ay, 0], [ax, ay, 0], [-ax, ay, 0]].map(([x, y]) => world(base, x, y, 0));
  const pts = [...floor, ...items.flatMap((it) => [...corners(it, 0), ...corners(it, it.size[2] + task.stud_h)])];
  const sx = pts.map((p) => dot(p, R)), sy = pts.map((p) => -dot(p, U));
  const minX = Math.min(...sx), maxX = Math.max(...sx), minY = Math.min(...sy), maxY = Math.max(...sy);
  const scale = Math.min(box.w / (maxX - minX), box.h / (maxY - minY)) * 0.92;
  const ox = box.x + box.w / 2 - ((minX + maxX) / 2) * scale, oy = box.y + box.h / 2 - ((minY + maxY) / 2) * scale;
  const P = (p: V3): [number, number] => [ox + dot(p, R) * scale, oy - dot(p, U) * scale];
  const poly = (ps: V3[]) => { g.beginPath(); ps.forEach((p, i) => (i ? g.lineTo(...P(p)) : g.moveTo(...P(p)))); g.closePath(); };

  // The build area: the base brick has to lie inside this taped square.
  poly(floor);
  g.fillStyle = "rgba(235,235,225,0.06)";
  g.fill();
  g.setLineDash([10, 8]);
  g.strokeStyle = "rgba(235,235,225,0.55)";
  g.lineWidth = 3;
  g.stroke();
  g.setLineDash([]);

  // Painter's order: lower bricks first, then farther ones.
  const order = items.slice().sort((a, b) => a.pos[2] - b.pos[2] || dot(b.pos, D) - dot(a.pos, D));
  for (const it of order) {
    const st = states[it.k];
    const alpha = st === "todo" ? 0.25 : 1;
    const h = it.size[2];
    const bot = corners(it, 0), top = corners(it, h);
    // Visible side faces (outward normal facing the viewer).
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      const mid = [0, 1, 2].map((a) => (bot[i][a] + bot[j][a]) / 2 - it.pos[a]) as V3;
      mid[2] = 0;
      if (dot(mid, D) >= 0) continue;
      poly([bot[i], bot[j], top[j], top[i]]);
      g.fillStyle = shade(it.rgba, Math.abs(mid[0]) > Math.abs(mid[1]) ? 0.72 : 0.58, alpha);
      g.fill();
    }
    poly(top);
    g.fillStyle = shade(it.rgba, 1.0, alpha);
    g.fill();
    // Studs, far to near.
    const [nx, ny] = it.studs, p = task.pitch, r = task.stud_r * scale, ry = r * Math.sin(EL);
    const studs: V3[] = [];
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) studs.push(world(it, (i - (nx - 1) / 2) * p, (j - (ny - 1) / 2) * p, h));
    studs.sort((a, b) => dot(b, D) - dot(a, D));
    for (const c of studs) {
      const [bx, by] = P(c), [tx, ty] = P([c[0], c[1], c[2] + task.stud_h]);
      g.fillStyle = shade(it.rgba, 0.7, alpha);
      g.beginPath(); g.ellipse(bx, by, r, ry, 0, 0, Math.PI * 2); g.fill();
      g.fillRect(bx - r, ty, 2 * r, by - ty);
      g.fillStyle = shade(it.rgba, 1.12, alpha);
      g.beginPath(); g.ellipse(tx, ty, r, ry, 0, 0, Math.PI * 2); g.fill();
    }
    // Outline: dashed for to-do bricks, glowing for the next one.
    const outline = () => {
      g.beginPath();
      for (const q of [bot, top]) {
        q.forEach((v, i) => (i ? g.lineTo(...P(v)) : g.moveTo(...P(v))));
        g.closePath();
      }
      for (let i = 0; i < 4; i++) { g.moveTo(...P(bot[i])); g.lineTo(...P(top[i])); }
    };
    if (st === "todo") {
      g.setLineDash([6, 6]);
      g.strokeStyle = shade(it.rgba, 1.2, 0.6);
      g.lineWidth = 2;
      outline();
      g.stroke();
      g.setLineDash([]);
    } else if (st === "next") {
      g.save();
      g.shadowColor = "#ffffff";
      g.shadowBlur = 10 + 10 * pulse;
      g.strokeStyle = `rgba(255,255,255,${0.6 + 0.4 * pulse})`;
      g.lineWidth = 3;
      poly(top);
      g.stroke();
      g.restore();
    }
  }
}
