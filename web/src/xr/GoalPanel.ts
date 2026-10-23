// World-anchored panel next to the HUD that shows the mission's goal (drawn from /api/task) and progress.

import { drawGoal, levelStates } from "../goal";
import type { Status, TaskSpec } from "../protocol";
import { glass, Panel, text, THEME } from "./Panel";

const W = 0.32, H = 0.42;

export class GoalPanel {
  readonly panel: Panel;
  private task: TaskSpec | null = null;
  private status: Status | null = null;

  constructor() {
    this.panel = new Panel(W, H, 1400, (g, w, h) => this.draw(g, w, h));
  }

  setTask(task: TaskSpec) {
    this.task = task;
    this.panel.invalidate();
  }

  update(status: Status | null) {
    const changed = status?.progress !== this.status?.progress || status?.in_area !== this.status?.in_area;
    this.status = status;
    if (changed) this.panel.invalidate();
  }

  private draw(g: CanvasRenderingContext2D, w: number, h: number) {
    const pad = 32, t = this.task, s = this.status;
    glass(g, 6, 6, w - 12, h - 12, 40, THEME.accent);
    text(g, "GOAL", pad, pad + 20, 34, { weight: 800, color: THEME.accent });
    if (!t) {
      text(g, "Loading…", w / 2, h / 2, 30, { align: "center", color: THEME.dim });
      return;
    }
    const states = levelStates(t, s?.progress ?? 0, s?.in_area ?? false);
    const done = states.filter((x) => x === "placed").length;
    text(g, `${done} / ${t.goal.length}`, w - pad, pad + 20, 34, { weight: 800, align: "right", color: done === t.goal.length ? THEME.ok : THEME.text });

    const listH = 44 * t.goal.length + 16;
    drawGoal(g, t, states, { x: pad, y: pad + 50, w: w - 2 * pad, h: h - 2 * pad - 50 - listH }, 1);

    // One row per brick, in build order (top of the list = bottom of the tower).
    let y = h - pad - listH + 26;
    t.goal.forEach((e, k) => {
      const b = t.bricks[e.brick], st = states[k];
      const [r, gg, bb] = b.rgba.map((v) => Math.round(v * 255));
      g.fillStyle = `rgb(${r},${gg},${bb})`;
      g.globalAlpha = st === "todo" ? 0.4 : 1;
      g.fillRect(pad, y - 13, 26, 26);
      g.globalAlpha = 1;
      const what = k === 0 ? `${b.label}  in taped area` : b.label;
      const color = st === "placed" ? THEME.ok : st === "next" ? THEME.text : THEME.dim;
      text(g, `${k + 1}  ${what}`, pad + 40, y, 27, { weight: st === "next" ? 800 : 600, color });
      text(g, st === "placed" ? "✓" : st === "next" ? "◀" : "", w - pad, y, 26, { align: "right", weight: 800, color });
      y += 44;
    });
  }

  /** Redraws only after a change (each redraw re-uploads the texture). */
  tick() {
    this.panel.redraw();
  }
}
