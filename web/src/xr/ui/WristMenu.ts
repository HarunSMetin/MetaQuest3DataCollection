// Wrist-attached control menu (left hand / left controller).

import type { Status } from "../../protocol";
import { button, glass, Panel, text, THEME } from "./Panel";

const W = 0.24, H = 0.4;
const PX = 2400; // high density: viewed up close (canvas 576 x 960)

export interface MenuActions {
  command: (name: string, args?: Record<string, unknown>) => void;
  exit: () => void;
}

export class WristMenu {
  readonly panel: Panel;
  status: Status | null = null;
  visible = false;

  constructor(private act: MenuActions) {
    this.panel = new Panel(W, H, PX, (g, w, h) => {
      glass(g, 4, 4, w - 8, h - 8, 48);
      const s = this.status;
      text(g, "QUESTCOLLECT", 40, 52, 30, { weight: 800, color: THEME.accent });
      text(g, s ? `${s.saved_count} saved` : "", w - 40, 52, 26, { align: "right", color: THEME.dim, weight: 600 });
      text(g, "INPUT MODE", 40, 112, 22, { color: THEME.dim, weight: 700 });
      text(g, "TASK", 40, 272, 22, { color: THEME.dim, weight: 700 });
      text(g, "EPISODE", 40, 432, 22, { color: THEME.dim, weight: 700 });
      text(g, "VIEW", 40, 718, 22, { color: THEME.dim, weight: 700 });
      if (s) text(g, `scale ×${s.scale.toFixed(2)}`, w - 40, 718, 22, { align: "right", color: THEME.dim, weight: 700 });
    });
    this.build();
    this.panel.mesh.visible = false;
  }

  private build() {
    const s = () => this.status;
    const cmd = this.act.command;
    const CW = this.panel.canvas.width;
    const m = 36, gap = 14;
    const col2 = (CW - 2 * m - gap) / 2;
    const col3 = (CW - 2 * m - 2 * gap) / 3;
    const c3 = (i: number) => m + i * (col3 + gap);
    const rec = () => s()?.rec_state ?? "idle";
    const idle = () => rec() === "idle" && !s()?.replay;
    const canSave = () => rec() === "review" || rec() === "recording";

    this.panel.widgets = [
      // input mode
      button("mode-ctrl", m, 134, col2, 110, () => "Controllers", () => cmd("set_mode", { mode: "controllers" }),
        { icon: () => "🎮", active: () => s()?.mode === "controllers" }),
      button("mode-hands", m + col2 + gap, 134, col2, 110, () => "Hands", () => cmd("set_mode", { mode: "hands" }),
        { icon: () => "✋", active: () => s()?.mode === "hands" }),

      // task picker
      button("task-prev", m, 294, 100, 110, () => "‹", () => this.cycleTask(-1), { enabled: idle }),
      {
        id: "task-label", x: m + 112, y: 294, w: CW - 2 * m - 224, h: 110,
        draw: (g) => text(g, (s()?.task ?? "—").replace(/_/g, " "), CW / 2, 349, 32, { align: "center", weight: 800 }),
      },
      button("task-next", CW - m - 100, 294, 100, 110, () => "›", () => this.cycleTask(1), { enabled: idle }),

      // recording
      button("rec", m, 454, CW - 2 * m, 120,
        () => (rec() === "recording" ? "Stop" : rec() === "countdown" ? "Cancel" : "Record"),
        () => cmd("record_toggle"),
        {
          icon: () => (rec() === "recording" ? "■" : "●"), color: () => THEME.rec,
          active: () => rec() === "recording" || rec() === "countdown",
          enabled: () => rec() !== "review" && !s()?.replay,
        }),
      button("save-ok", c3(0), 590, col3, 96, () => "Success", () => cmd("save", { success: true }),
        { icon: () => "✓", color: () => THEME.ok, enabled: canSave }),
      button("save-fail", c3(1), 590, col3, 96, () => "Failure", () => cmd("save", { success: false }),
        { icon: () => "✗", color: () => THEME.warn, enabled: canSave }),
      button("discard", c3(2), 590, col3, 96, () => "Discard", () => cmd("discard"),
        { icon: () => "🗑", color: () => THEME.rec, enabled: () => rec() !== "idle" }),

      // view
      button("recenter", c3(0), 740, col3, 96, () => "Recenter", () => cmd("recenter"), { icon: () => "⟲" }),
      button("scale-down", c3(1), 740, col3, 96, () => "Slower", () => cmd("set_scale", { scale: (s()?.scale ?? 1) - 0.25 }),
        { icon: () => "−" }),
      button("scale-up", c3(2), 740, col3, 96, () => "Faster", () => cmd("set_scale", { scale: (s()?.scale ?? 1) + 0.25 }),
        { icon: () => "+" }),

      button("reset", m, 850, col2, 80, () => "↺  New scene", () => cmd("reset"), { enabled: idle }),
      button("exit", m + col2 + gap, 850, col2, 80, () => "⏏  Exit XR", () => this.act.exit()),
    ];
  }

  private cycleTask(dir: number) {
    const s = this.status;
    if (!s || s.rec_state !== "idle") return;
    const i = s.tasks.indexOf(s.task);
    this.act.command("set_task", { task: s.tasks[(i + dir + s.tasks.length) % s.tasks.length] });
  }

  update(status: Status | null) {
    this.status = status;
    this.panel.invalidate();
  }

  setVisible(v: boolean) {
    this.visible = v;
    this.panel.mesh.visible = v;
    if (!v) this.panel.setHover(null);
  }
}
