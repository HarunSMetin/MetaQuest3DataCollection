// Wrist-attached control menu (left hand / left controller). Two pages: controls, and layout (camera preview
// grid, which panels show, moving panels).

import type { Status } from "../protocol";
import { PREVIEW_CAMS, type PreviewLayout } from "./CamPreviewPanel";
import { button, glass, Panel, text, THEME, type Widget, wrapText } from "./Panel";

const W = 0.24, H = 0.415;
const PX = 2400; // high density: viewed up close (canvas 576 x 996)

export type PanelId = "hud" | "goal" | "previews";

export interface MenuActions {
  command: (name: string, args?: Record<string, unknown>) => void;
  exit: () => void;
  toggleSound: () => void;
  soundOn: () => boolean;
  toggleHints: () => void;
  hintsOn: () => boolean;
  toggleCamViz: () => void;
  camVizOn: () => boolean;
  togglePanel: (id: PanelId) => void;
  panelOn: (id: PanelId) => boolean;
  previewLayout: () => PreviewLayout;
  setPreviewLayout: (l: PreviewLayout) => void;
  toggleCam: (name: string) => void;
  camOn: (name: string) => boolean;
  toggleLock: () => void;
  locked: () => boolean;
  resetLayout: () => void;
}

export class WristMenu {
  readonly panel: Panel;
  status: Status | null = null;
  visible = false;
  private page: "main" | "layout" = "main";
  private pages: Record<"main" | "layout", Widget[]>;

  constructor(private act: MenuActions) {
    this.panel = new Panel(W, H, PX, (g, w, h) => {
      glass(g, 4, 4, w - 8, h - 8, 48);
      const s = this.status;
      text(g, this.page === "main" ? "QUESTCOLLECT" : "LAYOUT", 40, 52, 30, { weight: 800, color: THEME.accent });
      if (this.page === "main") {
        text(g, "INPUT MODE", 40, 112, 22, { color: THEME.dim, weight: 700 });
        text(g, "EPISODE", 40, 272, 22, { color: THEME.dim, weight: 700 });
        if (s) text(g, `${s.saved_count} saved`, w - 40, 272, 22, { align: "right", color: THEME.dim, weight: 700 });
        text(g, "VIEW", 40, 558, 22, { color: THEME.dim, weight: 700 });
        if (s) text(g, `scale ×${s.scale.toFixed(2)}`, w - 40, 558, 22, { align: "right", color: THEME.dim, weight: 700 });
      } else {
        text(g, "CAMERA PREVIEWS", 40, 112, 22, { color: THEME.dim, weight: 700 });
        text(g, "PANELS", 40, 448, 22, { color: THEME.dim, weight: 700 });
        wrapText(g, this.act.locked()
          ? "Panels are locked. Unlock them to move them."
          : "Move a panel: release the arm (Grip / middle pinch), point at the panel and hold Trigger (hands: pinch "
            + "thumb + index). While holding, the stick moves it nearer/farther (up/down) and resizes it (left/right).",
        40, 680, w - 80, 34, 24, THEME.dim, 8);
      }
    });
    this.pages = { main: this.buildMain(), layout: this.buildLayout() };
    this.panel.widgets = this.pages.main;
    this.panel.mesh.visible = false;
  }

  private pageButton(): Widget {
    const CW = this.panel.canvas.width;
    return button("page", CW - 36 - 190, 20, 190, 64, () => (this.page === "main" ? "⚙  Layout" : "←  Back"), () => {
      this.page = this.page === "main" ? "layout" : "main";
      this.panel.widgets = this.pages[this.page];
      this.panel.setHover(null);
      this.panel.invalidate();
    });
  }

  private buildMain(): Widget[] {
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

    return [
      this.pageButton(),
      // input mode
      button("mode-ctrl", m, 134, col2, 110, () => "Controllers", () => cmd("set_mode", { mode: "controllers" }),
        { icon: () => "🎮", active: () => s()?.mode === "controllers" }),
      button("mode-hands", m + col2 + gap, 134, col2, 110, () => "Hands", () => cmd("set_mode", { mode: "hands" }),
        { icon: () => "✋", active: () => s()?.mode === "hands" }),

      // recording
      // During a replay: continue it from the shown frame as a new take (the operator's correction).
      button("rec", m, 294, CW - 2 * m, 120,
        () => (rec() === "recording" ? "Stop" : rec() === "countdown" ? "Cancel" : s()?.replay ? "Take over here" : "Record"),
        () => cmd(s()?.replay && rec() === "idle" ? "replay_branch" : "record_toggle"),
        {
          icon: () => (rec() === "recording" ? "■" : s()?.replay && rec() === "idle" ? "✋" : "●"), color: () => THEME.rec,
          active: () => rec() === "recording" || rec() === "countdown",
          enabled: () => rec() !== "review",
        }),
      button("save-ok", c3(0), 430, col3, 96, () => "Success", () => cmd("save", { success: true }),
        { icon: () => "✓", color: () => THEME.ok, enabled: canSave }),
      button("save-fail", c3(1), 430, col3, 96, () => "Failure", () => cmd("save", { success: false }),
        { icon: () => "✗", color: () => THEME.warn, enabled: canSave }),
      button("discard", c3(2), 430, col3, 96, () => "Discard", () => cmd("discard"),
        { icon: () => "🗑", color: () => THEME.rec, enabled: () => rec() !== "idle" }),

      // view
      button("recenter", c3(0), 580, col3, 96, () => "Recenter", () => cmd("recenter"), { icon: () => "⟲" }),
      button("scale-down", c3(1), 580, col3, 96, () => "Slower", () => cmd("set_scale", { scale: (s()?.scale ?? 1) - 0.25 }),
        { icon: () => "−" }),
      button("scale-up", c3(2), 580, col3, 96, () => "Faster", () => cmd("set_scale", { scale: (s()?.scale ?? 1) + 0.25 }),
        { icon: () => "+" }),

      button("reset", m, 690, col2, 80, () => "↺  New scene", () => cmd("reset"), { enabled: idle }),
      button("exit", m + col2 + gap, 690, col2, 80, () => "⏏  Exit XR", () => this.act.exit()),
      button("sound", m, 786, col2, 80, () => (this.act.soundOn() ? "🔊  Sound" : "🔇  Muted"),
        () => { this.act.toggleSound(); this.panel.invalidate(); }, { active: () => this.act.soundOn() }),
      button("hints", m + col2 + gap, 786, col2, 80, () => "🏷  Button hints",
        () => { this.act.toggleHints(); this.panel.invalidate(); }, { active: () => this.act.hintsOn() }),
      button("camviz", m, 882, col2, 80, () => "📷  Camera view",
        () => { this.act.toggleCamViz(); this.panel.invalidate(); }, { active: () => this.act.camVizOn() }),
      button("preview", m + col2 + gap, 882, col2, 80, () => "🖼  Previews",
        () => { this.act.togglePanel("previews"); this.panel.invalidate(); }, { active: () => this.act.panelOn("previews") }),
    ];
  }

  private buildLayout(): Widget[] {
    const a = this.act;
    const CW = this.panel.canvas.width;
    const m = 36, gap = 14;
    const col2 = (CW - 2 * m - gap) / 2;
    const col3 = (CW - 2 * m - 2 * gap) / 3;
    const c2 = (i: number) => m + i * (col2 + gap);
    const c3 = (i: number) => m + i * (col3 + gap);
    const redraw = (f: () => void) => () => { f(); this.panel.invalidate(); };
    const previewsOn = () => a.panelOn("previews");

    const layouts: [PreviewLayout, string, string][] = [["column", "Vertical", "↕"], ["row", "Horizontal", "↔"], ["grid", "Grid", "⊞"]];
    const panels: [PanelId, string][] = [["hud", "Status"], ["goal", "Goal"], ["previews", "Previews"]];
    return [
      this.pageButton(),
      ...layouts.map(([l, label, icon], i) => button(`layout-${l}`, c3(i), 134, col3, 96, () => label,
        redraw(() => { a.setPreviewLayout(l); if (!previewsOn()) a.togglePanel("previews"); }),
        { icon: () => icon, active: () => a.previewLayout() === l })),
      ...PREVIEW_CAMS.map((c, i) => button(`cam-${c.name}`, c2(i % 2), 246 + Math.floor(i / 2) * 88, col2, 76,
        () => c.short, redraw(() => a.toggleCam(c.name)), { color: () => c.color, active: () => a.camOn(c.name) })),

      ...panels.map(([id, label], i) => button(`panel-${id}`, c3(i), 470, col3, 80, () => label,
        redraw(() => a.togglePanel(id)), { active: () => a.panelOn(id) })),
      button("lock", c2(0), 566, col2, 80, () => (a.locked() ? "🔒  Locked" : "🔓  Movable"),
        redraw(() => a.toggleLock()), { active: () => a.locked(), color: () => THEME.warn }),
      button("reset-layout", c2(1), 566, col2, 80, () => "↺  Reset layout", redraw(() => a.resetLayout())),
    ];
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
