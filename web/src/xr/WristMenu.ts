// Wrist-attached control menu (left hand / left controller). Three pages: controls; replay (pick a dataset and
// episode, scrub, take over, save: offline DAgger without the dashboard); layout (camera preview grid, which panels
// show, moving panels).

import type { Status } from "../protocol";
import { PREVIEW_CAMS, type PreviewLayout } from "./CamPreviewPanel";
import { button, glass, Panel, roundRect, text, THEME, type Widget, wrapText } from "./Panel";
import { ReplayBrowser } from "./ReplayBrowser";

const W = 0.24, H = 0.415;
const PX = 2400; // high density: viewed up close (canvas 576 x 996)

export type PanelId = "hud" | "goal" | "previews";
type Page = "main" | "replay" | "layout";
const SPEEDS = [0.25, 0.5, 1, 2];

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
  private page: Page = "main";
  private pages: Record<Page, Widget[]>;
  readonly browser = new ReplayBrowser(() => this.panel.invalidate());
  private takeOver: { of: string; step: number; saved: number } | null = null; // the take-over being recorded

  constructor(private act: MenuActions) {
    this.panel = new Panel(W, H, PX, (g, w, h) => {
      glass(g, 4, 4, w - 8, h - 8, 48);
      const s = this.status;
      text(g, { main: "QUESTCOLLECT", replay: "REPLAY", layout: "LAYOUT" }[this.page], 40, 52,
        this.page === "main" ? 26 : 30, { weight: 800, color: THEME.accent });
      if (this.page === "replay") {
        this.drawReplay(g, w);
      } else if (this.page === "main") {
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
    this.pages = { main: this.buildMain(), replay: this.buildReplay(), layout: this.buildLayout() };
    this.panel.widgets = this.pages.main;
    this.panel.mesh.visible = false;
  }

  private goto(page: Page) {
    this.page = page;
    this.panel.widgets = this.pages[page];
    this.panel.setHover(null);
    this.panel.invalidate();
    if (page === "replay") void this.browser.refresh();
  }

  /** Top-right: on the main page the way to the other two pages, elsewhere the way back. */
  private pageButtons(page: Page): Widget[] {
    const CW = this.panel.canvas.width;
    if (page !== "main") return [button("back", CW - 36 - 190, 20, 190, 64, () => "←  Back", () => this.goto("main"))];
    return [
      button("to-replay", CW - 36 - 92 - 12 - 150, 20, 150, 64, () => "⏯  Replay", () => this.goto("replay"),
        { active: () => !!this.status?.replay }),
      button("to-layout", CW - 36 - 92, 20, 92, 64, () => "⚙", () => this.goto("layout")),
    ];
  }

  /** The big record button, shared by the main and replay pages: Record / Take over here / Stop / Cancel. */
  private recButton(y: number, h: number): Widget {
    const s = () => this.status;
    const rec = () => s()?.rec_state ?? "idle";
    const CW = this.panel.canvas.width;
    // During a replay: continue it from the shown frame as a new take (the operator's correction).
    return button(`rec-${y}`, 36, y, CW - 72, h,
      () => (rec() === "recording" ? "Stop" : rec() === "countdown" ? "Cancel" : s()?.replay ? "Take over here" : "Record"),
      () => this.act.command(s()?.replay && rec() === "idle" ? "replay_branch" : "record_toggle"),
      {
        icon: () => (rec() === "recording" ? "■" : s()?.replay && rec() === "idle" ? "✋" : "●"), color: () => THEME.rec,
        active: () => rec() === "recording" || rec() === "countdown",
        enabled: () => rec() !== "review",
      });
  }

  private saveButtons(y: number, h: number): Widget[] {
    const cmd = this.act.command;
    const CW = this.panel.canvas.width, m = 36, gap = 14;
    const col3 = (CW - 2 * m - 2 * gap) / 3;
    const c3 = (i: number) => m + i * (col3 + gap);
    const rec = () => this.status?.rec_state ?? "idle";
    const canSave = () => rec() === "review" || rec() === "recording";
    return [
      button(`save-ok-${y}`, c3(0), y, col3, h, () => "Success", () => cmd("save", { success: true }),
        { icon: () => "✓", color: () => THEME.ok, enabled: canSave }),
      button(`save-fail-${y}`, c3(1), y, col3, h, () => "Failure", () => cmd("save", { success: false }),
        { icon: () => "✗", color: () => THEME.warn, enabled: canSave }),
      button(`discard-${y}`, c3(2), y, col3, h, () => "Discard", () => cmd("discard"),
        { icon: () => "🗑", color: () => THEME.rec, enabled: () => rec() !== "idle" }),
    ];
  }

  private buildMain(): Widget[] {
    const s = () => this.status;
    const cmd = this.act.command;
    const CW = this.panel.canvas.width;
    const m = 36, gap = 14;
    const col2 = (CW - 2 * m - gap) / 2;
    const col3 = (CW - 2 * m - 2 * gap) / 3;
    const c3 = (i: number) => m + i * (col3 + gap);
    const idle = () => (s()?.rec_state ?? "idle") === "idle" && !s()?.replay;

    return [
      ...this.pageButtons("main"),
      // input mode
      button("mode-ctrl", m, 134, col2, 110, () => "Controllers", () => cmd("set_mode", { mode: "controllers" }),
        { icon: () => "🎮", active: () => s()?.mode === "controllers" }),
      button("mode-hands", m + col2 + gap, 134, col2, 110, () => "Hands", () => cmd("set_mode", { mode: "hands" }),
        { icon: () => "✋", active: () => s()?.mode === "hands" }),

      // recording
      this.recButton(294, 120),
      ...this.saveButtons(430, 96),

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

  private buildReplay(): Widget[] {
    const s = () => this.status;
    const cmd = this.act.command;
    const b = this.browser;
    const CW = this.panel.canvas.width;
    const m = 36, gap = 14, arrow = 84;
    const col3 = (CW - 2 * m - 2 * gap) / 3;
    const col4 = (CW - 2 * m - 3 * gap) / 4;
    const c3 = (i: number) => m + i * (col3 + gap);
    const col5 = (CW - 2 * m - 4 * gap) / 5;
    const c4 = (i: number) => m + i * (col4 + gap);
    const c5 = (i: number) => m + i * (col5 + gap);
    const idle = () => (s()?.rec_state ?? "idle") === "idle";
    const r = () => s()?.replay ?? null;
    const fps = () => r()?.fps ?? 50;
    const play = (e: { path: string } | null) => { if (e) cmd("replay", { path: e.path }); };
    const seek = (d: number) => { const x = r(); if (x) cmd("replay_seek", { i: x.i + d }); };
    const frame = (d: number) => { const x = r(); if (x) { if (!x.paused) cmd("replay_pause", { paused: true }); seek(d); } };
    const replaying = () => !!r() && r()!.path === b.episode?.path;

    return [
      ...this.pageButtons("replay"),
      // dataset: ◀ title ▶ (title drawn in drawReplay)
      button("ds-prev", m, 130, arrow, 84, () => "◀", () => void b.stepDataset(-1), { enabled: () => b.datasets.length > 1 }),
      button("ds-next", CW - m - arrow, 130, arrow, 84, () => "▶", () => void b.stepDataset(1), { enabled: () => b.datasets.length > 1 }),
      // episode: ◀ [play this one] ▶
      button("ep-prev", m, 262, arrow, 96, () => "◀", () => play(b.step(-1)), { enabled: () => idle() && b.ep > 0 }),
      button("ep-play", m + arrow + gap, 262, CW - 2 * (m + arrow + gap), 96,
        () => {
          const e = b.episode;
          if (!e) return b.dataset ? "no episodes" : "no datasets";
          return `#${e.file.replace(/^episode_0*(?=\d)/, "").replace(".hdf5", "")} · ${e.success ? "✓" : "✗"} · ${(e.duration_s ?? 0).toFixed(1)} s`;
        },
        () => play(b.episode),
        { icon: () => (replaying() ? "⟲" : "▶"), enabled: () => idle() && !!b.episode }),
      button("ep-next", CW - m - arrow, 262, arrow, 96, () => "▶", () => play(b.step(1)),
        { enabled: () => idle() && b.ep < b.episodes.length - 1 }),
      // transport
      ...([[-5, "−5 s"], [-1, "−1 s"], [0, ""], [1, "+1 s"], [5, "+5 s"]] as [number, string][]).map(([d, label], i) =>
        d === 0
          ? button("rp-play", c5(i), 432, col5, 92, () => (r()?.paused ?? true ? "▶" : "❚❚"), () => cmd("replay_pause"),
            { enabled: () => !!r(), active: () => !!r() && !r()!.paused })
          : button(`rp-${d}`, c5(i), 432, col5, 92, () => label, () => seek(d * fps()), { enabled: () => !!r() })),
      button("rp-start", c4(0), 538, col4, 76, () => "⏮", () => cmd("replay_seek", { i: 0 }), { enabled: () => !!r() }),
      button("rp-fb", c4(1), 538, col4, 76, () => "‹ frame", () => frame(-1), { enabled: () => !!r() }),
      button("rp-ff", c4(2), 538, col4, 76, () => "frame ›", () => frame(1), { enabled: () => !!r() }),
      button("rp-speed", c4(3), 538, col4, 76, () => `${r()?.speed ?? 1}×`,
        () => { const v = r()?.speed ?? 1; cmd("replay_options", { speed: SPEEDS[(SPEEDS.indexOf(v) + 1) % SPEEDS.length] }); },
        { enabled: () => !!r() }),
      // take over, then save
      this.recButton(630, 110),
      ...this.saveButtons(754, 90),
      button("rp-close", c3(0), 858, col3, 72, () => "✕  Close", () => cmd("replay_stop"), { enabled: () => !!r() && idle() }),
      // Shared: after a take-over, releasing every arm hands the arms back to the policy. Only me: you finish it.
      button("rp-control", c3(1), 858, col3, 72,
        () => (!s()?.control.policy ? "No policy" : s()?.control.mode === "shared" ? "🤝  Shared" : "🎮  Only me"),
        () => cmd("set_control_mode", { mode: s()?.control.mode === "shared" ? "teleop" : "shared" }),
        { enabled: () => !!s()?.control.policy, active: () => s()?.control.mode === "shared" }),
      button("rp-refresh", c3(2), 858, col3, 72, () => "↻  Refresh", () => void b.refresh()),
    ];
  }

  /** Replay page text: dataset title, episode position, time and a progress bar. */
  private drawReplay(g: CanvasRenderingContext2D, w: number) {
    const b = this.browser, r = this.status?.replay ?? null;
    const m = 36, arrow = 84, gap = 14;
    text(g, "DATASET", 40, 112, 22, { color: THEME.dim, weight: 700 });
    const d = b.dataset;
    let title = d?.title ?? "—";
    g.font = `700 26px ${THEME.font}`;
    const maxW = w - 2 * (m + arrow + gap);
    while (title.length > 4 && g.measureText(title).width > maxW) title = title.slice(0, -2);
    if (d && title !== d.title) title = title.slice(0, -1) + "…";
    text(g, title, w / 2, 160, 26, { align: "center", weight: 700 });
    if (d) text(g, `${d.count} episodes${d.recording ? " · recording folder" : ""}`, w / 2, 194, 20, { align: "center", color: THEME.dim });
    text(g, "EPISODE", 40, 244, 22, { color: THEME.dim, weight: 700 });
    if (b.episode) text(g, `${b.ep + 1} / ${b.episodes.length}`, w - 40, 244, 22, { align: "right", color: THEME.dim, weight: 700 });
    text(g, "TIME", 40, 386, 22, { color: THEME.dim, weight: 700 });
    if (r) {
      const fps = r.fps ?? 50;
      text(g, `${(r.i / fps).toFixed(2)} / ${((r.n - 1) / fps).toFixed(1)} s`, w - 40, 386, 22, { align: "right", weight: 700 });
      roundRect(g, m, 404, w - 2 * m, 12, 6);
      g.fillStyle = "rgba(148,180,255,0.18)";
      g.fill();
      roundRect(g, m, 404, Math.max(12, (w - 2 * m) * r.i / Math.max(1, r.n - 1)), 12, 6);
      g.fillStyle = THEME.accent;
      g.fill();
    } else {
      text(g, "no replay: pick an episode", w - 40, 386, 22, { align: "right", color: THEME.dim });
    }
    const c = this.status?.control;
    text(g, c?.waits ? "Policy waits: Grip to take an arm, release all to hand back"
      : c?.mode === "shared" && this.status?.rec_state === "recording"
        ? (c.driver === "policy" ? "Policy drives: Grip to take over" : "You drive: release all to hand back")
        : "Right stick: scrub  ·  click: play / pause", w / 2, 962, 20, { align: "center", color: c?.waits ? THEME.warn : THEME.dim });
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
      ...this.pageButtons("layout"),
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
    this.browser.follow(status?.replay?.path);
    this.afterTakeOver(status);
    this.panel.invalidate();
  }

  /** When a take-over ends, go on with the replays instead of a live scene: saved -> the next episode,
   * discarded -> the same one again, paused at the frame it was taken over from (try again). */
  private afterTakeOver(s: Status | null) {
    if (!s) return;
    if (s.branch && s.rec_state !== "idle") {
      this.takeOver ??= { of: s.branch.of, step: s.branch.step, saved: s.saved_count };
      return;
    }
    const t = this.takeOver;
    if (!t || s.rec_state !== "idle") return;
    this.takeOver = null;
    if (s.saved_count > t.saved) {
      this.browser.follow(t.of);
      const next = this.browser.episode?.path === t.of ? this.browser.step(1) : null;
      if (next) this.act.command("replay", { path: next.path });
    } else {
      this.act.command("replay", { path: t.of, paused: true, i: t.step });
    }
  }

  setVisible(v: boolean) {
    this.visible = v;
    this.panel.mesh.visible = v;
    if (!v) this.panel.setHover(null);
  }
}
