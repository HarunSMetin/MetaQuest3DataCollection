// World-anchored heads-up display above the far edge of the table.

import type { Status } from "../../protocol";
import { SRC_CONTROLLER, SRC_HAND } from "../../protocol";
import { glass, Panel, roundRect, text, THEME, wrapText } from "./Panel";

const W = 0.95, H = 0.34;

function fmtTime(s: number) {
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}

export class Hud {
  readonly panel: Panel;
  status: Status | null = null;
  rttMs: number | null = null;
  toast: { text: string; kind: string; until: number } | null = null;

  constructor() {
    this.panel = new Panel(W, H, 1400, (g, w, h) => this.draw(g, w, h));
  }

  showToast(text_: string, kind: string) {
    this.toast = { text: text_, kind, until: performance.now() + 2600 };
    this.panel.invalidate();
  }

  update(status: Status | null) {
    this.status = status;
    this.panel.invalidate();
  }

  private draw(g: CanvasRenderingContext2D, w: number, h: number) {
    const s = this.status;
    const pad = 36;
    glass(g, 6, 6, w - 12, h - 12, 46, s?.rec_state === "recording" ? THEME.rec : THEME.accent);
    if (!s) {
      text(g, "Connecting to QuestCollect…", w / 2, h / 2, 54, { align: "center", color: THEME.dim });
      return;
    }
    const now = performance.now();

    // --- state pill ---------------------------------------------------------------
    const pillW = 470, pillH = 120, px = pad, py = pad;
    let label = "READY", color = THEME.ok, sub = s.mode === "controllers" ? "Press A to record" : "Wrist menu → Record";
    if (s.replay) { label = "REPLAY"; color = THEME.accent; sub = `${s.replay.i} / ${s.replay.n}`; }
    else if (s.rec_state === "countdown") { label = `${Math.ceil(s.countdown)}`; color = THEME.warn; sub = "Get ready"; }
    else if (s.rec_state === "recording") { label = `REC ${fmtTime(s.elapsed)}`; color = THEME.rec; sub = `${s.steps} steps`; }
    else if (s.rec_state === "review") {
      label = s.success_seen ? "SUCCESS" : "REVIEW"; color = s.success_seen ? THEME.ok : THEME.warn;
      sub = s.mode === "controllers" ? "X save  ·  B discard" : "Save or discard in menu";
    }
    roundRect(g, px, py, pillW, pillH, 60);
    g.fillStyle = "rgba(0,0,0,0.35)";
    g.fill();
    g.lineWidth = 4;
    g.strokeStyle = color;
    g.save();
    g.shadowColor = color;
    g.shadowBlur = s.rec_state === "recording" ? 18 + 14 * Math.sin(now / 160) : 14;
    g.stroke();
    g.restore();
    const dotR = 18;
    g.beginPath();
    g.arc(px + 62, py + pillH / 2, dotR, 0, Math.PI * 2);
    g.fillStyle = color;
    g.globalAlpha = s.rec_state === "recording" ? 0.55 + 0.45 * Math.sin(now / 160) : 1;
    g.fill();
    g.globalAlpha = 1;
    text(g, label, px + 100, py + 48, label.length <= 2 ? 72 : 50, { weight: 800, color });
    text(g, sub, px + 102, py + 96, 28, { color: THEME.dim, weight: 500 });

    // --- task + instruction ---------------------------------------------------------
    const tx = px + pillW + 40;
    text(g, s.task.replace(/_/g, " ").toUpperCase(), tx, py + 26, 34, { weight: 800, color: THEME.text });
    wrapText(g, s.instruction, tx, py + 74, w - tx - 290, 38, 29, THEME.dim, 3);

    // --- stats column -----------------------------------------------------------------
    const sx = w - pad - 230;
    const stat = (label_: string, val: string, y: number, c = THEME.text) => {
      text(g, label_, sx, y, 24, { color: THEME.dim, weight: 600 });
      text(g, val, sx + 230, y, 30, { color: c, weight: 800, align: "right" });
    };
    stat("EPISODES", `${s.saved_count}`, py + 22);
    stat("MODE", s.mode === "controllers" ? "CTRL" : "HANDS", py + 66, THEME.accent);
    const rtt = this.rttMs ?? s.rtt_ms;
    stat("LATENCY", rtt != null ? `${Math.round(rtt)} ms` : "—", py + 110, rtt != null && rtt > 60 ? THEME.warn : THEME.ok);
    stat("INPUT", `${Math.round(s.input_hz)} Hz`, py + 154);

    // --- hint / success line ------------------------------------------------------------------
    const hy = py + 214;
    if (s.success && (s.rec_state === "recording" || s.rec_state === "review")) {
      text(g, "✓  Task solved", tx, hy, 34, { color: THEME.ok, weight: 800 });
    } else if (s.rec_state === "idle" && !s.replay) {
      const hint = s.mode === "controllers"
        ? "Hold GRIP to move  ·  TRIGGER closes gripper  ·  Y opens menu"
        : "Thumb+middle pinch to engage  ·  thumb+index = gripper  ·  palm up = menu";
      text(g, hint, tx, hy, 24, { color: THEME.dim, weight: 600 });
    }

    // --- bottom row: per-arm tracking + warnings ------------------------------------------
    const by = h - pad - 70;
    const arm = (key: "l" | "r", x: number) => {
      const c = key === "l" ? THEME.left : THEME.right;
      const tracked = s.tracked[key], engaged = s.engaged[key];
      roundRect(g, x, by, 300, 70, 35);
      g.fillStyle = engaged ? c : "rgba(0,0,0,0.3)";
      g.globalAlpha = engaged ? 0.9 : 1;
      g.fill();
      g.globalAlpha = 1;
      g.lineWidth = 3;
      g.strokeStyle = tracked ? c : "rgba(255,255,255,0.15)";
      g.stroke();
      const src = s.sources[key] === SRC_HAND ? "✋" : s.sources[key] === SRC_CONTROLLER ? "🎮" : "∅";
      text(g, src, x + 40, by + 36, 34, { align: "center" });
      text(g, `${key === "l" ? "LEFT" : "RIGHT"} ${engaged ? "ENGAGED" : tracked ? "TRACKED" : "LOST"}`, x + 74, by + 36, 26,
        { weight: 800, color: engaged ? "#0b1020" : tracked ? c : THEME.dim });
      // gripper bar
      const gw = s.grippers[key === "l" ? 0 : 1];
      roundRect(g, x + 74, by + 54, 200, 6, 3);
      g.fillStyle = "rgba(255,255,255,0.15)";
      g.fill();
      roundRect(g, x + 74, by + 54, 200 * gw, 6, 3);
      g.fillStyle = engaged ? "#0b1020" : c;
      g.fill();
    };
    arm("l", pad);
    arm("r", pad + 320);

    const warns: [string, string][] = [];
    if (s.mode_mismatch) warns.push([s.mode === "hands" ? "Put controllers down" : "Pick up controllers", THEME.warn]);
    if (s.warnings.arm_collision) warns.push(["Arms colliding", THEME.rec]);
    if (s.warnings.table_collision) warns.push(["Table contact", THEME.warn]);
    if (s.warnings.ik_saturated) warns.push(["Out of reach", THEME.warn]);
    if (!s.connected) warns.push(["No input", THEME.dim]);
    if (this.toast && this.toast.until > now) {
      const c = this.toast.kind === "success" ? THEME.ok : this.toast.kind === "warn" ? THEME.warn : this.toast.kind === "rec" ? THEME.rec : THEME.accent;
      warns.unshift([this.toast.text, c]);
    }
    let wx = pad + 660;
    for (const [msg, c] of warns.slice(0, 2)) {
      g.font = `700 26px ${THEME.font}`;
      const tw = g.measureText(msg).width + 56;
      if (wx + tw > w - pad) break;
      roundRect(g, wx, by + 6, tw, 58, 29);
      g.fillStyle = "rgba(0,0,0,0.35)";
      g.fill();
      g.strokeStyle = c;
      g.lineWidth = 3;
      g.stroke();
      text(g, msg, wx + 28, by + 36, 26, { color: c, weight: 700 });
      wx += tw + 14;
    }
  }

  private lastAnim = 0;

  /** Animate (pulse) while recording at ~12 fps (each redraw re-uploads the texture); otherwise redraw only on change. */
  tick() {
    const s = this.status;
    const now = performance.now();
    if (s && (s.rec_state === "recording" || s.rec_state === "countdown") && now - this.lastAnim > 80) {
      this.lastAnim = now;
      this.panel.invalidate();
    }
    if (this.toast && this.toast.until < performance.now()) { this.toast = null; this.panel.invalidate(); }
    this.panel.redraw();
  }
}
