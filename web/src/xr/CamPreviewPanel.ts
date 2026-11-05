// Live previews of the recorded cameras, as a vertical column, a horizontal row or a grid (wrist menu → Layout;
// the chosen layout and cameras persist). Polls the server's JPEG previews (GET /api/preview/<cam>.jpg, ~8 Hz)
// only while visible. The panel resizes to fit its cells.

import { glass, Panel, text, THEME } from "./Panel";

export const PREVIEW_CAMS = [
  { name: "cam_left_wrist", label: "LEFT WRIST", short: "L wrist", color: THEME.left },
  { name: "cam_high", label: "TOP", short: "Top", color: THEME.accent },
  { name: "cam_low", label: "LOW", short: "Low", color: THEME.warn },
  { name: "cam_right_wrist", label: "RIGHT WRIST", short: "R wrist", color: THEME.right },
];
const DEFAULT_CAMS = ["cam_left_wrist", "cam_high", "cam_right_wrist"];  // what the policy sees
export type PreviewLayout = "column" | "row" | "grid";

const PX = 1400;
const PAD = 18, GAP = 12, LABEL_H = 34;
const CELL_W = 342, CELL_H = (CELL_W * 3) / 4;  // px; the previews are 4:3
const POLL_MS = 125;
const LAYOUT_KEY = "qc.preview.layout", CAMS_KEY = "qc.preview.cams";

export class CamPreviewPanel {
  readonly panel: Panel;
  private images = new Map<string, ImageBitmap>();
  private visible = false;
  private inflight = false;
  private lastPoll = 0;
  private layout_: PreviewLayout = "column";
  private cams = new Set(DEFAULT_CAMS);

  constructor() {
    try {
      const l = localStorage.getItem(LAYOUT_KEY);
      if (l === "column" || l === "row" || l === "grid") this.layout_ = l;
      const c = JSON.parse(localStorage.getItem(CAMS_KEY) ?? "null");
      const known = Array.isArray(c) ? c.filter((n) => PREVIEW_CAMS.some((p) => p.name === n)) : [];
      if (known.length) this.cams = new Set(known);
    } catch { /* storage unavailable or corrupt: defaults */ }
    this.panel = new Panel(0.1, 0.1, PX, (g, w, h) => this.draw(g, w, h));
    this.fit();
    this.panel.mesh.visible = false;
  }

  setVisible(on: boolean) {
    this.visible = on;
    this.panel.mesh.visible = on;
    this.panel.invalidate();
  }

  get shown() { return this.visible; }
  get layout() { return this.layout_; }

  setLayout(l: PreviewLayout) {
    this.layout_ = l;
    this.fit();
    this.persist();
  }

  camOn(name: string) { return this.cams.has(name); }

  /** Show/hide one camera; the last one stays on. */
  toggleCam(name: string) {
    if (this.cams.has(name)) { if (this.cams.size > 1) this.cams.delete(name); }
    else this.cams.add(name);
    this.fit();
    this.persist();
  }

  private persist() {
    try {
      localStorage.setItem(LAYOUT_KEY, this.layout_);
      localStorage.setItem(CAMS_KEY, JSON.stringify([...this.cams]));
    } catch { /* storage unavailable */ }
  }

  private shownCams() { return PREVIEW_CAMS.filter((c) => this.cams.has(c.name)); }

  private grid(): [number, number] {
    const n = this.shownCams().length;
    if (this.layout_ === "column") return [1, n];
    if (this.layout_ === "row") return [n, 1];
    const cols = Math.ceil(Math.sqrt(n));
    return [cols, Math.ceil(n / cols)];
  }

  private fit() {
    const [cols, rows] = this.grid();
    const w = 2 * PAD + cols * CELL_W + (cols - 1) * GAP;
    const h = 2 * PAD + rows * (LABEL_H + CELL_H) + (rows - 1) * GAP;
    this.panel.resize(w / PX, h / PX);
    this.panel.invalidate();
  }

  private draw(g: CanvasRenderingContext2D, w: number, h: number) {
    glass(g, 4, 4, w - 8, h - 8, 30);
    const [cols] = this.grid();
    this.shownCams().forEach((c, i) => {
      const col = i % cols, row = Math.floor(i / cols);
      const x = PAD + col * (CELL_W + GAP), top = PAD + row * (LABEL_H + CELL_H + GAP), y = top + LABEL_H;
      text(g, c.label, x + 4, top + LABEL_H / 2, 24, { weight: 800, color: c.color });
      const img = this.images.get(c.name);
      if (img) g.drawImage(img, x, y, CELL_W, CELL_H);
      else {
        g.fillStyle = "rgba(0,0,0,0.4)";
        g.fillRect(x, y, CELL_W, CELL_H);
        text(g, "…", x + CELL_W / 2, y + CELL_H / 2, 40, { align: "center", color: THEME.dim });
      }
      g.strokeStyle = c.color;
      g.lineWidth = 3;
      g.strokeRect(x, y, CELL_W, CELL_H);
    });
  }

  private async poll() {
    this.inflight = true;
    try {
      const t = Date.now();
      await Promise.all(this.shownCams().map(async (c) => {
        const r = await fetch(`/api/preview/${c.name}.jpg?t=${t}`, { cache: "no-store" });
        if (!r.ok) return;  // 404 until the server has rendered a first frame (or the camera isn't in record.cameras)
        const img = await createImageBitmap(await r.blob());
        this.images.get(c.name)?.close();
        this.images.set(c.name, img);
      }));
      this.panel.invalidate();
    } catch { /* server restarting: try again next tick */ }
    this.inflight = false;
  }

  tick() {
    const now = performance.now();
    if (this.visible && !this.inflight && now - this.lastPoll > POLL_MS) {
      this.lastPoll = now;
      void this.poll();
    }
    if (this.visible) this.panel.redraw();
  }
}
