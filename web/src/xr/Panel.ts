// Canvas-rendered UI panels for WebXR: crisp text, glass look, ray + poke interaction.

import * as THREE from "three";

export const THEME = {
  font: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
  bg0: "rgba(12,16,26,0.82)",
  bg1: "rgba(22,28,44,0.88)",
  stroke: "rgba(148,180,255,0.28)",
  text: "#e8eefc",
  dim: "#8b97b5",
  accent: "#7c9cff",
  left: "#22d3ee",
  right: "#f472b6",
  rec: "#ff4d6d",
  ok: "#34d399",
  warn: "#fbbf24",
};

export function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath();
  g.roundRect(x, y, w, h, r);
}

export function glass(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number, tint = THEME.accent) {
  const grad = g.createLinearGradient(x, y, x, y + h);
  grad.addColorStop(0, THEME.bg1);
  grad.addColorStop(1, THEME.bg0);
  roundRect(g, x, y, w, h, r);
  g.fillStyle = grad;
  g.fill();
  g.save();
  g.shadowColor = tint;
  g.shadowBlur = 18;
  g.strokeStyle = THEME.stroke;
  g.lineWidth = 3;
  g.stroke();
  g.restore();
  // top sheen
  const sheen = g.createLinearGradient(x, y, x, y + h * 0.4);
  sheen.addColorStop(0, "rgba(255,255,255,0.07)");
  sheen.addColorStop(1, "rgba(255,255,255,0)");
  roundRect(g, x + 2, y + 2, w - 4, h * 0.4, r);
  g.fillStyle = sheen;
  g.fill();
}

export function text(g: CanvasRenderingContext2D, s: string, x: number, y: number, size: number,
  opts: { color?: string; weight?: number; align?: CanvasTextAlign; baseline?: CanvasTextBaseline } = {}) {
  g.font = `${opts.weight ?? 600} ${size}px ${THEME.font}`;
  g.fillStyle = opts.color ?? THEME.text;
  g.textAlign = opts.align ?? "left";
  g.textBaseline = opts.baseline ?? "middle";
  g.fillText(s, x, y);
}

export function wrapText(g: CanvasRenderingContext2D, s: string, x: number, y: number, maxW: number, lineH: number,
  size: number, color = THEME.dim, maxLines = 3) {
  g.font = `500 ${size}px ${THEME.font}`;
  const words = s.split(" ");
  let line = "", lines = 0;
  for (const w of words) {
    const test = line ? `${line} ${w}` : w;
    if (g.measureText(test).width > maxW && line) {
      text(g, line, x, y + lines * lineH, size, { color, weight: 500 });
      line = w;
      if (++lines >= maxLines) return;
    } else line = test;
  }
  if (line) text(g, line, x, y + lines * lineH, size, { color, weight: 500 });
}

export interface Widget {
  id: string;
  x: number; y: number; w: number; h: number;
  draw: (g: CanvasRenderingContext2D, st: { hover: boolean; pressed: boolean }) => void;
  onPress?: () => void;
  enabled?: () => boolean;
}

export class Panel {
  readonly mesh: THREE.Mesh;
  readonly canvas = document.createElement("canvas");
  readonly g: CanvasRenderingContext2D;
  readonly texture: THREE.CanvasTexture;
  widgets: Widget[] = [];
  hover: string | null = null;
  pressedUntil = new Map<string, number>();
  private dirty = true;
  private lastPress = 0;

  constructor(readonly widthM: number, readonly heightM: number, readonly pxPerM = 1400,
    private background?: (g: CanvasRenderingContext2D, w: number, h: number) => void) {
    this.canvas.width = Math.round(widthM * pxPerM);
    this.canvas.height = Math.round(heightM * pxPerM);
    this.g = this.canvas.getContext("2d")!;
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.anisotropy = 8;
    this.texture.generateMipmaps = true;
    this.texture.minFilter = THREE.LinearMipmapLinearFilter;
    const mat = new THREE.MeshBasicMaterial({ map: this.texture, transparent: true, depthWrite: false, toneMapped: false });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(widthM, heightM), mat);
    this.mesh.renderOrder = 10;
    this.mesh.userData.panel = this;
  }

  invalidate() { this.dirty = true; }

  redraw(force = false) {
    if (!this.dirty && !force) return;
    const { g, canvas } = this;
    g.clearRect(0, 0, canvas.width, canvas.height);
    this.background?.(g, canvas.width, canvas.height);
    const now = performance.now();
    for (const w of this.widgets) {
      g.save();
      if (w.enabled && !w.enabled()) g.globalAlpha = 0.35;
      w.draw(g, { hover: this.hover === w.id, pressed: (this.pressedUntil.get(w.id) ?? 0) > now });
      g.restore();
    }
    this.texture.needsUpdate = true;
    this.dirty = false;
  }

  /** Widget under a local-space point on the panel plane. */
  widgetAt(local: THREE.Vector3): Widget | null {
    const px = (local.x / this.widthM + 0.5) * this.canvas.width;
    const py = (0.5 - local.y / this.heightM) * this.canvas.height;
    for (const w of this.widgets) {
      if (px >= w.x && px <= w.x + w.w && py >= w.y && py <= w.y + w.h) return w;
    }
    return null;
  }

  setHover(id: string | null) {
    if (id !== this.hover) {
      this.hover = id;
      this.dirty = true;
    }
  }

  press(w: Widget) {
    const now = performance.now();
    if (now - this.lastPress < 250 || (w.enabled && !w.enabled())) return false;
    this.lastPress = now;
    this.pressedUntil.set(w.id, now + 180);
    this.dirty = true;
    setTimeout(() => this.invalidate(), 200);
    w.onPress?.();
    return true;
  }
}

/** A pill-shaped button widget. */
export function button(id: string, x: number, y: number, w: number, h: number, label: () => string,
  onPress: () => void, opts: { color?: () => string; active?: () => boolean; icon?: () => string; enabled?: () => boolean } = {}): Widget {
  return {
    id, x, y, w, h, onPress, enabled: opts.enabled,
    draw: (g, st) => {
      const active = opts.active?.() ?? false;
      const color = opts.color?.() ?? THEME.accent;
      roundRect(g, x, y, w, h, h / 2.6);
      const grad = g.createLinearGradient(x, y, x, y + h);
      if (active) {
        grad.addColorStop(0, color);
        grad.addColorStop(1, shade(color, -0.25));
      } else {
        grad.addColorStop(0, st.hover ? "rgba(70,86,128,0.95)" : "rgba(40,50,78,0.92)");
        grad.addColorStop(1, st.hover ? "rgba(52,64,98,0.95)" : "rgba(28,36,58,0.92)");
      }
      g.fillStyle = grad;
      g.fill();
      g.lineWidth = st.hover ? 4 : 2;
      g.strokeStyle = st.hover || active ? color : "rgba(148,180,255,0.25)";
      if (st.hover) { g.shadowColor = color; g.shadowBlur = 20; }
      g.stroke();
      g.shadowBlur = 0;
      if (st.pressed) {
        roundRect(g, x, y, w, h, h / 2.6);
        g.fillStyle = "rgba(255,255,255,0.25)";
        g.fill();
      }
      const icon = opts.icon?.();
      const label_ = label();
      const size = Math.min(h * 0.36, 44);
      if (icon) {
        text(g, icon, x + w / 2, y + h * 0.38, size * 1.25, { align: "center", color: active ? "#0b1020" : color });
        text(g, label_, x + w / 2, y + h * 0.74, size * 0.72, { align: "center", color: active ? "#0b1020" : THEME.text });
      } else {
        text(g, label_, x + w / 2, y + h / 2, size, { align: "center", color: active ? "#0b1020" : THEME.text });
      }
    },
  };
}

export function shade(hex: string, amt: number): string {
  const c = new THREE.Color(hex);
  const hsl = { h: 0, s: 0, l: 0 };
  c.getHSL(hsl);
  c.setHSL(hsl.h, hsl.s, THREE.MathUtils.clamp(hsl.l + amt, 0, 1));
  return `#${c.getHexString()}`;
}
