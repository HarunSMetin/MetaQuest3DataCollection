// Short synthesized sound cues for the headset (no audio files). The server sends {type: "sound", name, arm}
// events (questcollect/runtime.py); each cue is panned to the arm's side so you hear which hand it was.

const PAN = { l: -0.6, r: 0.6 } as const;
const STORE_KEY = "qc.sound";

type Tone = { f: number; f2?: number; t?: number; d: number; g: number; type?: OscillatorType };

// name -> tones (frequency [Hz], optional glide target, start offset, duration [s], gain) and/or a noise burst.
const CUES: Record<string, { tones?: Tone[]; noise?: { d: number; g: number; hp: number } }> = {
  engage: { tones: [{ f: 520, f2: 780, d: 0.09, g: 0.12 }] },
  disengage: { tones: [{ f: 780, f2: 420, d: 0.09, g: 0.1 }] },
  grasp: { tones: [{ f: 300, f2: 180, d: 0.07, g: 0.22 }] },
  release: { tones: [{ f: 520, d: 0.04, g: 0.08 }] },
  click: { noise: { d: 0.018, g: 0.35, hp: 2500 }, tones: [{ f: 2300, d: 0.012, g: 0.08, type: "square" }] },
  unclick: { noise: { d: 0.03, g: 0.15, hp: 900 }, tones: [{ f: 420, f2: 240, d: 0.08, g: 0.12 }] },
  placed: { tones: [{ f: 784, d: 0.12, g: 0.14 }, { f: 1047, t: 0.11, d: 0.2, g: 0.14 }] },
  complete: { tones: [523, 659, 784, 1047].map((f, i) => ({ f, t: i * 0.11, d: i === 3 ? 0.45 : 0.14, g: 0.13, type: "triangle" as const })) },
  area: { tones: [{ f: 660, d: 0.22, g: 0.07 }, { f: 990, d: 0.22, g: 0.05 }] },
  collision: { tones: [{ f: 110, d: 0.18, g: 0.12, type: "square" }, { f: 116, d: 0.18, g: 0.08, type: "square" }] },
  table: { tones: [{ f: 170, f2: 90, d: 0.09, g: 0.2 }] },
  out_of_reach: { tones: [{ f: 300, d: 0.07, g: 0.1, type: "triangle" }, { f: 300, t: 0.13, d: 0.07, g: 0.1, type: "triangle" }] },
};

export class Sounds {
  private ctx: AudioContext | null = null;
  private noiseBuf: AudioBuffer | null = null;
  enabled = true;

  constructor() {
    try { this.enabled = localStorage.getItem(STORE_KEY) !== "off"; } catch { /* storage unavailable */ }
  }

  /** Create/resume the audio context. Browsers only allow this from a user gesture (Enter VR click). */
  unlock() {
    try {
      this.ctx ??= new AudioContext();
      void this.ctx.resume();
    } catch { /* no WebAudio */ }
  }

  toggle() {
    this.enabled = !this.enabled;
    try { localStorage.setItem(STORE_KEY, this.enabled ? "on" : "off"); } catch { /* storage unavailable */ }
    if (this.enabled) this.play("engage");
  }

  play(name: string, arm: "l" | "r" | null = null) {
    const cue = CUES[name], ctx = this.ctx;
    if (!cue || !ctx || !this.enabled || ctx.state !== "running") return;
    const out = ctx.createStereoPanner();
    out.pan.value = arm ? PAN[arm] : 0;
    out.connect(ctx.destination);
    const t0 = ctx.currentTime + 0.005;
    for (const tn of cue.tones ?? []) {
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      const s = t0 + (tn.t ?? 0);
      osc.type = tn.type ?? "sine";
      osc.frequency.setValueAtTime(tn.f, s);
      if (tn.f2) osc.frequency.exponentialRampToValueAtTime(tn.f2, s + tn.d);
      gain.gain.setValueAtTime(0.0001, s);
      gain.gain.exponentialRampToValueAtTime(tn.g, s + 0.006);
      gain.gain.exponentialRampToValueAtTime(0.0001, s + tn.d);
      osc.connect(gain).connect(out);
      osc.start(s);
      osc.stop(s + tn.d + 0.02);
    }
    if (cue.noise) {
      this.noiseBuf ??= (() => {
        const b = ctx.createBuffer(1, ctx.sampleRate * 0.1, ctx.sampleRate);
        const d = b.getChannelData(0);
        for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
        return b;
      })();
      const src = ctx.createBufferSource(), hp = ctx.createBiquadFilter(), gain = ctx.createGain();
      src.buffer = this.noiseBuf;
      hp.type = "highpass";
      hp.frequency.value = cue.noise.hp;
      gain.gain.setValueAtTime(cue.noise.g, t0);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + cue.noise.d);
      src.connect(hp).connect(gain).connect(out);
      src.start(t0);
      src.stop(t0 + cue.noise.d + 0.01);
    }
  }
}
