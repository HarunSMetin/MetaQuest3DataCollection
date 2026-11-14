// Episode browser for the headset's Replay page: pick a dataset (GET /api/datasets) and step through its episodes
// (GET /api/episodes), so a replay can be started, scrubbed and taken over without the dashboard.

import type { DatasetRow, EpisodeRow } from "../protocol";

const KEY = "qc.xr.replayDataset";

export class ReplayBrowser {
  datasets: DatasetRow[] = [];
  episodes: EpisodeRow[] = [];
  ds = -1;
  ep = -1;
  private loaded = false;

  constructor(private changed: () => void) {}

  get dataset(): DatasetRow | null { return this.datasets[this.ds] ?? null; }
  get episode(): EpisodeRow | null { return this.episodes[this.ep] ?? null; }

  /** Reload the dataset list (keeps the chosen one; first time: the last one used, else the first that isn't
   * the recording folder, where the policy rollouts to correct usually are). */
  async refresh(): Promise<void> {
    const rows: DatasetRow[] = await fetch("/api/datasets").then((r) => r.json()).catch(() => []);
    const want = this.dataset?.name ?? load();
    this.datasets = rows;
    let i = rows.findIndex((d) => d.name === want);
    if (i < 0) i = rows.findIndex((d) => !d.recording && d.count > 0);
    this.ds = i < 0 ? (rows.length ? 0 : -1) : i;
    await this.loadEpisodes(this.loaded ? this.episode?.file : undefined);
    this.loaded = true;
  }

  async stepDataset(delta: number): Promise<void> {
    if (!this.datasets.length) return;
    this.ds = (this.ds + delta + this.datasets.length) % this.datasets.length;
    try { localStorage.setItem(KEY, this.datasets[this.ds].name); } catch { /* private window */ }
    await this.loadEpisodes();
  }

  /** Next/previous episode (clamped); returns it, or null at either end. */
  step(delta: number): EpisodeRow | null {
    const i = this.ep + delta;
    if (i < 0 || i >= this.episodes.length) return null;
    this.ep = i;
    this.changed();
    return this.episodes[i];
  }

  /** Follow a replay started elsewhere (dashboard) so prev/next continue from it. */
  follow(path: string | undefined) {
    if (!path || this.episode?.path === path) return;
    const i = this.episodes.findIndex((e) => e.path === path);
    if (i >= 0) { this.ep = i; this.changed(); }
  }

  private async loadEpisodes(keep?: string): Promise<void> {
    const d = this.dataset;
    this.episodes = d ? await fetch(`/api/episodes?dataset=${encodeURIComponent(d.name)}`)
      .then((r) => r.json()).catch(() => []) : [];
    const i = keep ? this.episodes.findIndex((e) => e.file === keep) : -1;
    this.ep = i >= 0 ? i : (this.episodes.length ? 0 : -1);
    this.changed();
  }
}

function load(): string | null {
  try { return localStorage.getItem(KEY); } catch { return null; }
}
