import { useEffect, useMemo, useRef, useState } from "react";
import type { AuditRows, DatasetRow, EpisodeEdit, EpisodeRow, Status, TaskSpec } from "../protocol";
import { fmt, type Cmd } from "./util";

type Filter = "all" | "success" | "failed" | "training" | "skipped";
const FILTERS: [Filter, string][] = [["all", "All"], ["success", "✓ Success"], ["failed", "✗ Failed"], ["training", "Training"], ["skipped", "Skipped"]];
export const REPLAY_SPEEDS = [0.25, 0.5, 1, 2, 4];
const PRESET_TAGS = ["clean", "slow", "messy", "recovery", "near miss"];

export const epNumber = (file: string) => `#${parseInt(file.match(/(\d+)\.hdf5$/)?.[1] ?? "0", 10)}`;
const fmtDate = (iso: string) => {
  const d = new Date(iso);
  return isNaN(+d) ? iso : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};
const SOURCE_LABEL: Record<string, string> = { teleop: "🎮 teleop", policy: "🤖 policy", shared: "🤝 shared", generated: "🧬 generated", retimed: "⏩ retimed" };
/** "data/episodes_generated/episode_000012.hdf5" -> "episodes_generated #12" */
const originLabel = (path: string) => { const p = path.split("/"); return `${p[p.length - 2] ?? ""} ${epNumber(p[p.length - 1])}`; };
const rgb = (c: number[]) => `rgb(${c.slice(0, 3).map((v) => Math.round(v * 255)).join(",")})`;

/** Training use of an episode: "train", "skipped" (the curate rules) or "excluded" (by hand); null = audit not loaded. */
function usage(e: EpisodeRow, audit: AuditRows): "train" | "skipped" | "excluded" | null {
  if (e.excluded) return "excluded";
  const a = audit[e.file];
  return !a ? null : a.verdict ? "skipped" : "train";
}

export function EpisodeBrowser({ episodes, audit, selected, onSelect, onEdit, onDelete, task, goRecord, datasets, dataset, onDataset }: {
  episodes: EpisodeRow[]; audit: AuditRows; selected: EpisodeRow | null; onSelect: (e: EpisodeRow) => void;
  onEdit: (e: EpisodeRow, edit: EpisodeEdit) => void; onDelete: (e: EpisodeRow) => void; task: TaskSpec | null; goRecord: () => void;
  datasets: DatasetRow[]; dataset: string; onDataset: (name: string) => void;
}) {
  const ds = datasets.find((d) => d.name === dataset);
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const rows = useMemo(() => episodes.slice().reverse(), [episodes]);
  const allTags = useMemo(() => [...new Set(rows.flatMap((e) => e.tags))].sort(), [rows]);
  const match = (e: EpisodeRow, f: Filter) => f === "all" || (f === "success" ? e.success : f === "failed" ? !e.success
    : f === "training" ? usage(e, audit) === "train" : usage(e, audit) === "skipped" || usage(e, audit) === "excluded");
  const q = query.trim().toLowerCase();
  const shown = rows.filter((e) => match(e, filter) && (!q || [epNumber(e.file), e.note, e.operator, ...e.tags].some((s) => s?.toLowerCase().includes(q))));
  const total = rows.reduce((a, e) => a + (e.duration_s || 0), 0);
  const ok = rows.filter((e) => e.success).length;
  const train = rows.filter((e) => usage(e, audit) === "train").length;

  const list = useRef<HTMLDivElement>(null);
  useEffect(() => { list.current?.querySelector(".tr.sel")?.scrollIntoView({ block: "nearest" }); }, [selected?.file]);

  const picker = datasets.length > 1 && (
    <div className="card dataset-pick">
      <label>
        <span className="muted small">Dataset</span>
        <select value={dataset} onChange={(e) => onDataset(e.target.value)}>
          {datasets.map((d) => <option key={d.name} value={d.name}>{d.title} ({d.count})</option>)}
        </select>
      </label>
      {ds?.description && <p className="muted small">{ds.description}</p>}
      <code className="muted small">{ds?.readonly ? `outputs/${dataset.split("/")[1]}/eval/${dataset.split("/")[2]}` : `data/${dataset}`}</code>
      {ds?.readonly && <p className="muted small">Replay a failure, pause where it went wrong and press <b>Take over here</b>: your correction is recorded as a new take in the recording folder.</p>}
    </div>
  );
  if (!rows.length && !ds?.recording && datasets.length) {
    return <>{picker}<div className="card empty-state"><b>No episodes in this dataset</b></div></>;
  }
  if (!rows.length) {
    return (
      <div className="card empty-state">
        <b>No episodes yet</b>
        <p className="muted">Recorded takes show up here. You can replay, label and clean them up before training.</p>
        <button className="btn" onClick={goRecord}>● Go to Record</button>
      </div>
    );
  }
  return (
    <>
      {picker}
      <div className="card ep-list">
        <div className="summary">
          <span><b>{rows.length}</b> episodes</span>
          <span><b>{Math.round((100 * ok) / rows.length)}%</b> success</span>
          <span><b>{fmt(total)}</b> total</span>
          <span title="Episodes a training dataset would use (the qc audit rules plus your exclusions)"><b>{train}</b> for training</span>
        </div>
        <input className="search" placeholder="Search #, tag, note…" value={query} onChange={(e) => setQuery(e.target.value)} />
        <div className="filters">
          {FILTERS.map(([f, label]) => (
            <button key={f} className={filter === f ? "active" : ""} onClick={() => setFilter(f)}>
              {label} <small>{rows.filter((e) => match(e, f)).length}</small>
            </button>
          ))}
        </div>
        {allTags.length > 0 && (
          <div className="tag-filter">{allTags.map((t) => (
            <button key={t} className={`tag ${q === t.toLowerCase() ? "on" : ""}`} onClick={() => setQuery(q === t.toLowerCase() ? "" : t)}>{t}</button>
          ))}</div>
        )}
        <div className="table" ref={list}>
          {shown.length === 0 && <div className="empty">Nothing matches.</div>}
          {shown.map((e) => {
            const use = usage(e, audit);
            return (
              <button key={e.file} className={`tr ${selected?.file === e.file ? "sel" : ""}`} onClick={() => onSelect(e)}>
                <span className={`badge ${e.success ? "ok" : "warn"}`} title={e.success ? "Success" : "Failure"}>{e.success ? "✓" : "✗"}</span>
                <span className="ep-main">
                  <b>{epNumber(e.file)}</b>
                  <span className="muted">{e.duration_s?.toFixed(1)} s · {e.origin ? `from ${originLabel(e.origin)}` : e.branch_of ? `↳ ${originLabel(e.branch_of)}` : fmtDate(e.created_at)}</span>
                </span>
                <span className="ep-tags">
                  {e.note && <i title={e.note}>✎</i>}
                  {e.tags.slice(0, 2).map((t) => <span key={t} className="tag">{t}</span>)}
                  {e.tags.length > 2 && <span className="tag">+{e.tags.length - 2}</span>}
                </span>
                {use && use !== "train" ? <span className={`pill ${use}`}>{use}</span> : <span />}
              </button>
            );
          })}
        </div>
        <div className="keys muted"><kbd>↑</kbd><kbd>↓</kbd> episode · <kbd>Space</kbd> play · <kbd>←</kbd><kbd>→</kbd> ±1 s · <kbd>,</kbd><kbd>.</kbd> frame · <kbd>[</kbd><kbd>]</kbd> speed · <kbd>T</kbd> take over</div>
      </div>
      {selected && <EpisodeDetail key={selected.file} ep={selected} audit={audit} task={task} allTags={allTags} onEdit={onEdit} readonly={!!ds?.readonly}
        onDelete={ds?.recording ? onDelete : null} />}
    </>
  );
}

function EpisodeDetail({ ep, audit, task, allTags, onEdit, onDelete, readonly }: {
  ep: EpisodeRow; audit: AuditRows; task: TaskSpec | null; allTags: string[]; readonly: boolean;
  onEdit: (e: EpisodeRow, edit: EpisodeEdit) => void; onDelete: ((e: EpisodeRow) => void) | null;
}) {
  const a = audit[ep.file];
  const g = ep.generation;
  const [note, setNote] = useState(ep.note);
  const use = usage(ep, audit);
  const relabelled = ep.success !== ep.recorded_success;
  return (
    <div className="card ep-detail">
      <div className="card-head">
        <h3>Episode {epNumber(ep.file)}</h3>
        {onDelete && <button className="btn ghost small danger" onClick={() => onDelete(ep)}>🗑 Delete</button>}
      </div>
      <div className="muted meta-line">
        {fmtDate(ep.created_at)} · {ep.source === "teleop" ? (ep.input_mode === "hands" ? "✋ hands" : "🎮 controllers") : SOURCE_LABEL[ep.source] ?? ep.source} · {ep.operator} · {ep.duration_s?.toFixed(1)} s
      </div>
      {ep.branch_of && (
        <div className="muted small origin" title={ep.branch_of}>
          ↳ Taken over from <b>{originLabel(ep.branch_of)}</b> at {((ep.branch_step ?? 0) / (a?.fps ?? 50)).toFixed(1)} s
        </div>
      )}
      {ep.origin && (
        <div className="muted small origin" title={ep.origin}>
          {ep.source === "retimed" ? "Retimed" : "Generated"} from <b>{originLabel(ep.origin)}</b>
          {ep.retime_moves != null && <> · {ep.retime_moves} moves overlapped</>}
          {g && <> · seed {g.seed} · tower {g.tower_yaw_deg >= 0 ? "+" : ""}{g.tower_yaw_deg.toFixed(1)}° · base shift {(Math.hypot(...g.base_shift_m) * 1000).toFixed(0)} mm</>}
        </div>
      )}
      <dl>
        {!readonly && <>
        <dt>Outcome</dt>
        <dd>
          <div className="seg mini">
            <button className={ep.success ? "active ok" : ""} onClick={() => onEdit(ep, { success: ep.recorded_success ? null : true })}>✓ Success</button>
            <button className={!ep.success ? "active bad" : ""} onClick={() => onEdit(ep, { success: ep.recorded_success ? false : null })}>✗ Failure</button>
          </div>
          {relabelled && <span className="muted small"> changed by you · saved as {ep.recorded_success ? "success" : "failure"}</span>}
        </dd>
        <dt>Training</dt>
        <dd className="training">
          <span className={`pill ${use ?? ""}`}>{use === "train" ? "used" : use ?? "…"}</span>
          {use === "skipped" && <span className="muted small">{a?.verdict}</span>}
          <button className="btn ghost small" onClick={() => onEdit(ep, { exclude: !ep.excluded })}>{ep.excluded ? "Include again" : "Exclude"}</button>
        </dd>
        </>}
        {a && (
          <>
            <dt>Quality</dt>
            <dd className="small bricks">
              <span title="Share of the take where nothing moved">idle {Math.round(a.idle_frac * 100)}%</span>
              <span title="Separate times an arm link touched the table / the arms touched / a hand dragged on the table">
                contacts {a.contact_events.link_table}/{a.contact_events.arm_arm}/{a.contact_events.tip_table}
              </span>
              {a.level_times.every((t) => t == null) ? <span className="muted">no brick clicked in</span>
                : a.level_times.map((t, k) => {
                  const b = task?.bricks[task.goal[k + 1]?.brick];
                  return t == null ? null : <span key={k} title={`${b?.label ?? "brick"} clicked in`}><i style={{ background: b ? rgb(b.rgba) : "#888" }} />{t.toFixed(1)} s</span>;
                })}
            </dd>
          </>
        )}
        {!readonly && <>
        <dt>Tags</dt>
        <dd><TagEditor tags={ep.tags} known={allTags} onChange={(tags) => onEdit(ep, { tags })} /></dd>
        <dt>Note</dt>
        <dd>
          <textarea rows={2} placeholder="Anything worth remembering about this take" value={note}
            onChange={(e) => setNote(e.target.value)} onBlur={() => note !== ep.note && onEdit(ep, { note })} />
        </dd>
        </>}
      </dl>
    </div>
  );
}

function TagEditor({ tags, known, onChange }: { tags: string[]; known: string[]; onChange: (t: string[]) => void }) {
  const [text, setText] = useState("");
  const [focus, setFocus] = useState(false);
  const add = (t: string) => {
    t = t.trim();
    if (t && !tags.includes(t)) onChange([...tags, t]);
    setText("");
  };
  const suggest = [...new Set([...known, ...PRESET_TAGS])].filter((t) => !tags.includes(t)).slice(0, 5);
  return (
    <div className="tag-edit">
      {tags.map((t) => <span key={t} className="tag">{t}<button title="Remove" onClick={() => onChange(tags.filter((x) => x !== t))}>×</button></span>)}
      <input placeholder="Add tag…" value={text} onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === ",") { e.preventDefault(); add(text); } }} onFocus={() => setFocus(true)} onBlur={() => { setTimeout(() => setFocus(false), 150); if (text) add(text); }} />
      {(focus || !tags.length) && suggest.slice(0, focus ? 5 : 3).map((t) => <button key={t} className="tag ghost" onMouseDown={(e) => e.preventDefault()} onClick={() => add(t)}>+ {t}</button>)}
    </div>
  );
}

/** Under the 3D view on the Episodes tab: play/pause and a timeline with a mark where each brick clicked in. */
export function ReplayBar({ status, ep, audit, task, cmd }: {
  status: Status | null; ep: EpisodeRow | null; audit: AuditRows; task: TaskSpec | null; cmd: Cmd;
}) {
  const [drag, setDrag] = useState<number | null>(null);
  const release = useRef(0);
  const r = status?.replay;
  if (!ep) return <div className="card replay-bar muted">Select an episode on the right to replay it here.</div>;
  if (!r || r.path !== ep.path) {
    return (
      <div className="card replay-bar">
        {status?.rec_state !== "idle"
          ? <span className="muted">A take is being recorded: replay is available after it's saved or discarded.</span>
          : <button className="btn" onClick={() => cmd("replay", { path: ep.path })}>▶ Replay {epNumber(ep.file)}</button>}
      </div>
    );
  }
  const a = audit[ep.file];
  const fps = a?.fps ?? 50;
  const last = Math.max(r.n - 1, 1);
  const i = drag ?? r.i;
  const seek = (v: number) => { setDrag(v); cmd("replay_seek", { i: v }); };
  const done = () => { clearTimeout(release.current); release.current = window.setTimeout(() => setDrag(null), 300); };
  const step = (d: number) => { if (!r.paused) cmd("replay_pause", { paused: true }); cmd("replay_seek", { i: r.i + d }); };
  const clicks = (a?.level_times ?? []).map((t, k) => ({ t, b: task?.bricks[task.goal[k + 1]?.brick] })).filter((c) => c.t != null);
  return (
    <div className="card replay-bar">
      <div className="scrub">
        <div className="fill" style={{ width: `${(100 * i) / last}%` }} />
        <input type="range" min={0} max={last} value={i} aria-label="Replay position"
          onChange={(e) => seek(+e.target.value)} onPointerUp={done} onKeyUp={done} />
        {clicks.map(({ t, b }, k) => (
          <button key={k} className="mark" style={{ left: `${(100 * t! * fps) / last}%`, background: b ? rgb(b.rgba) : "#fff" }}
            title={`${b?.label ?? "brick"} clicked in at ${t!.toFixed(1)} s: jump there`} onClick={() => cmd("replay_seek", { i: Math.round(t! * fps) })} />
        ))}
      </div>
      <div className="replay-controls">
        <button className="icon-btn" title="Previous frame (,)" onClick={() => step(-1)}>⏮</button>
        <button className="play" title="Play / pause (Space)" onClick={() => cmd("replay_pause")}>{r.paused ? "▶" : "❚❚"}</button>
        <button className="icon-btn" title="Next frame (.)" onClick={() => step(1)}>⏭</button>
        <span className="time">{fmt(i / fps)} <span className="muted">/ {fmt(last / fps)}</span></span>
        <div className="seg mini speed" title="Replay speed ([ slower, ] faster)">
          {REPLAY_SPEEDS.map((v) => (
            <button key={v} className={Math.abs(r.speed - v) < 1e-6 ? "active" : ""} onClick={() => cmd("replay_options", { speed: v })}>{v}×</button>
          ))}
        </div>
        <button className={`icon-btn ${r.loop ? "on" : ""}`} title={r.loop ? "Looping: on" : "Loop: off"} onClick={() => cmd("replay_options", { loop: !r.loop })}>⟳</button>
        <button className="btn small takeover" title="Start a new take from this exact frame and record your correction (T)"
          onClick={() => cmd("replay_branch")}>✋ Take over here</button>
        <button className="btn ghost small close" title="Close the replay and go back to a live scene" onClick={() => cmd("replay_stop")}>✕ Close</button>
      </div>
    </div>
  );
}
