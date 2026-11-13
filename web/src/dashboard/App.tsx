import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "../net";
import { drawGoal, levelStates } from "../goal";
import { type AuditRows, type DatasetRow, type EpisodeEdit, type EpisodeRow, fetchTask, type ServerEvent, SRC_CONTROLLER, SRC_HAND, type Status, type TaskSpec } from "../protocol";
import { EpisodeBrowser, epNumber, REPLAY_SPEEDS, ReplayBar } from "./Episodes";
import { type Cmd, fmt } from "./util";
import { Viewer3D, type ViewerHandle } from "./Viewer3D";

type Toast = { id: number; text: string; kind: string };
type Tab = "record" | "episodes";
const TAB_KEY = "qc.dash.tab";
const DATASET_KEY = "qc.dash.dataset";

function savedTab(): Tab {
  try { return localStorage.getItem(TAB_KEY) === "episodes" ? "episodes" : "record"; } catch { return "record"; }
}

export function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [online, setOnline] = useState(false);
  const [episodes, setEpisodes] = useState<EpisodeRow[]>([]);
  const [audit, setAudit] = useState<AuditRows>({});
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [task, setTask] = useState<TaskSpec | null>(null);
  const [tab, setTab] = useState<Tab>(savedTab);
  const [selFile, setSelFile] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [dataset, setDataset] = useState<string>(() => { try { return localStorage.getItem(DATASET_KEY) ?? ""; } catch { return ""; } });
  const viewer = useRef<ViewerHandle | null>(null);
  const link = useRef<Link | null>(null);
  const statusRef = useRef<Status | null>(null);
  statusRef.current = status;

  // "" = the recording folder. A remembered folder that is gone falls back to it.
  const ds = datasets.some((d) => d.name === dataset) ? dataset : (datasets.find((d) => d.recording)?.name ?? "");
  const q = ds ? `?dataset=${encodeURIComponent(ds)}` : "";
  const refreshEpisodes = useCallback(() => {
    fetch("/api/datasets").then((r) => r.json()).then(setDatasets).catch(() => {});
    fetch(`/api/episodes${q}`).then((r) => r.json()).then(setEpisodes).catch(() => {});
    fetch(`/api/audit${q}`).then((r) => r.json()).then(setAudit).catch(() => {});
  }, [q]);
  const pickDataset = useCallback((name: string) => {
    setEpisodes([]); setAudit({}); setSelFile(null);
    setDataset(name);
    try { localStorage.setItem(DATASET_KEY, name); } catch { /* private window */ }
  }, []);

  const toast = useCallback((text: string, kind = "info") => {
    const id = Math.random();
    setToasts((t) => [...t, { id, text, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2800);
  }, []);

  useEffect(() => {
    const l = new Link("/ws/dash", {
      // (Re)load the 3D scene on every connect, so a restarted server with a changed scene is picked up.
      onOpen: () => { setOnline(true); viewer.current?.reload(); fetchTask().then(setTask).catch(() => {}); },
      onClose: () => setOnline(false),
      onPose: (f) => viewer.current?.push(f),
      onStatus: (s) => { setStatus(s); viewer.current?.status(s); },
      onEvent: (e: ServerEvent) => {
        if (e.type === "toast") toast(e.text, e.kind);
        if (e.type === "saved") setTimeout(refreshEpisodes, 600);
      },
    });
    link.current = l;
    return () => l.close();
  }, [refreshEpisodes, toast]);
  useEffect(refreshEpisodes, [refreshEpisodes]);

  const cmd: Cmd = useCallback((name, args = {}) => link.current?.command(name, args), []);

  // Newest first, as listed on the Episodes tab.
  const rows = episodes.slice().reverse();
  const selected = rows.find((e) => e.file === selFile) ?? null;

  const switchTab = useCallback((t: Tab) => {
    setTab(t);
    try { localStorage.setItem(TAB_KEY, t); } catch { /* private window */ }
  }, []);

  // Episodes tab: the selected episode is loaded as a paused replay; leaving the tab goes back to the live scene.
  useEffect(() => {
    if (tab === "record" && statusRef.current?.replay) cmd("replay_stop");
  }, [tab, cmd]);
  useEffect(() => {
    if (tab === "episodes" && !selected && rows.length) setSelFile(rows[0].file);
  }, [tab, selected, rows]);
  const idle = status?.rec_state === "idle";  // also (re)loads once the first status arrives or a take ends
  // A take started from here or the headset (e.g. "Take over here"): its controls are on the Record tab.
  useEffect(() => {
    if (tab === "episodes" && status && !idle) switchTab("record");
  }, [tab, idle, status != null, switchTab]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (tab === "episodes" && selected && idle) cmd("replay", { path: selected.path, paused: true });
  }, [tab, selected?.path, idle, cmd]); // eslint-disable-line react-hooks/exhaustive-deps

  const edit = useCallback(async (e: EpisodeRow, body: EpisodeEdit) => {
    const r = await fetch(`/api/episodes/${encodeURIComponent(e.file)}${q}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }).catch(() => null);
    if (!r?.ok) toast("Couldn't save the change", "warn");
    refreshEpisodes();
  }, [refreshEpisodes, toast, q]);

  const remove = useCallback(async (e: EpisodeRow) => {
    if (!confirm(`Delete episode ${epNumber(e.file)} (${e.file})? This can't be undone.`)) return;
    const r = await fetch(`/api/episodes/${encodeURIComponent(e.file)}`, { method: "DELETE" }).catch(() => null);
    if (!r?.ok) { toast("Couldn't delete the episode", "warn"); return; }
    const k = rows.findIndex((x) => x.file === e.file);
    setSelFile((rows[k + 1] ?? rows[k - 1])?.file ?? null);
    toast(`Deleted ${epNumber(e.file)}`, "warn");
    refreshEpisodes();
  }, [rows, refreshEpisodes, toast]);

  // Keyboard shortcuts (per tab).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || e.ctrlKey || e.metaKey || e.altKey) return;
      if (tab === "record") {
        if (e.code === "Space") { e.preventDefault(); cmd("record_toggle"); }
        else if (e.key === "s") cmd("save", { success: true });
        else if (e.key === "f") cmd("save", { success: false });
        else if (e.key === "d" || e.key === "Backspace") cmd("discard");
        else if (e.key === "r") cmd("reset");
        else if (e.key === "c") cmd("recenter");
        return;
      }
      const r = statusRef.current?.replay;
      const fps = (selected && audit[selected.file]?.fps) || 50;
      if (e.code === "Space" && selected) {
        e.preventDefault();
        if (r) cmd("replay_pause"); else cmd("replay", { path: selected.path });
      } else if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && r) {
        e.preventDefault();
        cmd("replay_seek", { i: r.i + (e.key === "ArrowLeft" ? -1 : 1) * fps * (e.shiftKey ? 5 : 1) });
      } else if ((e.key === "," || e.key === ".") && r) {
        if (!r.paused) cmd("replay_pause", { paused: true });
        cmd("replay_seek", { i: r.i + (e.key === "," ? -1 : 1) });
      } else if (e.key === "t" && r) {
        cmd("replay_branch");
      } else if ((e.key === "[" || e.key === "]") && r) {
        const k = REPLAY_SPEEDS.findIndex((v) => v >= r.speed - 1e-6);
        const next = REPLAY_SPEEDS[Math.min(REPLAY_SPEEDS.length - 1, Math.max(0, (k < 0 ? REPLAY_SPEEDS.length - 1 : k) + (e.key === "[" ? -1 : 1)))];
        cmd("replay_options", { speed: next });
      } else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && rows.length) {
        e.preventDefault();
        const k = rows.findIndex((x) => x.file === selFile);
        setSelFile(rows[Math.min(rows.length - 1, Math.max(0, k + (e.key === "ArrowUp" ? -1 : 1)))].file);
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [cmd, tab, selected, selFile, rows, audit]);

  return (
    <div className="app">
      <TopBar status={status} online={online} tab={tab} setTab={switchTab}
        count={datasets.find((d) => d.recording)?.count ?? episodes.length} />
      <main className={`grid ${tab}`}>
        <section className="col-main">
          <div className="card viewer-card">
            <Viewer3D handle={viewer} />
            <div className="legend"><i className="dot l" /> left <i className="dot r" /> right</div>
            {tab === "record" && <Warnings status={status} />}
            <StateOverlay status={status} />
          </div>
          {tab === "episodes" && <ReplayBar status={status} ep={selected} audit={audit} task={task} cmd={cmd} />}
          <Cameras names={["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]} />
        </section>
        <aside className="col-side">
          {tab === "record" ? (
            <>
              <RecorderCard status={status} online={online} cmd={cmd} />
              <GoalCard task={task} status={status} />
              <TeleopCard status={status} cmd={cmd} />
            </>
          ) : (
            <EpisodeBrowser episodes={episodes} audit={audit} selected={selected} onSelect={(e) => setSelFile(e.file)}
              onEdit={edit} onDelete={remove} task={task} goRecord={() => switchTab("record")}
              datasets={datasets} dataset={ds} onDataset={pickDataset} />
          )}
        </aside>
      </main>
      <div className="toasts">{toasts.map((t) => <div key={t.id} className={`toast ${t.kind}`}>{t.text}</div>)}</div>
    </div>
  );
}

function TopBar({ status, online, tab, setTab, count }: {
  status: Status | null; online: boolean; tab: Tab; setTab: (t: Tab) => void; count: number;
}) {
  const quest = !!status?.connected;
  const recording = status?.rec_state === "recording" || status?.rec_state === "countdown";
  const health = online ? `Sim ${Math.round(status?.loop_hz ?? 0)} Hz` + (quest && status?.rtt_ms != null ? ` · ${Math.round(status.rtt_ms)} ms` : "") : "";
  return (
    <header className="topbar">
      <div className="brand"><span className="logo">◆</span> QuestCollect</div>
      <nav className="tabs">
        <button className={tab === "record" ? "active" : ""} onClick={() => setTab("record")}>
          <i className={`rec-dot ${recording ? "on" : ""}`} /> Record
        </button>
        <button className={tab === "episodes" ? "active" : ""} onClick={() => setTab("episodes")}>
          Episodes <small>{count}</small>
        </button>
      </nav>
      <div className="chips">
        <span className={`chip ${online ? "ok" : "bad"}`} title="The QuestCollect server (qc serve)"><i />{online ? "Server" : "Server offline"}</span>
        <span className={`chip ${quest ? "ok" : "warn"}`} title={quest ? `Input ${Math.round(status?.input_hz ?? 0)} Hz` : "Open the server's address in the Quest Browser and press Enter VR"}>
          <i />{quest ? "Headset" : "No headset"}
        </span>
        {health && <span className={`chip plain ${status && status.loop_hz < 45 ? "warn" : ""}`}>{health}</span>}
      </div>
    </header>
  );
}

function GoalCard({ task, status }: { task: TaskSpec | null; status: Status | null }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const states = task ? levelStates(task, status?.progress ?? 0, status?.in_area ?? false) : [];
  const key = states.join();
  useEffect(() => {
    const c = canvas.current;
    if (!c || !task) return;
    const dpr = devicePixelRatio || 1;
    c.width = c.clientWidth * dpr;
    c.height = c.clientHeight * dpr;
    const g = c.getContext("2d")!;
    g.clearRect(0, 0, c.width, c.height);
    drawGoal(g, task, states, { x: 0, y: 0, w: c.width, h: c.height }, 1);
  }, [task, key]); // eslint-disable-line react-hooks/exhaustive-deps
  const done = states.filter((s) => s === "placed").length;
  return (
    <div className="card goal">
      <div className="card-head"><h2>Goal</h2><span className="muted">{task ? `${done} / ${task.goal.length} placed` : ""}</span></div>
      <div className="goal-body">
        <canvas ref={canvas} />
        <ol>
          {task?.goal.map((e, k) => {
            const b = task.bricks[e.brick];
            const [r, g, bl] = b.rgba.map((v) => Math.round(v * 255));
            return (
              <li key={k} className={states[k]}>
                <i style={{ background: `rgb(${r},${g},${bl})` }} />
                <span>{b.label}{k === 0 && <em> in the taped area</em>}</span>
                <b>{states[k] === "placed" ? "✓" : states[k] === "next" ? "next" : ""}</b>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}

function StateOverlay({ status }: { status: Status | null }) {
  if (!status) return null;
  const st = status.replay ? "replay" : status.rec_state;
  const label = status.replay ? `REPLAY${status.replay.speed !== 1 ? ` ${status.replay.speed}×` : ""}${status.replay.paused ? " · paused" : ""}`
    : st === "recording" ? `REC ${fmt(status.elapsed)}`
    : st === "countdown" ? `${Math.ceil(status.countdown)}`
    : st === "review" ? (status.success_seen ? "SUCCESS · review" : "REVIEW") : "LIVE";
  return (
    <div className={`state-overlay ${st}`}>
      <i />{label}
      {status.success && st === "recording" && <span className="succ">✓ task solved</span>}
    </div>
  );
}

function Warnings({ status }: { status: Status | null }) {
  const w = status?.warnings;
  const on = [w?.arm_collision && "Arms touching", w?.table_collision && "Arm on the table", w?.ik_saturated && "Target out of reach"].filter(Boolean);
  return on.length ? <div className="warn-overlay">{on.map((t) => <span key={t as string}>⚠ {t}</span>)}</div> : null;
}

function Cameras({ names }: { names: string[] }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 150);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="cams">
      {names.map((n) => (
        <figure key={n} className="card cam">
          <img src={`/api/preview/${n}.jpg?t=${tick}`} alt={n} onError={(e) => ((e.target as HTMLImageElement).style.opacity = "0.15")}
            onLoad={(e) => ((e.target as HTMLImageElement).style.opacity = "1")} />
          <figcaption>{n.replace("cam_", "").replace("_", " ")}</figcaption>
        </figure>
      ))}
    </div>
  );
}

/** What to do next, in one line. */
function nextStep(s: Status | null, online: boolean): string {
  if (!online || !s) return "The server is offline. Start it with: uv run qc serve";
  if (s.rec_state === "countdown") return s.branch ? "Get ready: the scene holds at the frame you took over at…" : "Get ready…";
  if (s.rec_state === "recording" && s.branch) return "Show the correction from here: finish the tower, then save it.";
  if (s.rec_state === "recording") {
    return s.auto_stop ? "Build the tower. Recording stops by itself 1 s after it's finished." : "Build the tower, then press Stop.";
  }
  if (s.rec_state === "review") return "Keep this take? Save it as a success or failure, or discard it.";
  if (s.replay) return "The scene shows a replay. Take over here (T) continues it as a new take; Start recording goes back to a fresh scene.";
  if (!s.connected) return "Put on the Quest, open this server's address in the Quest Browser and press Enter VR.";
  return "Press Start (or Space). In the headset, squeeze Grip to take hold of an arm.";
}

function RecorderCard({ status, online, cmd }: { status: Status | null; online: boolean; cmd: Cmd }) {
  const s = status;
  const rec = s?.rec_state ?? "idle";
  const [operator, setOperator] = useState("");
  useEffect(() => { if (s && !operator) setOperator(s.operator); }, [s, operator]);
  const review = s?.review;
  return (
    <div className={`card recorder ${rec}`}>
      <div className="card-head"><h2>Recorder</h2><span className="muted">{s?.saved_count ?? 0} saved this session</span></div>
      <button className={`rec-btn ${rec}`} disabled={!online || rec === "review"} onClick={() => cmd("record_toggle")}>
        <i />
        <span>{rec === "recording" ? `Stop · ${fmt(s!.elapsed)}` : rec === "countdown" ? `Starting in ${Math.ceil(s!.countdown)}… (cancel)` : rec === "review" ? "Reviewing take" : "Start recording"}</span>
        <kbd>Space</kbd>
      </button>
      <p className="hint">
        {nextStep(s, online)}
        {s?.replay && rec === "idle" && <button className="link" onClick={() => cmd("replay_stop")}>Back to live</button>}
      </p>
      {review && (
        <div className={`review ${review.rejected ? "warn" : "ok"}`}>
          <b>{review.success ? "✓ Tower built" : "✗ Tower not finished"}</b> · {review.duration_s.toFixed(1)} s · idle {review.idle_pct}%
          <div>{review.rejected ? `Training would skip it: ${review.rejected}` : `Training would use ${review.used_s.toFixed(1)} s of it`}</div>
        </div>
      )}
      {(rec === "review" || rec === "recording") && (
        <div className="row3">
          <button className="btn ok" onClick={() => cmd("save", { success: true })}>✓ Success <kbd>S</kbd></button>
          <button className="btn warn" onClick={() => cmd("save", { success: false })}>✗ Failure <kbd>F</kbd></button>
          <button className="btn bad" onClick={() => cmd("discard")}>Discard <kbd>D</kbd></button>
        </div>
      )}
      <div className="rec-foot">
        <input className="operator" title="Operator name, saved in each episode" placeholder="Operator name" value={operator}
          onChange={(e) => setOperator(e.target.value)} onBlur={() => cmd("set_operator", { operator })} />
        <label className="toggle" title="Stop recording 1 s after the tower is finished"><input type="checkbox" checked={s?.auto_stop ?? true} onChange={(e) => cmd("set_auto_stop", { enabled: e.target.checked })} /> Auto-stop</label>
        <button className="btn ghost small" disabled={rec !== "idle"} title="New random brick layout" onClick={() => cmd("reset")}>↺ New scene <kbd>R</kbd></button>
      </div>
    </div>
  );
}

function TeleopCard({ status, cmd }: { status: Status | null; cmd: Cmd }) {
  const s = status;
  return (
    <div className="card">
      <div className="card-head"><h2>Arms</h2>
        <button className="btn ghost small" title="Re-place the table in front of where you stand" onClick={() => cmd("recenter")}>⟲ Recenter <kbd>C</kbd></button>
      </div>
      <div className="arms">
        <ArmTile side="l" status={s} />
        <ArmTile side="r" status={s} />
      </div>
      {s?.mode_mismatch && <div className="alert warn">Input doesn't match the mode: {s.mode === "hands" ? "put the controllers down" : "pick up the controllers"}.</div>}
      {s?.control.policy && (
        <>
          <div className="seg three">
            {(["teleop", "shared", "policy"] as const).map((m) => (
              <button key={m} className={s.control.mode === m ? "active" : ""} onClick={() => cmd("set_control_mode", { mode: m })}>
                {{ teleop: "🧑 You drive", shared: "🤝 Shared", policy: "🤖 Policy" }[m]}
              </button>
            ))}
          </div>
          <div className="muted control-note">
            Policy <b>{s.control.policy}</b> · driving now: <b>{s.control.driver === "policy" ? "policy" : "you"}</b>
            {s.control.driver === "policy" && <> · safety Δ {s.control.safety_rad.toFixed(3)} rad</>}
            {s.control.mode === "shared" && <> · press Grip to take over</>}
          </div>
        </>
      )}
      <details className="more">
        <summary>Input {s?.mode === "hands" ? "✋ hands" : "🎮 controllers"} · motion ×{(s?.scale ?? 1).toFixed(2)}</summary>
        <div className="seg">
          <button className={s?.mode === "controllers" ? "active" : ""} onClick={() => cmd("set_mode", { mode: "controllers" })}>🎮 Controllers</button>
          <button className={s?.mode === "hands" ? "active" : ""} onClick={() => cmd("set_mode", { mode: "hands" })}>✋ Hand tracking</button>
        </div>
        <label className="slider">Motion scale <b>×{(s?.scale ?? 1).toFixed(2)}</b>
          <input type="range" min={0.25} max={3} step={0.05} value={s?.scale ?? 1} onChange={(e) => cmd("set_scale", { scale: +e.target.value })} />
        </label>
      </details>
    </div>
  );
}

function ArmTile({ side, status }: { side: "l" | "r"; status: Status | null }) {
  const s = status;
  const tracked = s?.tracked[side], engaged = s?.engaged[side];
  const src = s?.sources[side];
  const grip = s?.grippers[side === "l" ? 0 : 1] ?? 1;
  const held = s?.held[side];
  return (
    <div className={`arm ${side} ${engaged ? "engaged" : ""} ${tracked ? "tracked" : ""}`}>
      <div className="arm-head">
        <b>{side === "l" ? "Left" : "Right"}</b>
        <span>{src === SRC_HAND ? "✋" : src === SRC_CONTROLLER ? "🎮" : ""} {engaged ? "moving" : tracked ? "tracked" : "not tracked"}</span>
      </div>
      <div className="meter" title={`Gripper ${Math.round(grip * 100)}% open`}><div style={{ width: `${grip * 100}%` }} /></div>
      <div className="arm-foot"><span>{held ? `holding ${held.replace(/_/g, " ")}` : `gripper ${Math.round(grip * 100)}%`}</span><span title="IK tracking error">{s?.ik_error_mm[side] ?? 0} mm</span></div>
    </div>
  );
}
