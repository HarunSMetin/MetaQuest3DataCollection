import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "../net";
import { bricksPlaced, drawGoal, levelStates } from "../goal";
import { type AuditRows, type EpisodeRow, fetchTask, type ServerEvent, SRC_CONTROLLER, SRC_HAND, type Status, type TaskSpec } from "../protocol";
import { Viewer3D, type ViewerHandle } from "./Viewer3D";

type Toast = { id: number; text: string; kind: string };

const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;

export function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [online, setOnline] = useState(false);
  const [episodes, setEpisodes] = useState<EpisodeRow[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [task, setTask] = useState<TaskSpec | null>(null);
  const viewer = useRef<ViewerHandle | null>(null);
  const link = useRef<Link | null>(null);

  const refreshEpisodes = useCallback(() => {
    fetch("/api/episodes").then((r) => r.json()).then(setEpisodes).catch(() => {});
  }, []);

  const toast = useCallback((text: string, kind: string) => {
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
    refreshEpisodes();
    return () => l.close();
  }, [refreshEpisodes, toast]);

  const cmd = useCallback((name: string, args: Record<string, unknown> = {}) => link.current?.command(name, args), []);

  // Keyboard shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
      if (e.code === "Space") { e.preventDefault(); cmd("record_toggle"); }
      else if (e.key === "s") cmd("save", { success: true });
      else if (e.key === "f") cmd("save", { success: false });
      else if (e.key === "d" || e.key === "Backspace") cmd("discard");
      else if (e.key === "r") cmd("reset");
      else if (e.key === "c") cmd("recenter");
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [cmd]);

  return (
    <div className="app">
      <TopBar status={status} online={online} />
      <main className="grid">
        <section className="col-main">
          <div className="card viewer-card">
            <div className="card-head">
              <h2>Live scene</h2>
              <div className="legend"><i className="dot l" /> left arm <i className="dot r" /> right arm <span className="muted">· dot = grasp point (red: out of reach)</span></div>
            </div>
            <Viewer3D handle={viewer} />
            <StateOverlay status={status} />
          </div>
          <Cameras names={["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]} />
        </section>
        <aside className="col-side">
          <GoalCard task={task} status={status} />
          <RecorderCard status={status} cmd={cmd} />
          <TeleopCard status={status} cmd={cmd} />
          <EpisodesCard episodes={episodes} status={status} cmd={cmd} refresh={refreshEpisodes} />
        </aside>
      </main>
      <div className="toasts">{toasts.map((t) => <div key={t.id} className={`toast ${t.kind}`}>{t.text}</div>)}</div>
    </div>
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

function TopBar({ status, online }: { status: Status | null; online: boolean }) {
  const quest = status?.connected;
  return (
    <header className="topbar">
      <div className="brand"><span className="logo">◆</span> QuestCollect <span className="muted">· bimanual Franka · MuJoCo</span></div>
      <div className="chips">
        <Chip label="Server" value={online ? "online" : "offline"} tone={online ? "ok" : "bad"} />
        <Chip label="Quest" value={quest ? "streaming" : "not connected"} tone={quest ? "ok" : "warn"} />
        <Chip label="RTT" value={status?.rtt_ms != null ? `${Math.round(status.rtt_ms)} ms` : "—"} tone={status?.rtt_ms != null && status.rtt_ms > 60 ? "warn" : "neutral"} />
        <Chip label="Input" value={`${Math.round(status?.input_hz ?? 0)} Hz`} />
        <Chip label="Sim" value={`${Math.round(status?.loop_hz ?? 0)} Hz`} tone={status && status.loop_hz < 45 ? "warn" : "neutral"} />
      </div>
    </header>
  );
}

function Chip({ label, value, tone = "neutral" }: { label: string; value: string; tone?: string }) {
  return <div className={`chip ${tone}`}><span>{label}</span><b>{value}</b></div>;
}

function StateOverlay({ status }: { status: Status | null }) {
  if (!status) return null;
  const st = status.replay ? "replay" : status.rec_state;
  const label = status.replay ? `REPLAY ${status.replay.i}/${status.replay.n}`
    : st === "recording" ? `REC ${fmt(status.elapsed)}`
    : st === "countdown" ? `${Math.ceil(status.countdown)}`
    : st === "review" ? (status.success_seen ? "SUCCESS · review" : "REVIEW") : "READY";
  return (
    <div className={`state-overlay ${st}`}>
      <i />{label}
      {status.success && st === "recording" && <span className="succ">✓ task solved</span>}
    </div>
  );
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
          <figcaption>{n}</figcaption>
        </figure>
      ))}
    </div>
  );
}

function RecorderCard({ status, cmd }: { status: Status | null; cmd: (n: string, a?: Record<string, unknown>) => void }) {
  const s = status;
  const rec = s?.rec_state ?? "idle";
  const [operator, setOperator] = useState("");
  useEffect(() => { if (s && !operator) setOperator(s.operator); }, [s, operator]);
  return (
    <div className="card">
      <div className="card-head"><h2>Recorder</h2><span className="muted">{s?.saved_count ?? 0} saved this session</span></div>
      <label>Operator
        <input value={operator} onChange={(e) => setOperator(e.target.value)} onBlur={() => cmd("set_operator", { operator })} />
      </label>
      <p className="instruction">{s?.instruction}{s && <><br /><b>Tower: {bricksPlaced(s.progress, s.in_area)} / {s.levels + 1} bricks placed</b></>}</p>
      <button className={`rec-btn ${rec}`} disabled={rec === "review" || !!s?.replay} onClick={() => cmd("record_toggle")}>
        <i />
        <span>{rec === "recording" ? `Stop · ${fmt(s!.elapsed)}` : rec === "countdown" ? `Starting in ${Math.ceil(s!.countdown)}…` : rec === "review" ? "Review take" : "Start recording"}</span>
        <kbd>Space</kbd>
      </button>
      <div className="row3">
        <button className="btn ok" disabled={rec !== "review" && rec !== "recording"} onClick={() => cmd("save", { success: true })}>✓ Save success <kbd>S</kbd></button>
        <button className="btn warn" disabled={rec !== "review" && rec !== "recording"} onClick={() => cmd("save", { success: false })}>✗ Save failure <kbd>F</kbd></button>
        <button className="btn bad" disabled={rec === "idle"} onClick={() => cmd("discard")}>Discard <kbd>D</kbd></button>
      </div>
      <div className="row-between">
        <label className="toggle"><input type="checkbox" checked={s?.auto_stop ?? true} onChange={(e) => cmd("set_auto_stop", { enabled: e.target.checked })} /> Auto-stop 1 s after success</label>
        <button className="btn ghost" disabled={rec !== "idle"} onClick={() => cmd("reset")}>↺ New scene <kbd>R</kbd></button>
      </div>
    </div>
  );
}

function TeleopCard({ status, cmd }: { status: Status | null; cmd: (n: string, a?: Record<string, unknown>) => void }) {
  const s = status;
  return (
    <div className="card">
      <div className="card-head"><h2>Teleoperation</h2>
        <button className="btn ghost small" onClick={() => cmd("recenter")}>⟲ Recenter <kbd>C</kbd></button>
      </div>
      <div className="seg">
        <button className={s?.mode === "controllers" ? "active" : ""} onClick={() => cmd("set_mode", { mode: "controllers" })}>🎮 Controllers</button>
        <button className={s?.mode === "hands" ? "active" : ""} onClick={() => cmd("set_mode", { mode: "hands" })}>✋ Hand tracking</button>
      </div>
      {s?.mode_mismatch && <div className="alert warn">Input source doesn't match the mode — {s.mode === "hands" ? "put the controllers down" : "pick up the controllers"}.</div>}
      {s?.control.policy && (
        <>
          <div className="seg">
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
      <div className="arms">
        <ArmTile side="l" status={s} />
        <ArmTile side="r" status={s} />
      </div>
      <label className="slider">Motion scale <b>×{(s?.scale ?? 1).toFixed(2)}</b>
        <input type="range" min={0.25} max={3} step={0.05} value={s?.scale ?? 1} onChange={(e) => cmd("set_scale", { scale: +e.target.value })} />
      </label>
      <div className="warnings">
        <Warn on={!!s?.warnings.arm_collision} text="Arm–arm contact" />
        <Warn on={!!s?.warnings.table_collision} text="Table contact" />
        <Warn on={!!s?.warnings.ik_saturated} text="Target out of reach" />
      </div>
    </div>
  );
}

function ArmTile({ side, status }: { side: "l" | "r"; status: Status | null }) {
  const s = status;
  const tracked = s?.tracked[side], engaged = s?.engaged[side];
  const src = s?.sources[side];
  const grip = s?.grippers[side === "l" ? 0 : 1] ?? 1;
  return (
    <div className={`arm ${side} ${engaged ? "engaged" : ""} ${tracked ? "tracked" : ""}`}>
      <div className="arm-head">
        <b>{side === "l" ? "Left" : "Right"}</b>
        <span>{src === SRC_HAND ? "✋ hand" : src === SRC_CONTROLLER ? "🎮 controller" : "∅ none"}</span>
      </div>
      <div className="arm-state">{engaged ? "ENGAGED" : tracked ? "tracked" : "not tracked"}</div>
      <div className="meter"><div style={{ width: `${grip * 100}%` }} /></div>
      <div className="arm-foot"><span>gripper {Math.round(grip * 100)}%</span><span>IK {s?.ik_error_mm[side] ?? 0} mm</span></div>
    </div>
  );
}

function Warn({ on, text }: { on: boolean; text: string }) {
  return <span className={`warn-chip ${on ? "on" : ""}`}>{text}</span>;
}

function EpisodesCard({ episodes, status, cmd, refresh }: {
  episodes: EpisodeRow[]; status: Status | null; cmd: (n: string, a?: Record<string, unknown>) => void; refresh: () => void;
}) {
  const rows = episodes.slice().reverse();
  const ok = rows.filter((e) => e.success).length;
  const total = rows.reduce((a, e) => a + (e.duration_s || 0), 0);
  const [audit, setAudit] = useState<AuditRows>({});
  const loadAudit = useCallback(() => {
    fetch("/api/audit").then((r) => r.json()).then(setAudit).catch(() => {});
  }, []);
  useEffect(loadAudit, [episodes.length, loadAudit]);
  const used = rows.filter((e) => audit[e.file] && !audit[e.file].verdict).length;
  const del = async (p: string) => {
    if (!confirm(`Delete ${p.split("/").slice(-2).join("/")}?`)) return;
    await fetch(`/api/episodes?path=${encodeURIComponent(p)}`, { method: "DELETE" });
    refresh();
  };
  const toggleExclude = async (file: string) => {
    await fetch("/api/curation", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file, exclude: !audit[file]?.excluded, note: "excluded on the dashboard" }),
    });
    loadAudit();
  };
  return (
    <div className="card episodes">
      <div className="card-head"><h2>Dataset</h2><span className="muted">{status?.task.replace(/_/g, " ")}</span></div>
      <div className="stats">
        <div><b>{rows.length}</b><span>episodes</span></div>
        <div><b>{rows.length ? Math.round((100 * ok) / rows.length) : 0}%</b><span>success</span></div>
        <div><b>{fmt(total)}</b><span>total time</span></div>
        <div title="Episodes a training dataset would use (qc audit)"><b>{used}</b><span>for training</span></div>
      </div>
      <div className="table">
        {rows.length === 0 && <div className="empty">No episodes yet — record your first demo.</div>}
        {rows.map((e) => (
          <div key={e.path} className="tr">
            <span className={`badge ${e.success ? "ok" : "warn"}`}>{e.success ? "✓" : "✗"}</span>
            <div className="ep-main">
              <b>{e.file.replace("episode_", "#").replace(".hdf5", "")}</b>
              <span className="muted">
                {e.source === "policy" ? "🤖" : e.input_mode === "hands" ? "✋" : "🎮"} · {e.duration_s?.toFixed(1)} s
                {audit[e.file] && <> · idle {Math.round(audit[e.file].idle_frac * 100)}%</>}
              </span>
            </div>
            {audit[e.file]?.verdict
              ? <span className="badge warn" title={audit[e.file].verdict ?? ""}>⚠</span>
              : audit[e.file] && <span className="badge ok" title="Used for training">✓</span>}
            <button className="icon" title={audit[e.file]?.excluded ? "Use for training again" : "Exclude from training"}
              onClick={() => toggleExclude(e.file)}>{audit[e.file]?.excluded ? "↺" : "⊘"}</button>
            <button className="icon" title="Replay in sim" disabled={status?.rec_state !== "idle"} onClick={() => cmd("replay", { path: e.path })}>▶</button>
            <button className="icon danger" title="Delete" onClick={() => del(e.path)}>🗑</button>
          </div>
        ))}
      </div>
      {status?.replay && <button className="btn ghost wide" onClick={() => cmd("replay_stop")}>■ Stop replay</button>}
    </div>
  );
}
