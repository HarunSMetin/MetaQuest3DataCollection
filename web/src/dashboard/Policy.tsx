import { useEffect, useState } from "react";
import type { PolicyRun, Status } from "../protocol";
import type { Cmd } from "./util";

/** Policy tab: load a trained run (any checkpoint), run it on a fresh or an eval layout, take over, record. */

const CHUNKS: (number | null)[] = [null, 1, 4, 8, 10, 16, 32];
const STAGE_SHORT: Record<string, string> = { base_in_area: "base", level_1: "L1", level_2: "L2", level_3: "L3" };
const pct = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const stepLabel = (s: string) => `${Math.round(+s / 1000)}k`;

/** What the policy reads, in a few words. */
function inputs(f: PolicyRun["features"]): string[] {
  const out = [(f.cameras ?? []).length ? (f.cameras ?? []).map((c) => c.replace("cam_", "").replace("_wrist", " wrist")).join(" + ") : "no cameras"];
  if (f.env_state) out.push("brick poses (sim only)");
  if (f.joint_vel) out.push("joint velocity");
  if (f.ee_wrench) out.push("hand force");
  if (f.action === "ee") out.push("TCP actions");
  return out;
}

/** The loaded policy and who drives: run / stop / shared, new scene or an eval layout. */
export function PolicyControl({ status, cmd }: { status: Status | null; cmd: Cmd }) {
  const c = status?.control;
  const [evalIdx, setEvalIdx] = useState(0);
  const rec = status?.rec_state ?? "idle";
  const busy = rec === "recording" || rec === "countdown";
  return (
    <div className="card policy-ctl">
      <div className="card-head"><h2>Policy</h2>
        {c?.policy && <button className="btn ghost small" title="Free the GPU and drive by hand again" onClick={() => cmd("unload_policy")}>⏏ Unload</button>}
      </div>
      {c?.loading && <div className="alert info"><span className="spin" /> Loading {c.loading.replace(/.*outputs\//, "")}…</div>}
      {c?.error && !c.loading && <div className="alert warn">Couldn't load: {c.error}</div>}
      {!c?.policy ? (
        !c?.loading && <p className="muted">No policy loaded. Pick a run and a checkpoint below and press <b>Load</b>.</p>
      ) : (
        <>
          <div className="loaded">
            <b>{c.policy}</b>
            <span className="muted small">{checkpointOf(c.path)} · {c.n_action_steps ?? "?"} steps per chunk</span>
          </div>
          <div className="seg three">
            {(["policy", "shared", "teleop"] as const).map((m) => (
              <button key={m} className={c.mode === m ? "active" : ""} onClick={() => cmd("set_control_mode", { mode: m })}
                title={{ policy: "The policy drives both arms", shared: "The policy drives until you press Grip (DAgger)", teleop: "You drive; the policy waits" }[m]}>
                {{ policy: "▶ Run", shared: "🤝 Shared", teleop: "■ Stop" }[m]}
              </button>
            ))}
          </div>
          <div className="policy-live">
            <span>driving: <b className={c.driver === "policy" ? "acc" : ""}>{c.driver === "policy" ? "policy" : "you"}</b></span>
            <span>policy time <b>{(c.policy_s ?? 0).toFixed(1)} s</b></span>
            <span title="How much the safety filter changed the last joint targets">safety Δ <b>{c.safety_rad.toFixed(3)}</b> rad</span>
          </div>
        </>
      )}
      <div className="scene-row">
        <button className="btn small" disabled={busy} onClick={() => cmd("reset")} title="New random brick layout">↺ New scene</button>
        <span className="muted small">or eval layout #</span>
        <input type="number" min={0} max={999} value={evalIdx} onChange={(e) => setEvalIdx(Math.max(0, +e.target.value))} />
        <button className="btn small" disabled={busy} onClick={() => cmd("reset", { eval_index: evalIdx })}
          title="The start layout of `qc eval` rollout #i (the first n of the same seed list), to compare live with the saved rollout">Load</button>
      </div>
      <p className="muted small hint">The scene and the policy restart together. Every action goes through the safety filter. Record below to keep a rollout (or your takeover) as an episode.</p>
    </div>
  );
}

function checkpointOf(path: string | null | undefined): string {
  if (!path) return "";
  const m = path.match(/checkpoints\/([^/]+)\//);
  return m ? `checkpoint ${m[1]}` : "checkpoint last";
}

/** Every trained run with its checkpoints, inputs and evaluations; Load swaps the live policy. */
export function PolicyRuns({ runs, status, cmd, refresh }: { runs: PolicyRun[] | null; status: Status | null; cmd: Cmd; refresh: () => void }) {
  const [open, setOpen] = useState<string | null>(null);
  const loadedPath = status?.control.path ?? null;
  useEffect(() => { if (open === null && runs?.length) setOpen(runs[0].name); }, [runs, open]);
  return (
    <div className="card policy-runs">
      <div className="card-head"><h2>Trained runs</h2><button className="btn ghost small" onClick={refresh}>⟳</button></div>
      {runs === null && <p className="muted">Loading…</p>}
      {runs?.length === 0 && <p className="muted">No runs in outputs/ yet. Train one with <code>uv run qc train &lt;dataset&gt;</code>.</p>}
      <div className="runs">
        {runs?.map((r) => (
          <RunRow key={r.name} run={r} open={open === r.name} onToggle={() => setOpen(open === r.name ? "" : r.name)}
            loadedPath={loadedPath} loading={!!status?.control.loading} cmd={cmd} />
        ))}
      </div>
    </div>
  );
}

function RunRow({ run, open, onToggle, loadedPath, loading, cmd }: {
  run: PolicyRun; open: boolean; onToggle: () => void; loadedPath: string | null; loading: boolean; cmd: Cmd;
}) {
  const [step, setStep] = useState<string>("last");
  const [chunk, setChunk] = useState<number | null>(10);
  const ev = run.evals[0];
  const path = step === "last" ? run.path : `${run.path}/train/checkpoints/${step}/pretrained_model`;
  const isLoaded = loadedPath != null && (loadedPath === run.path || loadedPath.startsWith(`${run.path}/`));
  const usable = run.steps.length > 0;
  return (
    <div className={`run ${open ? "open" : ""} ${isLoaded ? "loaded" : ""}`}>
      <button className="run-head" onClick={onToggle}>
        <span className="run-name">{isLoaded && <i className="dot-live" />}{run.name}</span>
        <span className="muted small">{run.dataset} · {run.type ?? "?"} · {run.steps.length ? `${stepLabel(run.last ?? run.steps[run.steps.length - 1])} steps` : "no checkpoint"}</span>
        {ev && <span className="small ev">{Object.entries(ev.stage_rates).filter(([k]) => k !== "level_3").map(([k, v]) => `${STAGE_SHORT[k] ?? k} ${pct(v)}`).join(" · ")}</span>}
      </button>
      {open && (
        <div className="run-body">
          <div className="muted small">{inputs(run.features).join(" · ")} · preset {run.preset}</div>
          {run.features.env_state && <div className="alert warn small">Trained on brick poses with constant inputs: live results can differ from the pinned evaluation (review V9).</div>}
          {run.evals.length > 0 && (
            <table className="evals small">
              <thead><tr><th>eval</th><th>ckpt</th><th>n</th><th>base</th><th>L1</th><th>L2</th><th>done</th></tr></thead>
              <tbody>
                {run.evals.slice(0, 4).map((e) => (
                  <tr key={e.time}><td>{e.time.slice(4, 8)} {e.time.slice(9, 13)}</td><td>{e.checkpoint === "last" ? "last" : e.checkpoint ? stepLabel(e.checkpoint) : "?"}</td><td>{e.n}</td>
                    <td>{pct(e.stage_rates.base_in_area)}</td><td>{pct(e.stage_rates.level_1)}</td><td>{pct(e.stage_rates.level_2)}</td><td>{pct(e.success_rate)}</td></tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="load-row">
            <label>Checkpoint
              <select value={step} onChange={(e) => setStep(e.target.value)} disabled={!usable}>
                {run.last && <option value="last">last ({stepLabel(run.last)})</option>}
                {run.steps.slice().reverse().map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </label>
            <label title="Steps executed from each predicted chunk before the policy looks again ('as trained' = like qc eval)">Chunk
              <select value={chunk ?? ""} onChange={(e) => setChunk(e.target.value === "" ? null : +e.target.value)}>
                {CHUNKS.map((n) => <option key={n ?? "t"} value={n ?? ""}>{n ?? "as trained"}</option>)}
              </select>
            </label>
            <button className="btn small ok" disabled={!usable || loading} onClick={() => cmd("load_policy", { path, n_action_steps: chunk })}>
              {isLoaded && loadedPath === path ? "↻ Reload" : "Load"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
