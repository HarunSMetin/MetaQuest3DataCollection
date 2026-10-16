// End-to-end test without a headset: an emulated Quest 3 (Meta's IWER runtime) in headless Chromium
// drives the real WebXR page -> websocket -> teleop -> IK -> MuJoCo, and checks the server's status.
//
//   cd web/e2e && npm install && npx playwright install chromium && npm run build
//   uv run qc serve --no-https --config <config with a scratch record.out_dir>   # in another shell
//   npm test
//
// NOTE: it records a few short episodes into the server's record.out_dir.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
const HERE = import.meta.dirname;
const OUT = `${HERE}/out`;
mkdirSync(OUT, { recursive: true });
const BASE = process.env.QC_URL ?? "http://localhost:8000";
const status = async () => { const s = await (await fetch(`${BASE}/api/status`)).json(); lastHz = s.input_hz; return s; };
const cmd = (name, args = {}) => fetch(`${BASE}/api/command`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, args }) });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
let lastHz = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}   [input ${lastHz} Hz]`); if (!ok) failures++; };

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("404")) errors.push(m.text()); });
await page.addInitScript({ path: `${HERE}/iwer.bundle.js` });
await cmd("set_mode", { mode: "controllers" });
await page.goto(`${BASE}/xr/`);
await page.waitForFunction(() => !document.getElementById("enter-vr").disabled, null, { timeout: 15000 });
await page.evaluate(() => {
  const d = window.__xr;
  d.position.set(0, 1.6, 0);
  d.controllers.left.position.set(-0.2, 1.1, -0.35);
  d.controllers.right.position.set(0.2, 1.1, -0.35);
});
// Software GL can't render the Quest-3 scene at real-time rates; hide it so the emulated frame loop
// (and therefore the input stream) runs near headset rate. This test checks the input pipeline.
await page.evaluate(() => { const q = window.__qc; q.renderer.xr.setFramebufferScaleFactor(0.12); q.renderer.shadowMap.enabled = false; q.mj.root.visible = false; });
await page.click("#enter-vr");
await sleep(2500);

let s = await status();
check(s.sources.l === 1 && s.sources.r === 1, `controllers seen by server (sources=${JSON.stringify(s.sources)})`);
check(s.tracked.l && s.tracked.r, "both arms tracked");
check(s.input_hz > 20, `input stream rate ${s.input_hz} Hz`);
check(s.rtt_ms !== null && s.rtt_ms < 100, `server-measured RTT ${s.rtt_ms} ms`);

// Engage right arm with grip, move controller 10 cm right and 10 cm down.
const r0 = s.ee_pos.r;
await page.evaluate(() => window.__xr.controllers.right.updateButtonValue("squeeze", 1));
await sleep(400);
s = await status();
check(s.engaged.r && !s.engaged.l, "grip engages only the right arm");
for (let i = 1; i <= 20; i++) {
  await page.evaluate((k) => window.__xr.controllers.right.position.set(0.2 + 0.005 * k, 1.1 - 0.005 * k, -0.35), i);
  await sleep(40);
}
await sleep(1200);
s = await status();
const d = s.ee_pos.r.map((v, i) => v - r0[i]);
// XR +x (right) -> world -y ; XR -y (down) -> world -z
check(d[1] < -0.07 && d[2] < -0.07 && Math.abs(d[0]) < 0.03, `right EE followed controller: Δworld=${d.map((v) => v.toFixed(3))}`);
await page.evaluate(() => window.__xr.controllers.right.updateButtonValue("trigger", 1));
await sleep(600);
s = await status();
check(s.grippers[1] < 0.05, `trigger closes right gripper (cmd=${s.grippers[1]})`);
await page.evaluate(() => { const c = window.__xr.controllers.right; c.updateButtonValue("trigger", 0); c.updateButtonValue("squeeze", 0); });
await sleep(300);
s = await status();
check(!s.engaged.r, "releasing grip disengages");

// Record with A, stop with A, save with X.
const press = async (side, btn) => {
  await page.evaluate(([sd, b]) => window.__xr.controllers[sd].updateButtonValue(b, 1), [side, btn]);
  await sleep(150);
  await page.evaluate(([sd, b]) => window.__xr.controllers[sd].updateButtonValue(b, 0), [side, btn]);
  await sleep(150);
};
const before = (await (await fetch(`${BASE}/api/episodes`)).json()).length;
await press("right", "a-button");
await sleep(300);
check((await status()).rec_state === "countdown", "A starts countdown");
await sleep(3300);
check((await status()).rec_state === "recording", "countdown -> recording");
await sleep(1500);
await press("right", "a-button");
check((await status()).rec_state === "review", "A stops -> review");
await press("left", "x-button");
await sleep(1500);
const eps = await (await fetch(`${BASE}/api/episodes`)).json();
check(eps.length === before + 1, `X saves episode (${eps.length} total) -> ${eps.at(-1)?.file} ${eps.at(-1)?.duration_s}s mode=${eps.at(-1)?.input_mode}`);
await page.screenshot({ path: `${OUT}/xr_session_controllers.png` });

// Hand tracking mode.
await cmd("set_mode", { mode: "hands" });
await page.evaluate(() => { window.__xr.primaryInputMode = "hand"; });
await sleep(1200);
s = await status();
check(s.mode === "hands" && s.sources.l === 2 && s.sources.r === 2, `hands seen by server (sources=${JSON.stringify(s.sources)})`);
check(s.tracked.l && s.tracked.r && !s.mode_mismatch, "hands tracked, no mode mismatch");
await page.evaluate(() => window.__xr.hands.right.position.set(0.15, 1.0, -0.3));  // place hand before engaging
await sleep(500);
const rh0 = (await status()).ee_pos.r;
await page.evaluate(() => { const h = window.__xr.hands.right; h.poseId = "midpinch"; });
await sleep(500);
await page.evaluate(() => { window.__xr.hands.right.poseId = "default"; });
await sleep(500);
s = await status();
check(s.engaged.r, "thumb-middle pinch toggles right arm engaged");
await page.evaluate(() => window.__xr.hands.right.updatePinchValue(1));
await sleep(800);
s = await status();
check(s.grippers[1] < 0.3, `index pinch closes gripper (cmd=${s.grippers[1]})`);
for (let i = 1; i <= 20; i++) {
  await page.evaluate((k) => window.__xr.hands.right.position.set(0.15, 1.0, -0.3 - 0.005 * k), i);
  await sleep(40);
}
await sleep(1200);
s = await status();
const dh = s.ee_pos.r.map((v, i) => v - rh0[i]);
// hand moved 10 cm forward (XR -z) -> world +x
check(dh[0] > 0.07 && Math.abs(dh[1]) < 0.03 && Math.abs(dh[2]) < 0.03, `right EE follows tracked hand: Δworld=${dh.map((v) => v.toFixed(3))}`);
await page.screenshot({ path: `${OUT}/xr_session_hands.png` });

// Switch back to controllers while hands are tracked -> mismatch warning.
await cmd("set_mode", { mode: "controllers" });
await sleep(600);
s = await status();
check(s.mode_mismatch, "mode mismatch flagged when hands are used in controller mode");

check(errors.length === 0, `no page errors${errors.length ? ": " + errors.join(" | ") : ""}`);
await browser.close();
console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
