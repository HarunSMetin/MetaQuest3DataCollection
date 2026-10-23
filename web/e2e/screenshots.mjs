// README screenshots (docs/media/) from a running server, using the emulated Quest 3 for the in-headset view.
//
//   uv run qc serve --no-https        # in another shell
//   npm run screenshots
import { chromium } from "playwright";
const OUT = `${import.meta.dirname}/../../docs/media`;
const BASE = process.env.QC_URL ?? "http://localhost:8000";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });

// Desktop dashboard.
const dash = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
await dash.goto(`${BASE}/dashboard/`);
await sleep(6000); // scene + camera previews
await dash.screenshot({ path: `${OUT}/dashboard.png` });

// Quest Browser landing page and the in-headset view.
const xr = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await xr.addInitScript({ path: `${import.meta.dirname}/iwer.bundle.js` });
await xr.goto(`${BASE}/xr/`);
await xr.waitForFunction(() => !document.getElementById("enter-vr").disabled, null, { timeout: 15000 });
await sleep(3000);
await xr.screenshot({ path: `${OUT}/xr_landing.png` });

await xr.evaluate(() => {
  const d = window.__xr;
  d.position.set(0, 1.6, 0);
  d.quaternion.set(-0.2164, 0, 0, 0.9763); // look ~25° down at the table
  d.controllers.left.position.set(-0.22, 1.25, -0.35);
  d.controllers.right.position.set(0.22, 1.15, -0.4);
});
await xr.click("#enter-vr");
await sleep(3000); // auto-recenter runs 1.2 s after entering
await xr.evaluate(() => window.__xr.controllers.right.updateButtonValue("squeeze", 1)); // engage (toggle): grasp-point marker brightens
await xr.evaluate(() => window.__xr.controllers.left.updateButtonValue("y-button", 1)); // wrist menu
await sleep(300);
await xr.evaluate(() => window.__xr.controllers.left.updateButtonValue("y-button", 0));
await sleep(4000);
await xr.screenshot({ path: `${OUT}/xr_headset.png` });

await browser.close();
console.log(`screenshots written to ${OUT}`);
