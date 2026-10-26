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
  d.controllers.left.position.set(-0.2, 1.3, -0.4);
  d.controllers.right.position.set(0.2, 1.3, -0.4);
});
await xr.click("#enter-vr");
await sleep(6000); // auto-recenter runs 1.2 s after entering; controller models load
// Software GL sends input at only a few Hz, so hold each button long enough for the server to see it.
const press = async (side, button) => {
  await xr.evaluate(([s, b]) => window.__xr.controllers[s].updateButtonValue(b, 1), [side, button]);
  await sleep(1500);
  await xr.evaluate(([s, b]) => window.__xr.controllers[s].updateButtonValue(b, 0), [side, button]);
};
await press("right", "squeeze"); // take control of the right arm: glow, status chip, brighter grasp-point marker
await sleep(4000);
await xr.screenshot({ path: `${OUT}/xr_headset.png` });

await press("left", "y-button"); // wrist menu
await sleep(1500);
await xr.screenshot({ path: `${OUT}/xr_menu.png` });

await browser.close();
console.log(`screenshots written to ${OUT}`);
