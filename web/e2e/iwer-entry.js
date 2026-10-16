import { XRDevice, metaQuest3 } from "iwer";
import { oculusHandConfig } from "iwer/lib/device/XRHandInput.js";

// Extra hand pose for tests: middle fingertip touching the thumb tip (our clutch gesture).
const d = oculusHandConfig.poses.default;
const mid = { jointTransforms: {}, gripOffsetMatrix: d.gripOffsetMatrix };
for (const [k, v] of Object.entries(d.jointTransforms)) mid.jointTransforms[k] = { offsetMatrix: Float32Array.from(v.offsetMatrix), radius: v.radius };
mid.jointTransforms["middle-finger-tip"].offsetMatrix = Float32Array.from(d.jointTransforms["thumb-tip"].offsetMatrix);
oculusHandConfig.poses.midpinch = mid;

const dev = new XRDevice(metaQuest3);
dev.installRuntime({ forceInstall: true });
window.__xr = dev;
