// Samples head, controllers and tracked hands from the WebXR frame into wire-format structs.

import {
  B_MENU, B_PRIMARY, B_SECONDARY, B_SQUEEZE, B_STICK_PRESS, B_STICK_X, B_STICK_Y, B_TRIGGER,
  emptyHand, HAND_JOINTS, type HandSample, J_INDEX_TIP, J_MIDDLE_TIP, J_THUMB_TIP, SRC_CONTROLLER, SRC_HAND,
  SRC_NONE,
} from "../protocol";

export type Side = "left" | "right";

export interface RayPose { origin: [number, number, number]; quat: [number, number, number, number] }

function dist(j: Float32Array, a: number, b: number) {
  const dx = j[a * 7] - j[b * 7], dy = j[a * 7 + 1] - j[b * 7 + 1], dz = j[a * 7 + 2] - j[b * 7 + 2];
  return Math.hypot(dx, dy, dz);
}

function writePose(dst: Float32Array, o: number, t: XRRigidTransform) {
  dst[o] = t.position.x; dst[o + 1] = t.position.y; dst[o + 2] = t.position.z;
  dst[o + 3] = t.orientation.x; dst[o + 4] = t.orientation.y; dst[o + 5] = t.orientation.z; dst[o + 6] = t.orientation.w;
}

export class XRInputSampler {
  readonly head = new Float32Array([0, 1.6, 0, 0, 0, 0, 1]);
  readonly hands: Record<Side, HandSample> = { left: emptyHand(), right: emptyHand() };
  readonly rays: Record<Side, RayPose | null> = { left: null, right: null };
  readonly sources: Record<Side, XRInputSource | null> = { left: null, right: null };
  readonly radii: Record<Side, Float32Array> = { left: new Float32Array(25), right: new Float32Array(25) };

  sample(frame: XRFrame, ref: XRReferenceSpace) {
    const viewer = frame.getViewerPose(ref);
    if (viewer) writePose(this.head, 0, viewer.transform);

    const seen: Record<Side, boolean> = { left: false, right: false };
    for (const src of frame.session.inputSources) {
      const side = src.handedness as Side;
      if (side !== "left" && side !== "right") continue;
      const h = this.hands[side];
      this.sources[side] = src;
      const ray = frame.getPose(src.targetRaySpace, ref);
      this.rays[side] = ray
        ? { origin: [ray.transform.position.x, ray.transform.position.y, ray.transform.position.z],
            quat: [ray.transform.orientation.x, ray.transform.orientation.y, ray.transform.orientation.z, ray.transform.orientation.w] }
        : null;

      if (src.hand) {
        let ok = true;
        for (let i = 0; i < HAND_JOINTS.length; i++) {
          const space = src.hand.get(HAND_JOINTS[i]);
          const jp = space && frame.getJointPose ? frame.getJointPose(space, ref) : undefined;
          if (!jp) { ok = false; break; }
          writePose(h.joints, i * 7, jp.transform);
          this.radii[side][i] = jp.radius ?? 0.008;
        }
        if (!ok) continue;
        h.source = SRC_HAND;
        h.pose.set(h.joints.subarray(0, 7));
        h.buttons.fill(0);
        h.pinch[0] = dist(h.joints, J_THUMB_TIP, J_INDEX_TIP);
        h.pinch[1] = dist(h.joints, J_THUMB_TIP, J_MIDDLE_TIP);
        seen[side] = true;
      } else if (src.gamepad) {
        const space = src.gripSpace ?? src.targetRaySpace;
        const p = frame.getPose(space, ref);
        if (!p) continue;
        h.source = SRC_CONTROLLER;
        writePose(h.pose, 0, p.transform);
        const b = src.gamepad.buttons, ax = src.gamepad.axes;
        const val = (i: number) => (b[i] ? b[i].value || (b[i].pressed ? 1 : 0) : 0);
        h.buttons[B_TRIGGER] = val(0);
        h.buttons[B_SQUEEZE] = val(1);
        h.buttons[B_STICK_X] = ax[2] ?? 0;
        h.buttons[B_STICK_Y] = ax[3] ?? 0;
        h.buttons[B_PRIMARY] = val(4);
        h.buttons[B_SECONDARY] = val(5);
        h.buttons[B_STICK_PRESS] = val(3);
        h.buttons[B_MENU] = val(6);
        h.joints.fill(0);
        h.pinch[0] = h.pinch[1] = 0.1;
        seen[side] = true;
      }
    }
    for (const side of ["left", "right"] as Side[]) {
      if (!seen[side]) {
        this.hands[side].source = SRC_NONE;
        this.rays[side] = null;
        this.sources[side] = null;
      }
    }
  }

  pulse(side: Side, intensity: number, ms: number) {
    const gp = this.sources[side]?.gamepad as (Gamepad & { hapticActuators?: { pulse?: (v: number, d: number) => void }[] }) | undefined;
    if (!gp) return;
    try {
      const pulse = gp.hapticActuators?.[0]?.pulse;
      if (pulse) pulse.call(gp.hapticActuators![0], intensity, ms);
      else gp.vibrationActuator?.playEffect("dual-rumble", { duration: ms, strongMagnitude: intensity, weakMagnitude: intensity });
    } catch { /* haptics are best effort */ }
  }
}
