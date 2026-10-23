// Binary wire protocol — mirror of questcollect/protocol.py. Change both together.

export const MSG_INPUT = 1;
export const MSG_POSE = 2;

export const SRC_NONE = 0;
export const SRC_CONTROLLER = 1;
export const SRC_HAND = 2;

export const N_HAND_JOINTS = 25;
export const HAND_FLOATS = 7 + 8 + N_HAND_JOINTS * 7 + 2;
export const HEADER_BYTES = 16; // <BBBBId
export const INPUT_FLOATS = 7 + 2 * HAND_FLOATS;
export const INPUT_BYTES = HEADER_BYTES + 4 * INPUT_FLOATS;
export const POSE_HEADER_BYTES = 16; // <BBHId

export const FLAG_ENGAGED_L = 1;
export const FLAG_ENGAGED_R = 2;
export const FLAG_RECORDING = 4;

// Button slots.
export const B_TRIGGER = 0, B_SQUEEZE = 1, B_STICK_X = 2, B_STICK_Y = 3,
  B_PRIMARY = 4, B_SECONDARY = 5, B_STICK_PRESS = 6, B_MENU = 7;

// WebXR hand joint order (index = slot in the message).
export const HAND_JOINTS: XRHandJoint[] = [
  "wrist",
  "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
  "index-finger-metacarpal", "index-finger-phalanx-proximal", "index-finger-phalanx-intermediate",
  "index-finger-phalanx-distal", "index-finger-tip",
  "middle-finger-metacarpal", "middle-finger-phalanx-proximal", "middle-finger-phalanx-intermediate",
  "middle-finger-phalanx-distal", "middle-finger-tip",
  "ring-finger-metacarpal", "ring-finger-phalanx-proximal", "ring-finger-phalanx-intermediate",
  "ring-finger-phalanx-distal", "ring-finger-tip",
  "pinky-finger-metacarpal", "pinky-finger-phalanx-proximal", "pinky-finger-phalanx-intermediate",
  "pinky-finger-phalanx-distal", "pinky-finger-tip",
];
export const J_WRIST = 0, J_THUMB_TIP = 4, J_INDEX_TIP = 9, J_MIDDLE_META = 10, J_MIDDLE_TIP = 14;

export interface HandSample {
  source: number;
  pose: Float32Array; // 7: pos xyz, quat xyzw
  buttons: Float32Array; // 8
  joints: Float32Array; // 25*7
  pinch: Float32Array; // 2
}

export function emptyHand(): HandSample {
  const pose = new Float32Array(7);
  pose[6] = 1;
  return { source: SRC_NONE, pose, buttons: new Float32Array(8), joints: new Float32Array(N_HAND_JOINTS * 7), pinch: new Float32Array([0.1, 0.1]) };
}

export class InputPacker {
  private buf = new ArrayBuffer(INPUT_BYTES);
  private view = new DataView(this.buf);
  private floats = new Float32Array(this.buf, HEADER_BYTES);
  private seq = 0;

  /** tEcho: t_server of the last pose frame received (the server measures RTT from it). */
  pack(head: Float32Array, left: HandSample, right: HandSample, tEcho: number): ArrayBuffer {
    const v = this.view;
    v.setUint8(0, MSG_INPUT);
    v.setUint8(1, left.source);
    v.setUint8(2, right.source);
    v.setUint8(3, 0);
    v.setUint32(4, this.seq++ >>> 0, true);
    v.setFloat64(8, tEcho, true);
    const f = this.floats;
    f.set(head, 0);
    let o = 7;
    for (const h of [left, right]) {
      f.set(h.pose, o); o += 7;
      f.set(h.buttons, o); o += 8;
      f.set(h.joints, o); o += N_HAND_JOINTS * 7;
      f.set(h.pinch, o); o += 2;
    }
    return this.buf;
  }
}

export interface PoseFrame {
  flags: number;
  nbody: number;
  seq: number;
  tServer: number;
  bodies: Float32Array; // nbody*7
}

export function parsePoseFrame(buf: ArrayBuffer): PoseFrame | null {
  const v = new DataView(buf);
  if (v.getUint8(0) !== MSG_POSE) return null;
  const flags = v.getUint8(1);
  const nbody = v.getUint16(2, true);
  const seq = v.getUint32(4, true);
  const tServer = v.getFloat64(8, true);
  const f = new Float32Array(buf, POSE_HEADER_BYTES);
  return { flags, nbody, seq, tServer, bodies: f.subarray(0, nbody * 7) };
}

// ---- JSON messages ----------------------------------------------------------------------

export interface Status {
  type: "status";
  task: string;
  instruction: string;
  mode: "controllers" | "hands";
  scale: number;
  rec_state: "idle" | "countdown" | "recording" | "review";
  countdown: number;
  elapsed: number;
  steps: number;
  saved_count: number;
  auto_stop: boolean;
  success: boolean;
  success_seen: boolean;
  progress: number; // bricks correctly stacked on the base
  levels: number;
  in_area: boolean; // base brick lies inside the build area
  held: { l: string | null; r: string | null };
  operator: string;
  engaged: { l: boolean; r: boolean };
  tracked: { l: boolean; r: boolean };
  mode_mismatch: boolean;
  sources: { l: number; r: number };
  grippers: [number, number];
  ik_error_mm: { l: number; r: number };
  out_of_reach: { l: boolean; r: boolean }; // IK can't get the gripper to its target
  ee_pos: { l: [number, number, number]; r: [number, number, number] };
  warnings: { arm_collision: boolean; table_collision: boolean; ik_saturated: boolean };
  input_hz: number;
  rtt_ms: number | null;
  loop_hz: number;
  connected: boolean;
  placement: number[]; // column-major 4x4, world -> xr
  replay: { path: string; i: number; n: number } | null;
}

export type ServerEvent =
  | { type: "toast"; text: string; kind: "info" | "success" | "warn" | "rec" }
  | { type: "haptic"; arm: "l" | "r"; intensity: number; ms: number; repeat?: number; gap_ms?: number }
  | { type: "saved"; path: string }
  | { type: "sound"; name: string; arm: "l" | "r" | null };

/** The mission as data (GET /api/task, questcollect.sim.task_spec). The UI draws the goal from this alone. */
export interface TaskSpec {
  task: string;
  instruction: string;
  pitch: number;
  stud_r: number;
  stud_h: number;
  bricks: { name: string; label: string; size: [number, number, number]; studs: [number, number]; rgba: number[] }[];
  /** Build order; pos/yaw relative to the base brick (bottom centre, its yaw), metres / radians. */
  goal: { brick: number; pos: [number, number, number]; yaw: number }[];
  area: { center: [number, number]; half: [number, number] };
}

export async function fetchTask(): Promise<TaskSpec> {
  return (await fetch("/api/task")).json();
}

export interface EpisodeRow {
  path: string;
  file: string;
  success: boolean;
  duration_s: number;
  num_steps: number;
  input_mode: string;
  operator: string;
  created_at: string;
}
