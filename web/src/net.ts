// Auto-reconnecting websocket to the QuestCollect server.

import { parsePoseFrame, type PoseFrame, type ServerEvent, type Status } from "./protocol";

export interface Handlers {
  onPose?: (f: PoseFrame) => void;
  onStatus?: (s: Status) => void;
  onEvent?: (e: ServerEvent) => void;
  onOpen?: () => void;
  onClose?: () => void;
}

export class Link {
  ws: WebSocket | null = null;
  open = false;
  private closed = false;

  constructor(private path: "/ws/xr" | "/ws/dash", private h: Handlers) {
    this.connect();
  }

  private connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}${this.path}`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.open = true;
      this.h.onOpen?.();
    };
    ws.onclose = () => {
      this.open = false;
      this.h.onClose?.();
      if (!this.closed) setTimeout(() => this.connect(), 1000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        if (msg.type === "status") this.h.onStatus?.(msg as Status);
        else this.h.onEvent?.(msg as ServerEvent);
      } else {
        const f = parsePoseFrame(ev.data as ArrayBuffer);
        if (f) this.h.onPose?.(f);
      }
    };
    this.ws = ws;
  }

  /** Send binary input unless the socket is backed up (drop rather than queue stale input). */
  sendBinary(buf: ArrayBuffer) {
    if (this.ws && this.open && this.ws.bufferedAmount < 32_768) this.ws.send(buf);
  }

  command(name: string, args: Record<string, unknown> = {}) {
    if (this.ws && this.open) this.ws.send(JSON.stringify({ type: "command", name, args }));
  }

  close() {
    this.closed = true;
    this.ws?.close();
  }
}
