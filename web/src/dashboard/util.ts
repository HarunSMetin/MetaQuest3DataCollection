export type Cmd = (name: string, args?: Record<string, unknown>) => void;

export const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
