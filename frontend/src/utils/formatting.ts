/** Pure display formatting. No ML/business logic here. */

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'Unavailable';
  return Math.round(v).toLocaleString('en-US');
}

export function fmtFloat(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'Unavailable';
  return v.toFixed(digits);
}

export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'Unavailable';
  return `${v.toFixed(1)} ms`;
}

export function fmtFps(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'Unavailable';
  return `${v.toFixed(2)} FPS`;
}

export function fmtMeters(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return 'Unavailable';
  return `${v.toFixed(2)} m`;
}

export function fmtTimestamp(ts: number | null | undefined): string {
  if (ts === null || ts === undefined || !Number.isFinite(ts)) return 'Unavailable';
  // nuScenes timestamps are microseconds since epoch.
  const ms = ts > 1e12 ? ts / 1000 : ts;
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return String(ts);
  return d.toISOString();
}

export function shortId(id: string | null | undefined): string {
  if (!id) return 'Unavailable';
  return id.length > 12 ? `${id.slice(0, 8)}…` : id;
}
