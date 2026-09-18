/** Display derivations from backend PipelineResult cells.
 *
 * Pure geometry/grouping for visualization ONLY — no ML recomputation, no new
 * claims. Distances are Euclidean from the sensor origin (x, y). Semantic
 * classes/sources pass through verbatim from the backend.
 */
import type { MapCell, PipelineResult } from '../types/api';

export function cellDistance(x: number, y: number): number {
  return Math.hypot(x, y);
}

export interface ObjectGroup {
  semantic_class: string;
  kind: 'dynamic' | 'static' | 'surface' | 'unclassified';
  cells: number;
  centroidX: number;
  centroidY: number;
  distanceM: number;
  maxImportance: number;
  source: string;
}

const KIND: Record<string, ObjectGroup['kind']> = {
  vehicle: 'dynamic',
  pedestrian_vru: 'dynamic',
  static_manmade: 'static',
  vegetation: 'static',
  road_driveable: 'surface',
  unknown: 'unclassified',
};

export const KIND_LABEL: Record<ObjectGroup['kind'], string> = {
  dynamic: 'Dynamic object',
  static: 'Static obstacle/terrain',
  surface: 'Drivable surface',
  unclassified: 'Unclassified',
};

/** Group annotation-sourced cells by class (fallback/unknown cells excluded). */
export function groupObjects(cells: MapCell[]): ObjectGroup[] {
  const acc = new Map<string, { n: number; sx: number; sy: number; maxI: number; src: Map<string, number> }>();
  for (const c of cells) {
    if (c.semantic_source === 'fallback' || c.semantic_source === 'unknown') continue;
    if (c.semantic_class === 'unknown') continue;
    let g = acc.get(c.semantic_class);
    if (!g) { g = { n: 0, sx: 0, sy: 0, maxI: -Infinity, src: new Map() }; acc.set(c.semantic_class, g); }
    g.n += 1; g.sx += c.x; g.sy += c.y;
    if (c.importance > g.maxI) g.maxI = c.importance;
    g.src.set(c.semantic_source, (g.src.get(c.semantic_source) ?? 0) + 1);
  }
  const out: ObjectGroup[] = [];
  for (const [cls, g] of acc) {
    const cx = g.sx / g.n, cy = g.sy / g.n;
    const dominantSrc = [...g.src.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'unknown';
    out.push({
      semantic_class: cls, kind: KIND[cls] ?? 'unclassified', cells: g.n,
      centroidX: cx, centroidY: cy, distanceM: cellDistance(cx, cy),
      maxImportance: g.maxI, source: dominantSrc,
    });
  }
  return out.sort((a, b) => b.maxImportance - a.maxImportance);
}

export interface TerrainSplit {
  drivable: number;
  nonDrivable: number;
  dynamic: number;
  unclassified: number;
  total: number;
}

/** Terrain split from backend cells (annotation classes; fallback = unclassified). */
export function terrainSplit(cells: MapCell[]): TerrainSplit {
  let drivable = 0, nonDrivable = 0, dynamic = 0, unclassified = 0;
  for (const c of cells) {
    if (c.semantic_source === 'fallback' || c.semantic_source === 'unknown' || c.semantic_class === 'unknown') { unclassified += 1; continue; }
    if (c.semantic_class === 'road_driveable') drivable += 1;
    else if (c.semantic_class === 'static_manmade' || c.semantic_class === 'vegetation') nonDrivable += 1;
    else if (c.semantic_class === 'vehicle' || c.semantic_class === 'pedestrian_vru') dynamic += 1;
    else unclassified += 1;
  }
  return { drivable, nonDrivable, dynamic, unclassified, total: cells.length };
}

export interface DistanceBand { label: string; cells: number; meanResolution: number | null; }

/** Mean backend resolution per distance band (analysis view; allocation stays importance-driven). */
export function resolutionByDistance(cells: MapCell[]): DistanceBand[] {
  const bands: { label: string; lo: number; hi: number; xs: number[] }[] = [
    { label: '0–20 m', lo: 0, hi: 20, xs: [] },
    { label: '20–40 m', lo: 20, hi: 40, xs: [] },
    { label: '40–60 m', lo: 40, hi: 60, xs: [] },
    { label: '60 m+', lo: 60, hi: Infinity, xs: [] },
  ];
  for (const c of cells) {
    const d = cellDistance(c.x, c.y);
    const b = bands.find((x) => d >= x.lo && d < x.hi) ?? bands[bands.length - 1];
    b.xs.push(c.resolution);
  }
  return bands.map((b) => ({
    label: b.label, cells: b.xs.length,
    meanResolution: b.xs.length ? b.xs.reduce((a, v) => a + v, 0) / b.xs.length : null,
  }));
}

export interface Alert { severity: 'high' | 'info' | 'ok'; text: string; detail: string; }

/** Data-driven alerts from the current backend result (no invented events). */
export function deriveAlerts(result: PipelineResult): Alert[] {
  const alerts: Alert[] = [];
  const groups = groupObjects(result.map_cells);
  for (const g of groups) {
    if (g.maxImportance >= 0.7) {
      alerts.push({
        severity: 'high',
        text: `High importance ${g.semantic_class} region (${KIND_LABEL[g.kind].toLowerCase()})`,
        detail: `Centroid (${g.centroidX.toFixed(1)}, ${g.centroidY.toFixed(1)}) · ${g.distanceM.toFixed(0)} m · max importance ${g.maxImportance.toFixed(2)} · source: ${g.source} (annotation reference)`,
      });
    }
    if (g.semantic_class === 'pedestrian_vru' && g.distanceM <= 50) {
      alerts.push({
        severity: 'high',
        text: 'Pedestrian-proximity region within 50 m',
        detail: `Centroid (${g.centroidX.toFixed(1)}, ${g.centroidY.toFixed(1)}) · ${g.distanceM.toFixed(0)} m · ${g.cells} annotation cells`,
      });
    }
  }
  const split = terrainSplit(result.map_cells);
  if (split.total > 0 && split.unclassified / split.total > 0.8) {
    alerts.push({
      severity: 'info',
      text: 'Limited annotation coverage for this frame',
      detail: `${split.unclassified.toLocaleString()} of ${split.total.toLocaleString()} cells fall back to heuristic labels; objects table lists annotation cells only.`,
    });
  }
  if (alerts.length === 0) {
    alerts.push({ severity: 'ok', text: 'System running normally', detail: 'Frame processed; no high-importance annotation regions.' });
  }
  return alerts.slice(0, 6);
}
