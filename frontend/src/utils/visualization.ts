/**
 * Pure visualization scales (color/size buckets). These do NOT compute
 * importance or resolution -- they only map backend-supplied values to
 * colors and marker sizes for canvas/SVG rendering.
 */

export type ResolutionTier = 'fine' | 'medium' | 'coarse' | 'very_coarse';

/** Bucket a backend-supplied resolution (m) into a display tier. */
export function resolutionTier(resolutionM: number): ResolutionTier {
  if (resolutionM <= 0.05) return 'fine';
  if (resolutionM <= 0.1) return 'medium';
  if (resolutionM <= 0.2) return 'coarse';
  return 'very_coarse';
}

export const RESOLUTION_STYLE: Record<ResolutionTier, { label: string; color: string; px: number; blurb: string }> = {
  fine: { label: 'FINE 0.05 m', color: '#1d4ed8', px: 5, blurb: 'Highest map detail' },
  medium: { label: 'MEDIUM 0.10 m', color: '#0d9488', px: 4, blurb: 'Balanced detail' },
  coarse: { label: 'COARSE 0.20 m', color: '#d97706', px: 6, blurb: 'Lower computational detail' },
  very_coarse: { label: 'VERY COARSE 0.50 m', color: '#b91c1c', px: 8, blurb: 'Lowest detail, lowest cost' },
};

/** Map a backend-supplied importance in [0,1] to a display color. */
export function importanceColor(importance: number): string {
  if (!Number.isFinite(importance)) return '#9ca3af';
  if (importance >= 0.7) return '#b91c1c';
  if (importance >= 0.45) return '#d97706';
  if (importance >= 0.2) return '#0d9488';
  return '#1d4ed8';
}

export function importanceBand(importance: number): 'high' | 'medium' | 'low' {
  if (importance >= 0.7) return 'high';
  if (importance >= 0.45) return 'medium';
  return 'low';
}

const SEMANTIC_COLORS: Record<string, string> = {
  vehicle: '#7c3aed',
  pedestrian_vru: '#db2777',
  pedestrian: '#db2777',
  static_manmade: '#475569',
  drivable: '#16a34a',
  vegetation: '#15803d',
  unknown: '#9ca3af',
};

export function semanticColor(semanticClass: string): string {
  return SEMANTIC_COLORS[semanticClass] ?? '#6b7280';
}

/** Project world X/Y onto a square canvas with padding. */
export function worldToCanvas(
  x: number,
  y: number,
  bounds: { minX: number; maxX: number; minY: number; maxY: number },
  size: number,
  pad = 12,
): { cx: number; cy: number } {
  const spanX = Math.max(bounds.maxX - bounds.minX, 1e-6);
  const spanY = Math.max(bounds.maxY - bounds.minY, 1e-6);
  const cx = pad + ((x - bounds.minX) / spanX) * (size - 2 * pad);
  const cy = pad + ((1 - (y - bounds.minY) / spanY)) * (size - 2 * pad);
  return { cx, cy };
}

export function cellBounds(cells: { x: number; y: number }[]): {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
} {
  if (cells.length === 0) return { minX: -80, maxX: 80, minY: -80, maxY: 80 };
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const c of cells) {
    if (c.x < minX) minX = c.x;
    if (c.x > maxX) maxX = c.x;
    if (c.y < minY) minY = c.y;
    if (c.y > maxY) maxY = c.y;
  }
  if (minX === maxX) { minX -= 1; maxX += 1; }
  if (minY === maxY) { minY -= 1; maxY += 1; }
  return { minX, maxX, minY, maxY };
}
