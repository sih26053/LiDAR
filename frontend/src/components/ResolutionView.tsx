import { useEffect, useRef } from 'react';
import type { PipelineResult } from '../types/api';
import { RESOLUTION_STYLE, cellBounds, resolutionTier, worldToCanvas } from '../utils/visualization';

/** Panel D (right) — Resolution view + current-frame decision summary. */
export function ResolutionView({ result }: { result: PipelineResult | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const cells = result?.map_cells ?? [];
  const limited = cells.length > 4000 ? cells.slice(0, 4000) : cells;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const S = canvas.width;
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(0, 0, S, S);
    if (limited.length === 0) return;
    const bounds = cellBounds(limited);
    for (const c of limited) {
      const { cx, cy } = worldToCanvas(c.x, c.y, bounds, S);
      const s = RESOLUTION_STYLE[resolutionTier(c.resolution)].px;
      ctx.fillStyle = RESOLUTION_STYLE[resolutionTier(c.resolution)].color;
      ctx.globalAlpha = 0.9;
      ctx.fillRect(cx - s / 2, cy - s / 2, s, s);
    }
    ctx.globalAlpha = 1;
  }, [limited]);

  return (
    <section className="panel" aria-label="Resolution view">
      <h2>Resolution + Decision</h2>
      {result ? (
        <>
          <canvas ref={ref} width={300} height={300} className="viz" aria-label="resolution map" />
          <ul className="kv">
            <li><span>Regions evaluated</span><b>{result.importance.count.toLocaleString()}</b></li>
            <li><span>Fine (0.05 m)</span><b>{result.resolution.fine_cells.toLocaleString()}</b></li>
            <li><span>Medium (0.10 m)</span><b>{result.resolution.medium_cells.toLocaleString()}</b></li>
            <li><span>Coarse (0.20/0.50 m)</span><b>{result.resolution.coarse_cells.toLocaleString()}</b></li>
            <li><span>Highest importance</span><b>{result.importance.max?.toFixed(3) ?? 'Unavailable'}</b></li>
          </ul>
          <p className="caption">High importance » finer resolution · Lower importance » coarser resolution</p>
        </>
      ) : (
        <p className="state">No result available for this frame.</p>
      )}
    </section>
  );
}
