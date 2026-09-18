import { useEffect, useRef, useState } from 'react';
import { useCanvasView, applyView } from '../hooks/useCanvasView';
import type { PipelineResult } from '../types/api';
import { RESOLUTION_STYLE, cellBounds, resolutionTier, worldToCanvas } from '../utils/visualization';

/** Measured cell extent (m), derived from backend cell positions. */
function extentLabel(result: PipelineResult): string {
  const b = cellBounds(result.map_cells);
  return `${(b.maxX - b.minX).toFixed(0)} m × ${(b.maxY - b.minY).toFixed(0)} m measured`;
}

/**
 * Panel C — Adaptive 2.5D Map. Cell size on screen follows the backend
 * `resolution` value; semantic tint is shown only where the backend
 * reports a non-fallback source.
 */
export function AdaptiveMapView({ result }: { result: PipelineResult | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [showSemantic, setShowSemantic] = useState(true);
  const cells = result?.map_cells ?? [];
  const limited = cells.length > 4000 ? cells.slice(0, 4000) : cells;
  const cam = useCanvasView(result ? result.frame_id : null);
  const view = cam.view;

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
    let minE = Infinity, maxE = -Infinity;
    for (const c of limited) {
      if (c.elevation < minE) minE = c.elevation;
      if (c.elevation > maxE) maxE = c.elevation;
    }
    const span = Math.max(maxE - minE, 1e-6);
    for (const c of limited) {
      const w = worldToCanvas(c.x, c.y, bounds, S);
      const { cx, cy } = applyView(w.cx, w.cy, S, view);
      const tier = resolutionTier(c.resolution);
      const style = RESOLUTION_STYLE[tier];
      // Base: resolution color; semantic overlay: white ring when valid.
      const t = (c.elevation - minE) / span;
      ctx.fillStyle = style.color;
      ctx.globalAlpha = 0.35 + t * 0.55;
      const s = style.px;
      ctx.fillRect(cx - s / 2, cy - s / 2, s, s);
      ctx.globalAlpha = 1;
      if (showSemantic && c.semantic_source !== 'fallback' && c.semantic_source !== 'unknown') {
        ctx.strokeStyle = '#f8fafc';
        ctx.lineWidth = 1;
        ctx.strokeRect(cx - s / 2 - 1, cy - s / 2 - 1, s + 2, s + 2);
      }
    }
  }, [limited, showSemantic, view]);

  const semanticCells = result ? result.map_cells.filter((c) => c.semantic_source !== 'fallback' && c.semantic_source !== 'unknown').length : 0;

  return (
    <section className="panel map-main" aria-label="Adaptive 2.5D map">
      <h2>4. Adaptive 2.5D Map (Final Output)</h2>
      <label className="check">
        <input type="checkbox" checked={showSemantic} onChange={(e) => setShowSemantic(e.target.checked)} />
        Semantic overlay (white ring = valid source)
      </label>
      {result ? (
        <>
          <canvas ref={ref} width={360} height={360} className="viz cam" aria-label="adaptive 2.5D map, marker size encodes resolution"
            onWheel={cam.onWheel} onMouseDown={cam.onMouseDown} onMouseMove={cam.onMouseMove}
            onMouseUp={cam.onMouseUp} onMouseLeave={cam.onMouseLeave} onDoubleClick={cam.onDoubleClick} />
          <p className="chip">
            Map Cells: {result.map_cell_count.toLocaleString()} <span className="tag">measured</span>
            {' · '}Extent: {extentLabel(result)}
          </p>
          <ul className="kv mapinfo">
            <li><span>Elevation (height)</span><b>per-cell, m</b></li>
            <li><span>Occupancy</span><b>per-cell</b></li>
            <li><span>Semantic class</span><b>{[...new Set(result.map_cells.map((c) => c.semantic_class))].slice(0, 6).join(', ') || 'none'}</b></li>
            <li><span>Confidence</span><b>n/a — annotation reference, no model</b></li>
            <li><span>Importance</span><b>mean {result.importance.mean?.toFixed(3) ?? 'unavailable'}</b></li>
            <li><span>Resolution</span><b>4 tiers (0.05–0.50 m)</b></li>
          </ul>
          <p className="caption">
            X/Y position · marker size = backend <code>resolution</code> · brightness = elevation ·{' '}
            {semanticCells.toLocaleString()} of {result.map_cell_count.toLocaleString()} cells carry valid semantic source
            {' · '}scroll = zoom · drag = pan · <button className="link" onClick={cam.reset}>reset view</button>
          </p>
        </>
      ) : (
        <p className="state">No result available for this frame. Select a frame and press Run.</p>
      )}
    </section>
  );
}
