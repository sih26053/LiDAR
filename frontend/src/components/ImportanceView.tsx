import { useEffect, useRef } from 'react';
import { useCanvasView, applyView } from '../hooks/useCanvasView';
import type { PipelineResult } from '../types/api';
import { cellBounds, importanceColor, worldToCanvas } from '../utils/visualization';

/** Panel D (left) — Importance view. Colors visualize backend `importance` only. */
export function ImportanceView({ result }: { result: PipelineResult | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
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
    for (const c of limited) {
      const w = worldToCanvas(c.x, c.y, bounds, S);
      const { cx, cy } = applyView(w.cx, w.cy, S, view);
      ctx.fillStyle = importanceColor(c.importance);
      ctx.globalAlpha = 0.85;
      ctx.fillRect(cx - 2, cy - 2, 4, 4);
    }
    ctx.globalAlpha = 1;
  }, [limited, view]);

  return (
    <section className="panel" aria-label="Importance view">
      <h2>3. Importance Map</h2>
      {result ? (
        <>
          <canvas ref={ref} width={300} height={300} className="viz cam" aria-label="importance heatmap"
            onWheel={cam.onWheel} onMouseDown={cam.onMouseDown} onMouseMove={cam.onMouseMove}
            onMouseUp={cam.onMouseUp} onMouseLeave={cam.onMouseLeave} onDoubleClick={cam.onDoubleClick} />
          <p className="caption">
            importance » resolution decision · range {result.importance.min?.toFixed(2) ?? '–'}–
            {result.importance.max?.toFixed(2) ?? '–'} · mean {result.importance.mean?.toFixed(3) ?? '–'}
            {' · '}scroll = zoom · drag = pan · <button className="link" onClick={cam.reset}>reset view</button>
          </p>
        </>
      ) : (
        <p className="state">No result available for this frame.</p>
      )}
    </section>
  );
}
