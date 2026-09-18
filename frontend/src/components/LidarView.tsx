import { useEffect, useRef } from 'react';
import { useCanvasView, applyView } from '../hooks/useCanvasView';
import type { PipelineResult } from '../types/api';
import { cellBounds, semanticColor, worldToCanvas } from '../utils/visualization';

type Mode = 'elevation' | 'semantic' | 'importance';

/**
 * Panel B — Original LiDAR / Scene. Renders the replayed input as returned
 * by the backend (map_cells of the current PipelineResult serve as the
 * lightweight local representation; full raw clouds are never fetched).
 * Caption always states the data source.
 */
export function LidarView({ result, mode, onMode, title = 'Panel B — Original LiDAR / Scene', hideModeSwitch = false }: {
  result: PipelineResult | null; mode: Mode; onMode: (m: Mode) => void; title?: string; hideModeSwitch?: boolean;
}) {
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
    let minE = Infinity, maxE = -Infinity;
    for (const c of limited) {
      if (c.elevation < minE) minE = c.elevation;
      if (c.elevation > maxE) maxE = c.elevation;
    }
    const span = Math.max(maxE - minE, 1e-6);
    for (const c of limited) {
      const w = worldToCanvas(c.x, c.y, bounds, S);
      const { cx, cy } = applyView(w.cx, w.cy, S, view);
      if (mode === 'semantic') ctx.fillStyle = semanticColor(c.semantic_class);
      else if (mode === 'importance') {
        const t = (c.importance - 0.2) / 0.6;
        ctx.fillStyle = t >= 0.83 ? '#b91c1c' : t >= 0.42 ? '#d97706' : t >= 0 ? '#0d9488' : '#1d4ed8';
      } else {
        const t = (c.elevation - minE) / span;
        const g = Math.round(40 + t * 200);
        ctx.fillStyle = `rgb(${Math.round(30 + t * 180)},${g},${Math.round(160 - t * 80)})`;
      }
      ctx.fillRect(cx, cy, 2, 2);
    }
  }, [limited, mode, view]);

  return (
    <section className="panel" aria-label="Original LiDAR scene">
      <h2>{title}</h2>
      {result && result.input_point_count != null && (
        <p className="chip">Points: {(result.input_point_count / 1000).toFixed(1)} K <span className="tag">measured replay input</span></p>
      )}
      {!hideModeSwitch && (
      <div className="seg" role="group" aria-label="color encoding">
        {(['elevation', 'semantic', 'importance'] as Mode[]).map((m) => (
          <button key={m} className={mode === m ? 'active' : ''} onClick={() => onMode(m)}>
            {m}
          </button>
        ))}
      </div>
      )}
      {result ? (
        <>
          <canvas ref={ref} width={360} height={360} className="viz cam" aria-label="replayed LiDAR top-down view"
            onWheel={cam.onWheel} onMouseDown={cam.onMouseDown} onMouseMove={cam.onMouseMove}
            onMouseUp={cam.onMouseUp} onMouseLeave={cam.onMouseLeave} onDoubleClick={cam.onDoubleClick} />
          <p className="caption">
            Replayed input · frame {result.frame_id.slice(0, 8)}… · {limited.length.toLocaleString()} of{' '}
            {result.map_cell_count.toLocaleString()} cells shown · encoding: {mode}
            {' · '}scroll = zoom · drag = pan · <button className="link" onClick={cam.reset}>reset view</button>
          </p>
        </>
      ) : (
        <p className="state">No result available for this frame. Select a frame and press Run.</p>
      )}
    </section>
  );
}
