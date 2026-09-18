import type { PipelineResult } from '../types/api';

/** Explains the pipeline + what the system decided for the current frame. */
export function ExplanationPanel({ result }: { result: PipelineResult | null }) {
  const critical = result
    ? result.map_cells
        .filter((c) => ['vehicle', 'pedestrian', 'pedestrian_vru'].includes(c.semantic_class) && c.semantic_source !== 'fallback' && c.semantic_source !== 'unknown')
        .slice(0, 6)
    : [];
  return (
    <section className="panel wide" aria-label="Explanation">
      <h2>How it works — what the system decided</h2>
      <div className="flow" aria-label="pipeline flow">
        {['LiDAR', 'Perception + geometry', 'Importance estimation', 'Resolution allocation', 'Adaptive 2.5D map'].map((s, i, a) => (
          <span key={s}>{s}{i < a.length - 1 ? '  »  ' : ''}</span>
        ))}
      </div>
      <p>
        Regions that are more important, dynamic, uncertain, or geometrically complex can receive finer
        spatial resolution, while lower-priority regions remain coarser. Importance shown here is the
        backend/model output — the frontend only visualizes it.
      </p>
      {result ? (
        <ul className="kv">
          <li><span>Regions evaluated</span><b>{result.importance.count.toLocaleString()}</b></li>
          <li><span>Fine regions</span><b>{result.resolution.fine_cells.toLocaleString()}</b></li>
          <li><span>Medium regions</span><b>{result.resolution.medium_cells.toLocaleString()}</b></li>
          <li><span>Coarse regions</span><b>{result.resolution.coarse_cells.toLocaleString()}</b></li>
          <li><span>Highest importance observed</span><b>{result.importance.max?.toFixed(3) ?? 'Unavailable'}</b></li>
          <li><span>Critical semantic regions</span><b>{critical.length > 0 ? critical.map((c) => `${c.semantic_class}@(${c.x.toFixed(1)},${c.y.toFixed(1)})`).join('; ') : 'none valid in this frame'}</b></li>
        </ul>
      ) : (
        <p className="state">No frame selected — run a frame to see what the system decided.</p>
      )}
      {result && (
        <div className="takeaway" aria-label="judge takeaway">
          <h3>Takeaway</h3>
          <ul className="kv">
            <li><span>What it sees</span><b>Real LiDAR environment ({result.input_point_count != null ? result.input_point_count.toLocaleString() : 'unavailable'} points)</b></li>
            <li><span>What it decides</span><b>Importance of each spatial region (mean {result.importance.mean?.toFixed(3) ?? 'unavailable'})</b></li>
            <li><span>What it adapts</span><b>Map resolution per region ({result.resolution.fine_cells} fine / {result.resolution.medium_cells} medium / {result.resolution.coarse_cells} coarse cells)</b></li>
            <li><span>What it produces</span><b>Variable-resolution 2.5D map ({result.map_cell_count.toLocaleString()} cells)</b></li>
            <li><span>Measured performance</span><b>{result.timing.total_latency_ms != null ? `${result.timing.total_latency_ms.toFixed(1)} ms` : 'unavailable'} · {result.timing.fps != null ? `${result.timing.fps.toFixed(2)} FPS` : 'unavailable'} (backend-measured)</b></li>
          </ul>
          <p className="caption">
            Detail where needed, coarser elsewhere. Semantic labels are annotation references
            ({result.semantic.mode}), never model predictions. Benchmark comparison is shown separately
            from stored results.
          </p>
        </div>
      )}
    </section>
  );
}
