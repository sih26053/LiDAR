import { fmtFloat, fmtFps, fmtInt, fmtMeters, fmtMs } from '../utils/formatting';
import type { DemoMetrics, PipelineResult } from '../types/api';

/**
 * Panel E — Live Metrics. Every number comes from the backend response
 * (PipelineResult + DemoMetrics). Unavailable values render as such.
 */
export function MetricsPanel({ result, metrics }: { result: PipelineResult | null; metrics: DemoMetrics | null }) {
  const t = result?.timing;
  const cards: { label: string; value: string; hint?: string }[] = result
    ? [
        { label: 'Input points', value: fmtInt(result.input_point_count), hint: 'points' },
        { label: 'Processed points', value: fmtInt(result.processed_point_count), hint: 'points' },
        { label: 'Map cells', value: fmtInt(result.map_cell_count), hint: 'cells' },
        { label: 'Fine cells', value: fmtInt(result.resolution.fine_cells), hint: 'cells' },
        { label: 'Medium cells', value: fmtInt(result.resolution.medium_cells), hint: 'cells' },
        { label: 'Coarse cells', value: fmtInt(result.resolution.coarse_cells), hint: 'cells' },
        { label: 'Avg resolution', value: fmtMeters(result.resolution.average_resolution), hint: 'm' },
        { label: 'Mapping latency', value: fmtMs(t?.mapping_latency_ms), hint: 'core ML time' },
        { label: 'Total latency', value: fmtMs(t?.total_latency_ms), hint: 'compute only' },
        { label: 'API wall clock', value: fmtMs(t?.wall_clock_ms), hint: 'incl. overhead' },
        { label: 'Throughput', value: fmtFps(t?.fps), hint: '1000/total ms' },
        { label: 'Importance mean', value: fmtFloat(result.importance.mean) },
      ]
    : [];
  return (
    <section className="panel" aria-label="Live metrics">
      <h2>6. Key Metrics</h2>
      {result ? (
        <>
          <div className="cards">
            {cards.map((c) => (
              <div className="card" key={c.label}>
                <span className="card-label">{c.label}</span>
                <b className="card-value">{c.value}</b>
                {c.hint && <span className="card-hint">{c.hint}</span>}
              </div>
            ))}
          </div>
          <p className="caption">
            Mapping latency = core ML time (preprocessing + perception + feature + mapping). API wall clock
            includes network/serialization overhead and is shown separately.
            {metrics && metrics.latency_ms !== t?.total_latency_ms ? ' Metrics endpoint confirms stored result.' : ''}
          </p>
        </>
      ) : (
        <p className="state">No result available for this frame. Select a frame and press Run.</p>
      )}
    </section>
  );
}
