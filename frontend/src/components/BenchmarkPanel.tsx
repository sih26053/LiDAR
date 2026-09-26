import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { BenchmarkAsset, PipelineResult } from '../types/api';

/**
 * Benchmark comparison. Stored rows come ONLY from the stored asset
 * (frontend/public/benchmark.json, generated from results/benchmark_v2/:
 * verified re-run under the frozen runtime config). Nothing is
 * recalculated here; the current replay row is shown separately and is
 * NOT directly comparable (1 frame vs 10, current machine load).
 */
function fmt(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || !Number.isFinite(v) ? 'Not measured' : v.toFixed(digits);
}

export function BenchmarkPanel({ current }: { current: PipelineResult | null }) {
  const [asset, setAsset] = useState<BenchmarkAsset | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setLoading(true);
    api.getBenchmarkResults()
      .then((a) => { if (live) { setAsset(a); setError(null); } })
      .catch(() => { if (live) setError('Benchmark results are not available.'); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, []);

  return (
    <section className="panel wide" aria-label="Benchmark comparison">
      <h2>Benchmark Comparison — stored results vs current replay</h2>
      {loading && <p className="state">Loading benchmark…</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {asset && (
        <>
          <table className="bench">
            <thead>
              <tr><th>Method</th><th>Frames</th><th>Mean ± std mapping latency</th><th>Median mapping latency</th><th>Mean end-to-end</th><th>Mean cells</th><th>Mean resolution</th><th>Source</th></tr>
            </thead>
            <tbody>
              {asset.methods.map((m) => (
                <tr key={m.method} className={m.method === 'proposed' ? 'highlight' : ''}>
                  <td>{m.label}</td>
                  <td>{m.successful}/{m.frames}</td>
                  <td>{m.mean_mapping_latency_ms !== null ? `${fmt(m.mean_mapping_latency_ms)} ± ${fmt(m.std_mapping_latency_ms)} ms` : 'Not measured'}</td>
                  <td>{fmt(m.median_mapping_latency_ms)} ms</td>
                  <td>{fmt(m.mean_end_to_end_latency_ms)} ms</td>
                  <td>{m.mean_cells !== null ? Math.round(m.mean_cells).toLocaleString() : 'Not measured'}</td>
                  <td>{m.mean_resolution_m !== null ? `${m.mean_resolution_m.toFixed(3)} m` : 'Not measured'}</td>
                  <td className="tag">Stored benchmark result — not reproduced in current run</td>
                </tr>
              ))}
              <tr className="current">
                <td>Current replay{current ? ` (${current.frame_id.slice(0, 8)}…)` : ''}</td>
                <td>{current ? '1' : '—'}</td>
                <td>{current?.timing.mapping_latency_ms != null ? `${current.timing.mapping_latency_ms.toFixed(1)} ms` : 'Not measured'}</td>
                <td>n/a (single frame)</td>
                <td>{current?.timing.total_latency_ms != null ? `${current.timing.total_latency_ms.toFixed(1)} ms` : 'Not measured'}</td>
                <td>{current ? current.map_cell_count.toLocaleString() : 'Not measured'}</td>
                <td>{current?.resolution.average_resolution != null ? `${current.resolution.average_resolution.toFixed(3)} m` : 'Not measured'}</td>
                <td className="tag">Current replay result (this machine, frozen runtime config)</td>
              </tr>
            </tbody>
          </table>
          <div className="bars" aria-hidden="true">
            {asset.methods.map((m) => {
              const max = Math.max(...asset.methods.map((x) => x.mean_mapping_latency_ms ?? 0), 1);
              const w = ((m.mean_mapping_latency_ms ?? 0) / max) * 100;
              return (
                <div className="bar-row" key={m.method}>
                  <span>{m.label}</span>
                  <div className="bar"><div style={{ width: `${w}%` }} /></div>
                </div>
              );
            })}
          </div>
          <h3>Benchmark methodology (stored v2 run)</h3>
          <ul className="kv">
            <li><span>Dataset</span><b>nuScenes Mini, same 10 samples for every method (see results/benchmark_v2/benchmark_config_v2.json)</b></li>
            <li><span>Input</span><b>LiDAR_TOP via src/lidar_loader; shared preprocessing + perception + region prefix per frame</b></li>
            <li><span>Methods</span><b>Proposed importance-adaptive · Distance-adaptive · Uniform 5 cm (analytic grid) · Uniform 5 cm region-level (same 2 m regions, fair per-cell comparison)</b></li>
            <li><span>Timer</span><b>time.perf_counter(); 3 repetitions measured, 1 warm-up run discarded; mean ± std reported</b></li>
            <li><span>Latency scope</span><b>Mapping = map-build stage only; end-to-end = shared prefix + mapping. Rendering, file I/O, serialization, network excluded</b></li>
            <li><span>Config</span><b>Frozen runtime config: weights 0.30/0.30/0.15/0.15/0.10, thresholds I ≥ 0.70/0.45/0.20 → 0.05/0.10/0.20/else 0.50 m</b></li>
            <li><span>Hardware</span><b>Recorded at run time in results/benchmark_v2/environment_v2.json; live current-machine values in the Hardware &amp; Software panel</b></li>
          </ul>
          <p className="caption">
            Stored benchmark source: {asset.provenance.source}. Values are loaded, not recalculated
            by the frontend. Do not compare the 1-frame current-replay row against the 10-frame stored
            rows as though measured under identical conditions.
          </p>
          <p className="caption">
            Uniform 5 cm note: its mapping latency covers analytic uniform-grid sizing over the shared
            evaluation area (vectorized NumPy; per-cell arrays are not materialized), not a per-cell
            mapping loop — hence the small value beside ~7.45M theoretical cells. The
            uniform_5cm_region row maps the same regions at 0.05 m and is the fair per-cell comparison.
          </p>
        </>
      )}
    </section>
  );
}
