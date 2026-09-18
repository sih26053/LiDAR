import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { BenchmarkAsset, PipelineResult } from '../types/api';

/**
 * Benchmark comparison. Data comes ONLY from the stored asset
 * (frontend/public/benchmark.json, generated from results/benchmark/).
 * Nothing is recalculated here; the current replay row is shown separately.
 */
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
              <tr><th>Method</th><th>Frames</th><th>Mean mapping latency</th><th>Mean cells</th><th>Mean resolution</th><th>Source</th></tr>
            </thead>
            <tbody>
              {asset.methods.map((m) => (
                <tr key={m.method} className={m.method === 'proposed' ? 'highlight' : ''}>
                  <td>{m.label}</td>
                  <td>{m.successful}/{m.frames}</td>
                  <td>{m.mean_mapping_latency_ms !== null ? `${m.mean_mapping_latency_ms.toFixed(1)} ms` : 'Unavailable'}</td>
                  <td>{m.mean_cells !== null ? Math.round(m.mean_cells).toLocaleString() : 'Unavailable'}</td>
                  <td>{m.mean_resolution_m !== null ? `${m.mean_resolution_m.toFixed(3)} m` : 'Unavailable'}</td>
                  <td className="tag">Stored benchmark result</td>
                </tr>
              ))}
              <tr className="current">
                <td>Current replay{current ? ` (${current.frame_id.slice(0, 8)}…)` : ''}</td>
                <td>{current ? '1' : '—'}</td>
                <td>{current?.timing.mapping_latency_ms != null ? `${current.timing.mapping_latency_ms.toFixed(1)} ms` : 'Unavailable'}</td>
                <td>{current ? current.map_cell_count.toLocaleString() : 'Unavailable'}</td>
                <td>{current?.resolution.average_resolution != null ? `${current.resolution.average_resolution.toFixed(3)} m` : 'Unavailable'}</td>
                <td className="tag">Current replay result</td>
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
          <p className="caption">
            Benchmark values shown here are loaded from the validated benchmark results and are not
            recalculated by the frontend. Source: {asset.provenance.source}
          </p>
        </>
      )}
    </section>
  );
}
