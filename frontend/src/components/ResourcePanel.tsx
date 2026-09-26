import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { ResourceMetrics } from '../types/api';

/**
 * Resource Efficiency: fair same-frame comparison
 * (GET /resource/metrics <- results/resource/resource.json, measured by
 * scripts/measure_resources.py). Uniform memory is estimated from exact
 * cell counts (arrays never materialized) and labelled as such.
 */
function mb(v: number): string {
  return v >= 1048576 ? `${(v / 1048576).toFixed(1)} MB` : `${(v / 1024).toFixed(0)} KB`;
}

export function ResourcePanel() {
  const [m, setM] = useState<ResourceMetrics | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    let live = true;
    api.getResourceMetrics()
      .then((r) => { if (live) setM(r); })
      .catch(() => { if (live) setMissing(true); });
    return () => { live = false; };
  }, []);

  const maxCells = m ? Math.max(...m.methods.map((x) => x.mean_cells), 1) : 1;

  return (
    <section className="panel wide" aria-label="Resource efficiency">
      <h2>Resource Efficiency (same frames, measured)</h2>
      {missing && <p className="state">Resource benchmark not generated — run scripts/measure_resources.py.</p>}
      {!missing && !m && <p className="state">Loading resource metrics…</p>}
      {m && (
        <>
          <table className="bench">
            <thead>
              <tr><th>Method</th><th>Mean cells</th><th>Mean memory</th><th>Mean mapping latency</th><th>Mean FPS</th></tr>
            </thead>
            <tbody>
              {m.methods.map((r) => (
                <tr key={r.method} className={r.method === 'proposed' ? 'highlight' : ''}>
                  <td>{r.method === 'proposed' ? 'Proposed Adaptive' : r.method === 'distance_adaptive' ? 'Distance-based Adaptive' : 'Uniform 5 cm'}</td>
                  <td>{Math.round(r.mean_cells).toLocaleString()}</td>
                  <td>{mb(r.mean_memory_bytes)}{r.method === 'uniform_5cm' ? ' (est.)' : ''}</td>
                  <td>{r.mean_mapping_latency_ms.toFixed(1)} ms</td>
                  <td>{r.mean_fps.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="bars" aria-hidden="true">
            {m.methods.map((r) => (
              <div className="bar-row" key={r.method}>
                <span>{r.method === 'proposed' ? 'Proposed' : r.method === 'distance_adaptive' ? 'Distance' : 'Uniform'}</span>
                <div className="bar"><div style={{ width: `${(r.mean_cells / maxCells) * 100}%` }} /></div>
              </div>
            ))}
          </div>
          <ul className="kv">
            <li><span>Cell reduction (proposed vs uniform)</span><b>{m.cell_reduction_proposed_vs_uniform_pct.toFixed(2)}%</b></li>
            <li><span>Memory reduction (proposed vs uniform)</span><b>{m.memory_reduction_proposed_vs_uniform_pct.toFixed(2)}%</b></li>
          </ul>
          <p className="caption">{m.provenance}. {m.memory_note}.</p>
        </>
      )}
    </section>
  );
}
