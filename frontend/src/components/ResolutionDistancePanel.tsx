import { resolutionByDistance } from '../utils/terrain';
import type { PipelineResult } from '../types/api';

/**
 * Resolution-vs-distance analysis. Read-only view of backend output:
 * mean resolution per distance band. Allocation itself stays importance-driven
 * (frozen THRESH_C); distance is shown for analysis, and the distance-based
 * adaptive baseline lives in the benchmark comparison panel.
 */
export function ResolutionDistancePanel({ result }: { result: PipelineResult | null }) {
  const bands = result ? resolutionByDistance(result.map_cells) : [];
  const max = Math.max(...bands.map((b) => b.meanResolution ?? 0), 1e-6);
  return (
    <section className="panel" id="panel-resdist" aria-label="Resolution versus distance">
      <h2>Resolution vs Distance (measured)</h2>
      {result ? (
        <>
          <table className="bench">
            <thead><tr><th>Distance band</th><th>Cells</th><th>Mean resolution</th></tr></thead>
            <tbody>
              {bands.map((b) => (
                <tr key={b.label}>
                  <td>{b.label}</td>
                  <td>{b.cells.toLocaleString()}</td>
                  <td>{b.meanResolution != null ? `${b.meanResolution.toFixed(3)} m` : 'Unavailable'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="bars" aria-hidden="true">
            {bands.map((b) => (
              <div className="bar-row" key={b.label}>
                <span>{b.label}</span>
                <div className="bar"><div style={{ width: `${((b.meanResolution ?? 0) / max) * 100}%` }} /></div>
              </div>
            ))}
          </div>
          <p className="caption">
            Measured from backend cell output. Cell size is allocated by importance
            (frozen thresholds 0.70/0.45/0.20), not by distance; the distance-based
            allocation strategy is compared separately in the benchmark panel.
          </p>
        </>
      ) : (
        <p className="state">No result available for this frame. Select a frame and press Run.</p>
      )}
    </section>
  );
}
