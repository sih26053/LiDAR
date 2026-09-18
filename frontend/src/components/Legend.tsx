import { RESOLUTION_STYLE, importanceColor, semanticColor } from '../utils/visualization';

/** Reusable legend for importance / resolution / semantic encodings. */
export function Legend() {
  return (
    <section className="panel legend" aria-label="Legend">
      <h2>Legend</h2>
      <div className="legend-grid">
        <div>
          <b>Importance</b>
          <ul>
            <li><i style={{ background: importanceColor(0.85) }} /> High (≥ 0.70)</li>
            <li><i style={{ background: importanceColor(0.55) }} /> Medium (0.45–0.70)</li>
            <li><i style={{ background: importanceColor(0.3) }} /> Low (0.20–0.45)</li>
            <li><i style={{ background: importanceColor(0.1) }} /> Minimal (&lt; 0.20)</li>
          </ul>
        </div>
        <div>
          <b>Resolution</b>
          <ul>
            {Object.values(RESOLUTION_STYLE).map((s) => (
              <li key={s.label}><i style={{ background: s.color }} /> {s.label} — {s.blurb}</li>
            ))}
          </ul>
        </div>
        <div>
          <b>Semantic</b>
          <ul>
            {[['vehicle', 'vehicle'], ['pedestrian_vru', 'pedestrian / VRU'], ['static_manmade', 'static man-made'], ['unknown', 'unknown / fallback']].map(([k, label]) => (
              <li key={k}><i style={{ background: semanticColor(k) }} /> {label}</li>
            ))}
            <li><i className="ring" /> white ring = valid (non-fallback) source</li>
          </ul>
        </div>
      </div>
    </section>
  );
}
