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
          <b>Semantic class</b>
          <ul>
            {[['vehicle', 'vehicle'], ['pedestrian_vru', 'pedestrian / VRU'], ['static_manmade', 'static man-made'], ['unknown', 'unknown / fallback']].map(([k, label]) => (
              <li key={k}><i style={{ background: semanticColor(k) }} /> {label}</li>
            ))}
          </ul>
        </div>
        <div>
          <b>Semantic source</b>
          <ul>
            <li><i className="ring" /> white ring = valid (non-fallback) source</li>
            <li>annotation = nuScenes ground-truth/reference annotation</li>
            <li>fallback = heuristic rule where no annotation applied</li>
            <li>unknown = no semantic information</li>
            <li>model = trained point-classifier prediction (select “trained model” channel)</li>
          </ul>
        </div>
      </div>
      <p className="caption">
        Importance bands follow the frozen runtime thresholds (I ≥ 0.70 → 0.05 m;
        I ≥ 0.45 → 0.10 m; I ≥ 0.20 → 0.20 m; else 0.50 m — results/final/config/final_config.json).
      </p>
    </section>
  );
}
