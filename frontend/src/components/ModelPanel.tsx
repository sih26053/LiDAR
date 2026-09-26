import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { ModelInfo } from '../types/api';

/**
 * Trained-model card. Every number comes from GET /model/info
 * (models/segmentation/ + results/segmentation/metrics.json) — measured
 * held-out metrics, never hard-coded. Weak spots (foreground F1) are
 * shown, not hidden.
 */
export function ModelPanel() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let live = true;
    api.getModelInfo()
      .then((m) => { if (live) setInfo(m); })
      .catch(() => { if (live) setError(true); });
    return () => { live = false; };
  }, []);

  return (
    <section className="panel wide" aria-label="Trained model">
      <h2>Trained Perception Model (measured, not claimed)</h2>
      {error && <p className="error" role="alert">Trained model unavailable — run scripts/train_segmentation_mlp.py then scripts/evaluate_segmentation.py.</p>}
      {!error && !info && <p className="state">Loading model card…</p>}
      {info && !info.metrics_held_out && (
        <p className="state">Weights present but held-out evaluation not generated — run scripts/evaluate_segmentation.py.</p>
      )}
      {info && info.metrics_held_out && (
        <>
          <ul className="kv">
            <li><span>Model</span><b>{info.model_name} ({info.model_file})</b></li>
            <li><span>Input</span><b>Per-point LiDAR features: {info.input_features.join(', ')}</b></li>
            <li><span>Output</span><b>Classes: {info.classes.join(', ')} (project taxonomy; vegetation never predicted — zero box support)</b></li>
            <li><span>Inference stage</span><b>{info.inference_stage}</b></li>
            <li><span>Supervision</span><b>{info.supervision}</b></li>
            <li><span>Not supervised</span><b>{info.not_supervised.join('; ')}</b></li>
            <li><span>Train / test</span><b>{info.n_train_points.toLocaleString()} pts ({info.train_frames.length} frames) / {info.n_test_points.toLocaleString()} pts ({info.test_frames.length} held-out frames)</b></li>
            <li><span>Held-out accuracy</span><b>{info.metrics_held_out.accuracy !== null ? `${(info.metrics_held_out.accuracy * 100).toFixed(1)}%` : 'n/a'} (reference-dominated — read F1 below)</b></li>
            <li><span>Held-out F1 macro</span><b>{info.metrics_held_out.f1_macro?.toFixed(3) ?? 'n/a'}</b></li>
            {Object.entries(info.metrics_held_out.f1_per_class).map(([c, f]) => (
              <li key={c}><span>F1 {c}</span><b>{f?.toFixed(3) ?? 'n/a (no support)'}</b></li>
            ))}
            <li><span>Mean predicted confidence</span><b>{info.metrics_held_out.mean_predicted_confidence?.toFixed(3) ?? 'n/a'}</b></li>
          </ul>
          <p className="caption">
            Confusion matrix (rows true, cols predicted {info.metrics_held_out.confusion_labels.join('/')}):{' '}
            {info.metrics_held_out.confusion_matrix_rows_true_cols_pred.map((r) => `[${r.join(', ')}]`).join(' ')}.
            Select “model” as the semantic channel in Replay Controls to map with these predictions;
            region uncertainty then equals 1 − predicted confidence.
          </p>
        </>
      )}
    </section>
  );
}
