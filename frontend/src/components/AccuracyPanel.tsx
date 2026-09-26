import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { SegmentationMetrics } from '../types/api';

/**
 * Segmentation Performance: stored held-out evaluation
 * (GET /segmentation/metrics <- results/segmentation/metrics.json).
 * Reference = annotation-derived weak labels on unseen frames, NOT human
 * point labels. Null = unmeasurable (e.g. zero-support class), never zero-filled.
 */
function pct(v: number | null | undefined): string {
  return v === null || v === undefined || !Number.isFinite(v) ? 'n/a' : `${(v * 100).toFixed(1)}%`;
}
function num(v: number | null | undefined, digits = 3): string {
  return v === null || v === undefined || !Number.isFinite(v) ? 'n/a' : v.toFixed(digits);
}

export function AccuracyPanel() {
  const [m, setM] = useState<SegmentationMetrics | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    let live = true;
    api.getSegmentationMetrics()
      .then((r) => { if (live) setM(r); })
      .catch(() => { if (live) setMissing(true); });
    return () => { live = false; };
  }, []);

  return (
    <section className="panel wide" aria-label="Segmentation performance">
      <h2>Segmentation Performance (held-out frames, measured)</h2>
      {missing && <p className="state">Evaluation not generated — run scripts/evaluate_segmentation.py.</p>}
      {!missing && !m && <p className="state">Loading segmentation metrics…</p>}
      {m && (
        <>
          <ul className="kv">
            <li><span>Overall accuracy vs reference</span><b>{pct(m.overall_accuracy_vs_reference)} ({m.n_eval_points.toLocaleString()} pts, {m.eval_frames.length} unseen frames)</b></li>
            <li><span>Mean IoU vs reference</span><b>{num(m.mean_iou_vs_reference)}</b></li>
            <li><span>Mean confidence</span><b>{num(m.mean_confidence)}</b></li>
            {m.per_class.map((c) => (
              <li key={c.class}><span>{c.class} (support {c.support.toLocaleString()})</span><b>P {num(c.precision)} · R {num(c.recall)} · F1 {num(c.f1)} · IoU {num(c.iou)}</b></li>
            ))}
          </ul>
          <h3 className="sub">Accuracy by distance</h3>
          <table className="bench">
            <thead>
              <tr><th>Bin</th><th>Points</th><th>Accuracy vs ref</th><th>Mean IoU</th><th>Mean conf</th><th>Status</th></tr>
            </thead>
            <tbody>
              {m.distance_bins.map((b) => (
                <tr key={b.distance_bin_m}>
                  <td>{b.distance_bin_m} m</td>
                  <td>{b.n_points.toLocaleString()}</td>
                  <td>{pct(b.accuracy_vs_reference)}</td>
                  <td>{num(b.mean_iou)}</td>
                  <td>{num(b.mean_confidence)}</td>
                  <td className="tag">{b.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="caption">
            Reference: {m.reference} Vegetation: {m.classes_without_support.join('; ')}.
            Pedestrian over-prediction (low precision, high recall) is a known weakness, shown above.
          </p>
        </>
      )}
    </section>
  );
}
