import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { TrackingResult } from '../types/api';

/**
 * Object tracking over consecutive in-scene samples. Runs the real
 * pipeline per frame, clusters dynamic cells, associates with persistent
 * IDs and measured velocity (POST /tracking/run). Track counts, gaps and
 * ID switches are reported, not hidden.
 */
export function TracksPanel() {
  const [scenes, setScenes] = useState<{ scene_id: string; nbr_samples: number; description: string }[]>([]);
  const [scene, setScene] = useState('scene-0061');
  const [count, setCount] = useState(6);
  const [mode, setMode] = useState<'annotation' | 'model'>('annotation');
  const [result, setResult] = useState<TrackingResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api.getTrackingScenes()
      .then((s) => { if (live) setScenes(s.scenes); })
      .catch(() => {});
    return () => { live = false; };
  }, []);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const out = await api.runTracking({ scene_id: scene, scene_samples: count, semantic_mode: mode });
      setResult(out);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Tracking failed.');
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  const speeds = (result?.tracks ?? []).flatMap((t) => t.states.map((s) => s.speed_m_s).filter((v): v is number => v !== null));
  const meanSpeed = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null;

  return (
    <section className="panel wide" aria-label="Object tracking">
      <h2>Object Tracking (measured velocity, persistent IDs)</h2>
      <div className="btn-row">
        <label className="field inline">
          Scene
          <select value={scene} onChange={(e) => setScene(e.target.value)} aria-label="tracking scene">
            {scenes.map((s) => (
              <option key={s.scene_id} value={s.scene_id}>{s.scene_id} ({s.nbr_samples} samples)</option>
            ))}
            {scenes.length === 0 && <option value={scene}>{scene}</option>}
          </select>
        </label>
        <label className="field inline">
          Frames
          <select value={count} onChange={(e) => setCount(Number(e.target.value))} aria-label="tracking frame count">
            {[4, 6, 8, 10].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <label className="field inline">
          Semantics
          <select value={mode} onChange={(e) => setMode(e.target.value as 'annotation' | 'model')} aria-label="tracking semantic mode">
            <option value="annotation">annotation</option>
            <option value="model">model</option>
          </select>
        </label>
        <button className="primary" onClick={() => void run()} disabled={running}>
          {running ? 'Tracking…' : 'Run tracking'}
        </button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {result ? (
        <>
          <ul className="kv">
            <li><span>Frames tracked</span><b>{result.frame_ids.length} ({result.frame_ids[0]?.slice(0, 8)}… → {result.frame_ids[result.frame_ids.length - 1]?.slice(0, 8)}…)</b></li>
            <li><span>Tracks</span><b>{result.n_tracks} ({result.n_associations} associations, {result.n_new_tracks} new, {result.n_scene_breaks} scene breaks)</b></li>
            <li><span>Mean measured speed</span><b>{meanSpeed !== null ? `${meanSpeed.toFixed(2)} m/s over ${speeds.length} links` : 'Not measured'}</b></li>
            <li><span>Gate</span><b>{result.parameters.gate_base_m} m + {result.parameters.gate_speed_m_s} m/s·dt, dt ≤ {result.parameters.max_dt_s} s, eps {result.parameters.cluster_eps_m} m</b></li>
          </ul>
          <table className="bench objs">
            <thead>
              <tr><th>ID</th><th>Class</th><th>Frames</th><th>Trajectory (x, y per frame)</th><th>Latest speed</th></tr>
            </thead>
            <tbody>
              {result.tracks.filter((t) => t.age_frames >= 2).slice(0, 12).map((t) => {
                const last = t.states[t.states.length - 1];
                return (
                  <tr key={t.id}>
                    <td>#{t.id}</td>
                    <td>{t.class}</td>
                    <td>{t.age_frames} (hits {t.hits}, misses {t.misses})</td>
                    <td className="mono">{t.states.map((s) => `(${s.x.toFixed(1)},${s.y.toFixed(1)})`).join(' → ')}</td>
                    <td>{last.speed_m_s !== null ? `${last.speed_m_s.toFixed(1)} m/s` : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="caption">
            Showing tracks seen in ≥ 2 frames ({result.tracks.filter((t) => t.age_frames >= 2).length} of {result.n_tracks});
            single-frame detections are new/unassociated by definition. {result.note}
          </p>
          {result.errors.length > 0 && (
            <p className="error" role="alert">Frame errors: {result.errors.map((e) => `${e.frame_id.slice(0, 8)}:${e.error_code}`).join(', ')}</p>
          )}
        </>
      ) : (
        !running && <p className="state">Pick a scene and press Run tracking for consecutive-sample tracks.</p>
      )}
    </section>
  );
}
