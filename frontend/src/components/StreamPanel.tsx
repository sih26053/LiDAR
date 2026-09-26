import { useState } from 'react';
import { api } from '../api/client';
import type { FrameInfo, PipelineResult } from '../types/api';
import { readPointFile } from '../utils/stream';

/**
 * Live-stream ingestion demo. Two honest paths into the streaming API:
 * simulated (a replay frame pushed server-side through the live path, no
 * annotations) and external (user-supplied .npy/.json point cloud).
 * Both run the same pipeline with no dataset attached.
 */
export function StreamPanel({ frames }: { frames: FrameInfo[] }) {
  const [mode, setMode] = useState<'annotation' | 'model'>('model');
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [simulated, setSimulated] = useState<boolean | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const simulate = async (frameId: string) => {
    setRunning(true);
    setError(null);
    try {
      const { result: r } = await api.streamSimulate(frameId, mode);
      setResult(r);
      setSimulated(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Stream failed.');
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  const upload = async (file: File) => {
    setRunning(true);
    setError(null);
    try {
      const points = await readPointFile(file);
      const { result: r } = await api.streamPush(points, mode);
      setResult(r);
      setSimulated(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Stream failed.');
      setResult(null);
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="panel wide" aria-label="Live stream">
      <h2>Live Stream (no dataset attached)</h2>
      <div className="btn-row">
        <label className="field inline">
          Semantics
          <select value={mode} onChange={(e) => setMode(e.target.value as 'annotation' | 'model')} aria-label="stream semantic mode">
            <option value="model">trained model</option>
            <option value="annotation">annotation channel (all-fallback live)</option>
          </select>
        </label>
        <button
          className="primary"
          disabled={running || frames.length === 0}
          onClick={() => { const f = frames[0]; if (f) void simulate(f.frame_id); }}
          title="Push a replay frame through the live path server-side (no annotations)"
        >
          {running ? 'Streaming…' : 'Simulate live feed'}
        </button>
        <label className="field inline">
          External cloud (.npy Nx4 f8 / .json)
          <input
            type="file"
            accept=".npy,.json"
            disabled={running}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void upload(f); e.target.value = ''; }}
            aria-label="upload point cloud"
          />
        </label>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {result ? (
        <ul className="kv">
          <li><span>Input</span><b>{simulated ? 'SIMULATED (replay frame via live path)' : 'EXTERNAL live point cloud'} · {result.input_source ?? 'live-stream'}</b></li>
          <li><span>Points → cells</span><b>{result.input_point_count?.toLocaleString() ?? '—'} → {result.map_cell_count.toLocaleString()}</b></li>
          <li><span>Semantic mode</span><b>{result.semantic.mode} · {Object.entries(result.semantic.source_counts).map(([k, v]) => `${k}=${v}`).join(', ')}</b></li>
          <li><span>Total latency</span><b>{result.timing.total_latency_ms?.toFixed(1) ?? '—'} ms (compute, backend-measured)</b></li>
        </ul>
      ) : (
        !running && <p className="state">No live physical LiDAR is attached. Simulate the feed or upload a cloud to exercise the streaming interface.</p>
      )}
      <p className="caption">
        Live inputs carry no dataset boxes: the annotation channel is all-fallback by construction;
        the model channel runs the trained classifier. Same pipeline, same timing methodology.
      </p>
    </section>
  );
}
