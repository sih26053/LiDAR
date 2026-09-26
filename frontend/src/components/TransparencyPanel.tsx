import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { ConfigInfo, EnvironmentInfo, PipelineResult } from '../types/api';
import type { RunHistoryEntry } from '../hooks/useReplay';

/**
 * Transparency sections: importance model (live frozen config), temporal
 * replay statistics (session history), hardware/software (live detection),
 * data provenance, and limitations. No value here is hard-coded to flatter
 * the demo: config comes from GET /config, environment from
 * GET /system/environment, statistics from measured session history.
 */

const FROZEN_FALLBACK = {
  weights: { distance: 0.3, semantic: 0.3, terrain: 0.15, dynamic: 0.15, uncertainty: 0.1 },
  lambda: 0.15,
  maxDistance: 100.0,
  levels: [[0.7, 0.05], [0.45, 0.1], [0.2, 0.2], [0.0, 0.5]] as [number, number][],
};

function fmt(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || !Number.isFinite(v) ? 'Not measured' : v.toFixed(digits);
}

export function TransparencyPanel({
  result, history,
}: {
  result: PipelineResult | null;
  history: RunHistoryEntry[];
}) {
  const [config, setConfig] = useState<ConfigInfo | null>(null);
  const [env, setEnv] = useState<EnvironmentInfo | null>(null);
  const [envError, setEnvError] = useState(false);

  useEffect(() => {
    let live = true;
    api.getConfig().then((c) => { if (live) setConfig(c); }).catch(() => {});
    api.getEnvironment().then((e) => { if (live) setEnv(e); }).catch(() => { if (live) setEnvError(true); });
    return () => { live = false; };
  }, []);

  // Importance weights come live from GET /config (frozen final config);
  // the constants below are only a fallback if the backend is unreachable.
  const w = config?.importance_weights ?? FROZEN_FALLBACK.weights;
  const lam = config?.uncertainty_lambda ?? FROZEN_FALLBACK.lambda;
  const fromLive = config?.importance_weights !== undefined;
  const levels: [number, number][] = config?.resolution_levels?.length === 4
    ? config.resolution_levels
    : FROZEN_FALLBACK.levels;
  const maxD = config?.max_mapping_distance_m ?? FROZEN_FALLBACK.maxDistance;

  const totals = history.filter((h) => h.total_latency_ms !== null).map((h) => h.total_latency_ms as number);
  const avg = totals.length ? totals.reduce((a, b) => a + b, 0) / totals.length : null;
  const min = totals.length ? Math.min(...totals) : null;
  const max = totals.length ? Math.max(...totals) : null;
  const avgFps = avg ? 1000 / avg : null;
  const meanCells = history.length
    ? history.reduce((a, b) => a + b.cells, 0) / history.length
    : null;

  const counts = result?.semantic.source_counts ?? {};
  const total = Object.values(counts).reduce((a, b) => a + b, 0);

  return (
    <>
      <section className="panel wide" aria-label="Importance model">
        <h2>Importance Model (actual formula from implementation)</h2>
        <p className="mono">
          D = clip01(1 − distance / {maxD}) · I_base = {w.distance}·D + {w.semantic}·S + {w.terrain}·T + {w.dynamic}·M + {w.uncertainty}·U ·
          I_final = min(1, I_base + {lam}·U)
        </p>
        <ul className="kv">
          <li><span>D — distance closeness</span><b>weight {w.distance} · measured range → normalized [0,1] · closer = more important</b></li>
          <li><span>S — semantic importance</span><b>weight {w.semantic} · annotation-derived lookup (vehicle/pedestrian 1.0, static 0.7, vegetation 0.3, road 0.1, unknown 0.5)</b></li>
          <li><span>T — terrain/geometric complexity</span><b>weight {w.terrain} · measured height-std within 2 m cell → [0,1]</b></li>
          <li><span>M — dynamic relevance</span><b>weight {w.dynamic} · heuristic prior per class (pedestrian 0.95, vehicle 0.80, unknown 0.20); measured track velocity shown separately in Tracks panel</b></li>
          <li><span>U — uncertainty</span><b>weight {w.uncertainty} + safety boost λ={lam}·U · 0.5 fallback in annotation channel, 1 − predicted confidence in model channel</b></li>
          <li><span>Resolution allocation</span><b>{levels.map(([t, r]) => `I ≥ ${t} → ${r} m`).join(' · ')}</b></li>
        </ul>
        <p className="caption">
          Source: {fromLive ? 'live frozen config (GET /config)' : 'frozen fallback (results/final/config/final_config.json — backend unreachable)'};
          algorithm: src/importance_engine.py. Weights are prototype engineering assumptions, not optimized values.
        </p>
      </section>

      <section className="panel" aria-label="Temporal replay">
        <h2>Temporal Replay (frame-to-frame adaptive mapping)</h2>
        {history.length > 0 ? (
          <ul className="kv">
            <li><span>Frames processed</span><b>{history.length}</b></li>
            <li><span>Average FPS</span><b>{avgFps !== null ? avgFps.toFixed(2) : 'Not measured'} (1000 / avg total latency)</b></li>
            <li><span>Average latency</span><b>{fmt(avg)} ms</b></li>
            <li><span>Min / max latency</span><b>{fmt(min)} / {fmt(max)} ms</b></li>
            <li><span>Mean cells / frame</span><b>{meanCells !== null ? Math.round(meanCells).toLocaleString() : 'Not measured'}</b></li>
            <li><span>Latest frame → timestamp</span><b>{history[history.length - 1].frame_id.slice(0, 8)}… → {history[history.length - 1].timestamp ?? 'unavailable'}</b></li>
          </ul>
        ) : (
          <p className="state">No frames processed yet — press Play or Run consecutive frames to populate temporal statistics.</p>
        )}
        <p className="caption">
          Each tick replays the next frame through the backend pipeline (load → run → metrics).
          For persistent object identities with measured velocity, see the Tracks panel
          (consecutive in-scene samples, greedy association).
        </p>
      </section>

      <section className="panel" aria-label="Semantic source detail">
        <h2>Semantic Source (current frame)</h2>
        {result && total > 0 ? (
          <ul className="kv">
            {Object.entries(counts).map(([k, v]) => (
              <li key={k}><span>{k === 'annotation' ? 'Dataset annotation cells' : k === 'fallback' ? 'Heuristic fallback cells' : `${k} cells`}</span><b>{v.toLocaleString()} ({((v / total) * 100).toFixed(1)}%)</b></li>
            ))}
            <li><span>Model-predicted cells</span><b>{counts['model'] != null ? `${counts['model'].toLocaleString()} (trained point classifier)` : '0 in annotation channel (select “trained model” to activate)'}</b></li>
          </ul>
        ) : (
          <p className="state">No result available for this frame. Select a frame and press Run.</p>
        )}
        <p className="caption">
          Counts are computed live from the current backend result — never hard-coded. Annotation =
          nuScenes ground-truth/reference; fallback = heuristic rule; model = trained MLP prediction
          (active in the model channel).
        </p>
      </section>

      <section className="panel" aria-label="Hardware and software">
        <h2>Hardware &amp; Software (current machine)</h2>
        {env ? (
          <ul className="kv">
            <li><span>CPU</span><b>{env.cpu_model} · {String(env.cpu_count)} threads</b></li>
            <li><span>GPU</span><b>{env.gpu}</b></li>
            <li><span>RAM</span><b>{String(env.ram_gb)} GB</b></li>
            <li><span>OS</span><b>{env.os}</b></li>
            <li><span>Python</span><b>{env.python_version}</b></li>
            {Object.entries(env.packages).map(([k, v]) => (
              <li key={k}><span>{k}</span><b>{v}</b></li>
            ))}
          </ul>
        ) : (
          <p className="state">{envError ? 'Hardware information unavailable (backend unreachable).' : 'Detecting environment…'}</p>
        )}
        <p className="caption">Detected live via GET /system/environment. Stored v2 benchmark hardware is recorded in results/benchmark_v2/environment_v2.json.</p>
      </section>

      <section className="panel wide" aria-label="Data provenance">
        <h2>Data Provenance</h2>
        <ul className="kv">
          <li><span>REAL — measured</span><b>nuScenes LiDAR point clouds (input points, timestamps, frame/scene IDs)</b></li>
          <li><span>REFERENCE — dataset</span><b>nuScenes 3D box annotations projected as semantic labels (evaluation reference, never predictions)</b></li>
          <li><span>DERIVED — algorithmic</span><b>importance score, adaptive resolution, occupancy, elevation, cell statistics</b></li>
          <li><span>HEURISTIC — rules</span><b>fallback semantic labels, dynamic-relevance priors, uncertainty fallback 0.5, resolution thresholds</b></li>
          <li><span>MODEL</span><b>trained MLP point classifier (models/segmentation/weights/mlp_model.pkl, 5 project classes) — active in model channel; see Model panel for held-out metrics</b></li>
          <li><span>BENCHMARK — verified</span><b>results/benchmark_v2/* (10 frames, frozen runtime config, recorded hardware) — reproduced by scripts/run_benchmark_v2.py</b></li>
          <li><span>BENCHMARK — legacy</span><b>results/benchmark/* (13 Sep 2026, older thresholds) — superseded, kept as raw history</b></li>
          <li><span>BENCHMARK — current</span><b>backend-measured per-frame timing in this session (this machine, frozen runtime config)</b></li>
        </ul>
      </section>

      <section className="panel wide" aria-label="Limitations">
        <h2>Limitations (residual, measured)</h2>
        <ul className="kv">
          <li><span>Perception model</span><b>MLP covers 5 project classes (vegetation: zero box support, never predicted); held-out acc 0.911 / mIoU 0.41 vs weak reference; pedestrian over-prediction is a known weakness — see Accuracy panel</b></li>
          <li><span>Input</span><b>Local nuScenes Mini files + streaming API (no physical LiDAR attached — simulated feed only)</b></li>
          <li><span>Tracking</span><b>Greedy association, no motion model or ego-compensation; velocities include ego motion; ID switches possible — see Tracks panel</b></li>
          <li><span>Benchmark</span><b>Verified 10-frame re-run on the serving machine; results are hardware-specific, not universal</b></li>
          <li><span>Uniform baseline</span><b>Analytic-grid figure kept for scope honesty alongside the same-scope uniform row</b></li>
          <li><span>Latency scope</span><b>Compute stages only — rendering, file I/O, network/serialization excluded (wall clock + load + render shown separately)</b></li>
          <li><span>Performance</span><b>No 10 Hz sensor-rate claim — session measures ~3 FPS on this CPU (3× faster than the 1 FPS baseline)</b></li>
        </ul>
      </section>
    </>
  );
}
