import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { layaStatusOf, useLiveSimulation } from '../hooks/useLiveSimulation';
import { AdaptiveMapView } from './AdaptiveMapView';
import { ImportanceView } from './ImportanceView';
import { ResolutionView } from './ResolutionView';
import type { LayaServerStatus, PipelineResult } from '../types/api';

/**
 * Live PyBullet + Laya console.
 * Two explicitly separated modes: LAYA AUTONOMOUS MODE vs MANUAL CONTROL
 * (manual actions are labelled MANUAL TEST and never presented as Laya
 * decisions). Every value comes from the live backend snapshot
 * (GET /simulation/state or /ws/live); pre-start states render as
 * "—", never as fabricated numbers. The 2.5D map, importance and
 * resolution views reuse the existing visualizations fed by the live
 * pipeline_result of the CURRENT frame — never prerecorded data.
 */
export function LiveControlPanel() {
  const [live, setLive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [laya, setLaya] = useState<LayaServerStatus | null>(null);
  const [checkpoint, setCheckpoint] = useState<Record<string, unknown> | null>(null);
  const { snapshot: s, connection } = useLiveSimulation(live);

  useEffect(() => {
    let on = true;
    api.layaStatus()
      .then((st) => { if (on) setLaya(st); })
      .catch(() => undefined);
    api.layaCheckpoint()
      .then((cp) => { if (on) setCheckpoint(cp); })
      .catch(() => undefined);
    const t = setInterval(() => {
      api.layaStatus()
        .then((st) => { if (on) setLaya(st); })
        .catch(() => undefined);
    }, 5000);
    return () => { on = false; clearInterval(t); };
  }, []);

  const call = async (fn: () => Promise<unknown>, okMsg: string) => {
    setBusy(true);
    setNotice(null);
    try {
      await fn();
      setNotice(okMsg);
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Live call failed.');
    } finally {
      setBusy(false);
    }
  };

  const num = (v: number | null | undefined, digits = 3): string =>
    typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';

  const probs = s?.jev_probabilities
    ? Object.entries(s.jev_probabilities).map(([k, v]) => `${k}=${num(v, 2)}`).join(' · ')
    : '—';

  const liveResult = (s?.pipeline_result ?? null) as PipelineResult | null;
  const m = s?.metrics;
  const v = s?.vehicle;

  const stageLabel = (key: string): string => {
    const st = s?.pipeline?.[key];
    if (!st) return '—';
    return `${st.state}${st.verified ? ' (verified)' : ''}`;
  };

  return (
    <section className="panel wide" aria-label="Live PyBullet and Laya control">
      <h2>Live Control — PyBullet + Laya</h2>
      <div className="btn-row">
        <button
          className="primary"
          onClick={() => void call(() => api.liveStart('autonomous').then(() => setLive(true)), 'Laya autonomous loop started.')}
          disabled={busy}
        >
          Start Laya Autonomous
        </button>
        <button
          onClick={() => void call(() => api.liveStart('manual').then(() => setLive(true)), 'Manual mode started. Laya will NOT decide.')}
          disabled={busy}
        >
          Start Manual
        </button>
        <button onClick={() => void call(() => api.liveStop().then(() => setLive(false)), 'Live loop stopped.')} disabled={busy}>
          Stop
        </button>
        <button onClick={() => void call(() => api.liveReset().then(() => setLive(true)), 'Live loop reset.')} disabled={busy}>
          Reset
        </button>
        <span className={`pill ${s?.frame_id ? 'ok' : 'wait'}`}>
          {s?.frame_id ? `RUNNING (${s.mode})` : 'IDLE'}
        </span>
        <span className="pill">{connection === 'ws' ? 'WS LIVE' : connection === 'polling' ? 'POLLING' : 'OFF'}</span>
      </div>
      {notice && <p className="state">{notice}</p>}

      <h3 className="sub">Laya — local decision engine</h3>
      <ul className="kv">
        <li><span>Decision engine</span><b>Laya (local{ s?.decision?.backend ? `, ${s.decision.backend}` : ''})</b></li>
        <li><span>Model</span><b>{s?.decision?.model ?? laya?.model ?? '—'}</b></li>
        <li><span>Endpoint</span><b className="mono">{s?.decision?.endpoint ?? '—'}</b></li>
        <li><span>Inference device</span><b>{s?.decision?.device ?? laya?.device ?? '—'}</b></li>
        <li><span>Server</span><b>{laya ? (laya.independent ? 'INDEPENDENT process' : laya.managed_process ? 'MANAGED child (dev)' : 'NOT RUNNING') : 'checking…'}</b></li>
        <li><span>Laya status</span><b>{laya ? `${laya.state} (model ${laya.model})` : 'checking…'}</b></li>
        <li><span>Server PID</span><b>{laya?.pid ?? '—'}</b></li>
        <li><span>Uptime</span><b>{laya?.uptime_s != null ? `${Math.round(laya.uptime_s)} s (measured)` : '—'}</b></li>
        <li><span>Restart count</span><b>{laya?.restart_count ?? '—'}</b></li>
        <li><span>Checkpoint</span><b>{laya?.checkpoint ?? (checkpoint?.state as string) ?? '—'}</b></li>
        <li><span>Revision</span><b className="mono">{s?.decision?.checkpoint_revision ?? (laya?.checkpoint_revision as string) ?? (checkpoint?.revision as string) ?? '—'}</b></li>
        <li><span>Offline mode</span><b>{laya?.offline ?? (checkpoint?.offline_inference as string) ?? '—'}</b></li>
        <li><span>Calibration</span><b>{laya?.calibration?.gate.state ?? 'NOT LOADED'}</b></li>
        <li><span>Gate metric</span><b>{s?.decision?.gate_metric ?? laya?.calibration?.gate.metric ?? '—'}</b></li>
        <li><span>Gate threshold</span><b>{s?.decision?.gate_threshold ?? laya?.calibration?.gate.threshold ?? '—'}</b></li>
        <li><span>Status</span><b>{layaStatusOf(s)}</b></li>
        <li><span>Current action</span><b>{s?.execution?.proposed_action ?? s?.jev_action ?? '—'}</b></li>
        <li><span>Confidence</span><b>{s?.decision?.confidence != null ? num(s.decision.confidence) : s?.jev_confidence != null ? num(s.jev_confidence) : '—'}</b></li>
        <li><span>Probabilities</span><b className="mono">{probs}</b></li>
        <li><span>Decision latency</span><b>{m?.laya_latency_ms != null ? `${num(m.laya_latency_ms, 2)} ms (measured)` : s?.jev_latency_ms != null ? `${num(s.jev_latency_ms, 2)} ms (measured)` : '—'}</b></li>
      </ul>

      <h3 className="sub">Decision chain — raw model vs constrained execution</h3>
      <ul className="kv">
        <li><span>Mode</span><b>{s?.decision?.mode ?? '—'}</b></li>
        <li><span>Raw Laya action</span><b>{s?.decision?.raw_action ?? '— (constrained live path: no raw candidate)'}</b></li>
        <li><span>Raw confidence</span><b>{s?.decision?.raw_confidence != null ? num(s.decision.raw_confidence) : '—'}</b></li>
        <li><span>Answer confidence</span><b>{s?.decision?.answer_confidence != null ? num(s.decision.answer_confidence) : '—'}</b></li>
        <li><span>Eligible actions</span><b>{s?.decision?.eligible_actions ? s.decision.eligible_actions.join(', ') : '—'}{s?.decision?.category ? ` (${s.decision.category})` : ''}</b></li>
        <li><span>Constrained action</span><b>{s?.decision?.constrained_action ?? '—'}{s?.decision?.laya_skipped ? ' (deterministic, no inference)' : ''}</b></li>
        <li><span>Execution source</span><b>{s?.source ?? '—'}</b></li>
      </ul>

      <h3 className="sub">Safety</h3>
      <ul className="kv">
        <li><span>Proposed action</span><b>{s?.execution?.proposed_action ?? '—'}</b></li>
        <li><span>Safety decision</span><b>{s?.safety?.status ?? s?.safety_status ?? '—'}</b></li>
        <li><span>Safety override</span><b>{s?.safety ? (s.safety.override ? 'YES' : 'NO') : '—'}</b></li>
        <li><span>Override reason</span><b>{s?.safety?.reason ?? '—'}</b></li>
        <li><span>Executed action</span><b>{s?.execution?.executed_action ?? s?.executed_action ?? '—'}{s?.source ? ` (source=${s.source})` : ''}</b></li>
      </ul>

      <h3 className="sub">Manual Control — MANUAL TEST (not Laya)</h3>
      <div className="btn-row">
        {(['forward', 'left', 'right', 'stop'] as const).map((a) => (
          <button key={a} onClick={() => void call(() => api.manualAction(a), `Manual ${a} sent (source=manual).`)} disabled={busy}>
            {a[0].toUpperCase() + a.slice(1)}
          </button>
        ))}
      </div>

      <h3 className="sub">Vehicle — PyBullet (live)</h3>
      <ul className="kv">
        <li><span>X / Y / Z (m)</span><b>{v ? `${num(v.x)} / ${num(v.y)} / ${num(v.z)}` : '—'}</b></li>
        <li><span>Yaw (deg)</span><b>{v ? num(v.yaw_deg, 2) : '—'}</b></li>
        <li><span>Speed (m/s)</span><b>{v ? num(v.speed_mps) : '—'}</b></li>
        <li><span>Yaw rate (rad/s)</span><b>{v ? num(v.yaw_rate) : '—'}</b></li>
        <li><span>Collision</span><b>{s?.collision == null ? '—' : s.collision ? 'CONTACT (obstacle)' : 'none'}</b></li>
      </ul>

      <h3 className="sub">LiDAR (live)</h3>
      <ul className="kv">
        <li><span>Frame ID</span><b>{s?.frame_id ?? '—'}</b></li>
        <li><span>Point count</span><b>{s?.lidar?.point_count ?? s?.lidar_points ?? '—'}</b></li>
        <li><span>Frame rate</span><b>{s?.lidar?.fps != null ? `${num(s.lidar.fps, 1)} fps (measured)` : '—'}</b></li>
        <li><span>Map cells</span><b>{s?.map?.cell_count ?? s?.map_cells ?? '—'}</b></li>
        <li><span>Simulation time</span><b>{s?.simulation_time != null ? `${num(s.simulation_time, 2)} s` : '—'}</b></li>
      </ul>

      <h3 className="sub">Pipeline status (runtime state)</h3>
      <ul className="kv">
        {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => (
          <li key={n}><span>Stage {n}</span><b>{stageLabel(`stage${n}`)}</b></li>
        ))}
      </ul>

      <h3 className="sub">Metrics (measured; NOT AVAILABLE where unmeasured)</h3>
      <ul className="kv">
        <li><span>Laya latency</span><b>{m?.laya_latency_ms != null ? `${num(m.laya_latency_ms, 2)} ms` : 'NOT AVAILABLE'}</b></li>
        <li><span>Perception latency</span><b>{m?.perception_latency_ms != null ? `${num(m.perception_latency_ms, 2)} ms` : 'NOT AVAILABLE'}</b></li>
        <li><span>Safety latency</span><b>{m?.safety_latency_ms != null ? `${num(m.safety_latency_ms, 3)} ms` : 'NOT AVAILABLE'}</b></li>
        <li><span>Action-exec latency</span><b>{m?.action_execution_latency_ms != null ? `${num(m.action_execution_latency_ms, 2)} ms` : 'NOT AVAILABLE'}</b></li>
        <li><span>Loop latency</span><b>{m?.loop_latency_ms != null ? `${num(m.loop_latency_ms, 2)} ms` : 'NOT AVAILABLE'}</b></li>
        <li><span>Vehicle speed</span><b>{m?.vehicle_speed_mps != null ? `${num(m.vehicle_speed_mps)} m/s` : 'NOT AVAILABLE'}</b></li>
        <li><span>Distance</span><b>{m?.distance_m != null ? `${num(m.distance_m, 2)} m` : 'NOT AVAILABLE'}</b></li>
        <li><span>Collisions</span><b>{m?.collisions_total ?? 'NOT AVAILABLE'}</b></li>
        <li><span>Decision calls / success / failed</span><b>{m ? `${m.laya_calls} / ${m.laya_successful} / ${m.laya_failed}` : 'NOT AVAILABLE'}</b></li>
        <li><span>Directional / STOP</span><b>{m ? `${m.directional_actions} / ${m.stop_actions}` : 'NOT AVAILABLE'}</b></li>
        <li><span>Safety overrides / manual</span><b>{m ? `${m.safety_overrides} / ${m.manual_calls}` : 'NOT AVAILABLE'}</b></li>
      </ul>

      <h3 className="sub">Live 2.5D map + importance + resolution (current frame)</h3>
      {liveResult ? (
        <>
          <AdaptiveMapView result={liveResult} />
          <ImportanceView result={liveResult} />
          <ResolutionView result={liveResult} />
        </>
      ) : (
        <p className="state">No live frame yet — start the loop to stream the current PyBullet map. Never prerecorded.</p>
      )}
      <p className="caption">Physical testing: NOT EXECUTED. Manual actions are human commands, never Laya decisions.</p>
    </section>
  );
}
