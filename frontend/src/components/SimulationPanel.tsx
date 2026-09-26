import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { SimStatus, SimStep } from '../types/api';

/**
 * Closed-loop simulation console (offline replay loop). Every value is a
 * measured step result; CARLA rows report BLOCKED (not installed) and
 * unmeasurable rewards render as n/a (null, never zero-filled).
 */
export function SimulationPanel() {
  const [status, setStatus] = useState<SimStatus | null>(null);
  const [last, setLast] = useState<SimStep | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try { setStatus(await api.simulationStatus()); } catch { /* backend down */ }
  };

  useEffect(() => { void refresh(); }, []);

  const call = async (fn: () => Promise<SimStatus | SimStep>, isStep = false) => {
    setBusy(true);
    setError(null);
    try {
      const r = await fn();
      if (isStep) setLast(r as SimStep);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Simulation call failed.');
    } finally {
      setBusy(false);
    }
  };

  const reward = (v: unknown): string => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : 'n/a');

  return (
    <section className="panel wide" aria-label="Closed-loop simulation">
      <h2>Closed-Loop Simulation (offline replay loop)</h2>
      <div className="btn-row">
        <button onClick={() => void call(() => api.simulationStart())} disabled={busy}>Start</button>
        <button className="primary" onClick={() => void call(() => api.simulationStep(), true)} disabled={busy || !status?.active}>Step</button>
        <button onClick={() => void call(() => api.simulationStop())} disabled={busy}>Stop</button>
        <button onClick={() => void call(() => api.simulationReset())} disabled={busy}>Reset</button>
        <span className={`pill ${status?.active ? 'ok' : 'wait'}`}>{status?.active ? 'RUNNING' : 'IDLE'}</span>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {status && (
        <ul className="kv">
          <li><span>Episode</span><b>{status.episode_id ?? '—'} ({status.frames_done}/{status.frames_total} frames)</b></li>
          <li><span>Stops / safety overrides</span><b>{status.stops} / {status.safety_overrides}</b></li>
          <li><span>Mean decide latency</span><b>{status.mean_decide_latency_ms !== null ? `${status.mean_decide_latency_ms} ms` : 'n/a'}</b></li>
          <li><span>CARLA</span><b>{status.carla?.installed ? 'installed' : 'NOT INSTALLED — simulator execution blocked (docs/carla_setup.md)'}</b></li>
        </ul>
      )}
      {last && (
        <>
          <h3 className="sub">Last step: {last.frame_id.slice(0, 8)}… → {last.final_action} ({last.safety_verdict})</h3>
          <ul className="kv">
            <li><span>Stages executed</span><b>{last.stages_executed.join(' → ')}</b></li>
            <li><span>Q-values (untrained DQN)</span><b className="mono">{last.q_values.map((q) => q.toFixed(3)).join(' · ')}</b></li>
            <li><span>Stage latencies ms</span><b className="mono">{Object.entries(last.latency_ms).map(([k, v]) => `${k}=${v}`).join(' · ')}</b></li>
            <li><span>Rewards (measured only)</span><b>clearance={reward(last.rewards.safe_clearance)} · progress={reward(last.rewards.progress_towards_goal)} · collision={reward(last.rewards.collision)} · stops={reward(last.rewards.unnecessary_stop)}</b></li>
          </ul>
          <p className="caption">DQN untrained — actions demonstrate the data path only. Collisions/goals are null without ego control.</p>
        </>
      )}
    </section>
  );
}
