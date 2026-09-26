import { useState } from 'react';
import { api } from '../api/client';
import type { PipelineResult, RLDecision } from '../types/api';

/**
 * RL Decision panel: state vector + untrained-DQN Q-values + safety
 * override for the current frame's stored result. The UNTRAINED warning
 * is structural (not dismissible): Q-values demonstrate the data path
 * only and must never be used for control.
 */
export function RLPanel({ result }: { result: PipelineResult | null }) {
  const [d, setD] = useState<RLDecision | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const decide = async () => {
    if (!result) return;
    setBusy(true);
    setError(null);
    try {
      setD(await api.rlDecide(result.frame_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'RL decide failed.');
      setD(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel wide" aria-label="RL decision">
      <h2>RL Decision (state → DQN → safety override)</h2>
      {!result && <p className="state">Run a frame first, then request an RL decision.</p>}
      {result && (
        <div className="btn-row">
          <button className="primary" onClick={() => void decide()} disabled={busy}>
            {busy ? 'Deciding…' : `Decide from ${result.frame_id.slice(0, 8)}… (${result.semantic.mode})`}
          </button>
        </div>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      {d && (
        <>
          <ul className="kv">
            <li><span>State vector (13-dim)</span><b className="mono">[{d.state.state_vector.join(', ')}]</b></li>
            <li><span>Sectors (ranges m)</span><b className="mono">[{d.state.sector_ranges_m.join(', ')}]</b></li>
            <li><span>Obstacle / moving / static / terrain</span><b>{d.state.obstacle_density} / {d.state.moving_share} / {d.state.static_share} / {d.state.terrain_share} ({d.state.cells_in_radius}/{d.state.cells_total} cells in 30 m)</b></li>
            <li><span>DQN Q-values (untrained)</span><b className="mono">{d.decision.actions.map((a, i) => `${a}=${d.decision.q_values[i]}`).join(' · ')}</b></li>
            <li><span>Network action → final action</span><b>{d.decision.network_action} → {d.decision.final_action}{d.decision.safety_override ? ' (SAFETY OVERRIDE)' : ''}</b></li>
            <li><span>Safety check</span><b>{d.state.safety.emergency_stop ? `EMERGENCY STOP — nearest forward obstacle ${d.state.safety.nearest_forward_obstacle_m} m` : 'clear'} · decide latency {d.decide_latency_ms} ms</b></li>
          </ul>
          <p className="error" role="alert">
            Untrained network — Q-values carry no driving competence ({d.decision.model}). {d.decision.warning}
          </p>
        </>
      )}
    </section>
  );
}
