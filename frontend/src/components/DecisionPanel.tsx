import { useEffect, useState } from 'react';
import { api } from '../api/client';

interface Decision {
  frame_id: string;
  decision_model: string;
  proposed_action: string | null;
  confidence: number | null;
  answer_confidence?: number | null;
  mode?: string | null;
  eligible_actions?: string[] | null;
  raw_laya_action?: string | null;
  constrained_action?: string | null;
  gate?: string | null;
  gate_metric?: string | null;
  gate_threshold?: number | null;
  safety_status: string;
  safety_override?: boolean | null;
  executed_action: string;
  source?: string | null;
  decision_latency_ms: number | null;
}

interface SimState {
  simulator: string;
  decision_backend: string;
  carla?: { installed: boolean };
}

/**
 * Closed-loop decision panel: simulator, decision engine,
 * current action + recorded confidence, decision latency, safety,
 * executed action. Everything comes from real
 * endpoints; pre-execution states render as BLOCKED/empty, never as
 * fabricated values.
 */
export function DecisionPanel() {
  const [cur, setCur] = useState<Decision | null>(null);
  const [sim, setSim] = useState<SimState | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    let live = true;
    api.decisionCurrent()
      .then((d) => { if (live) { setCur(d); setMissing(false); } })
      .catch(() => { if (live) setMissing(true); });
    api.pybulletStatus()
      .then((s) => { if (live) setSim(s); })
      .catch(() => { /* leave null: backend down */ });
    return () => { live = false; };
  }, []);

  const fmtConf = (c: number | null) =>
    c === null || c === undefined || !Number.isFinite(c) ? 'n/a (no calibrated confidence)' : c.toFixed(3);

  return (
    <section className="panel wide" aria-label="Closed-loop decision">
      <h2>Closed-Loop Decision (PyBullet + Laya track)</h2>
      <ul className="kv">
        <li><span>Simulator</span><b>{sim ? sim.simulator : 'checking…'}</b></li>
        <li><span>Decision model</span><b>{sim ? sim.decision_backend : 'checking…'}</b></li>
        <li><span>Mode</span><b>{cur?.mode ?? 'n/a'}</b></li>
        <li><span>Raw Laya action</span><b>{cur?.raw_laya_action ?? 'n/a (constrained live path)'}</b></li>
        <li><span>Eligible actions</span><b>{cur?.eligible_actions ? cur.eligible_actions.join(', ') : 'n/a'}</b></li>
        <li><span>Constrained action</span><b>{cur?.constrained_action ?? cur?.proposed_action ?? 'none (safe fallback)'}</b></li>
        <li><span>Current action</span><b>{cur ? cur.proposed_action ?? 'none (safe fallback)' : 'none recorded'}</b></li>
        <li><span>Confidence</span><b>{cur ? fmtConf(cur.confidence) : 'n/a'}</b></li>
        <li><span>Gate</span><b>{cur?.gate ?? 'n/a'}{cur?.gate_metric ? ` (${cur.gate_metric} ≥ ${cur.gate_threshold})` : ''}</b></li>
        <li><span>Decision latency</span><b>{cur?.decision_latency_ms != null ? `${cur.decision_latency_ms} ms (measured)` : 'n/a'}</b></li>
        <li><span>Safety</span><b>{cur ? cur.safety_status : 'n/a'}</b></li>
        <li><span>Safety override</span><b>{cur?.safety_override == null ? 'n/a' : cur.safety_override ? 'YES' : 'NO'}</b></li>
        <li><span>Executed action</span><b>{cur ? cur.executed_action : 'none recorded'}</b></li>
        <li><span>Execution source</span><b>{cur?.source ?? 'n/a'}</b></li>
      </ul>
      {missing && (
        <p className="state">
          No decision recorded yet — run the replay simulation (Step), the PyBullet
          closed loop, or the live Laya loop. Laya runs locally; if it reports
          STARTING/ERROR, see the Live Control panel.
        </p>
      )}
      <p className="caption">PyBullet simulation is never described as physical testing. Physical testing: NOT EXECUTED.</p>
    </section>
  );
}
