import { useEffect, useState } from 'react';
import { api } from '../api/client';

interface Decision {
  frame_id: string;
  decision_model: string;
  proposed_action: string | null;
  confidence: number | null;
  safety_status: string;
  executed_action: string;
  decision_latency_ms: number | null;
}

interface SimState {
  simulator: string;
  decision_backend: string;
  carla?: { installed: boolean };
}

/**
 * Closed-loop decision panel (Phase 23): simulator, decision model,
 * current action + recorded confidence, decision latency, safety,
 * executed action, live LiDAR/map stats. Everything comes from real
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
      <h2>Closed-Loop Decision (PyBullet + Jev track)</h2>
      <ul className="kv">
        <li><span>Simulator</span><b>{sim ? sim.simulator : 'checking…'}</b></li>
        <li><span>Decision model</span><b>{sim ? sim.decision_backend : 'checking…'}</b></li>
        <li><span>Current action</span><b>{cur ? cur.proposed_action ?? 'none (safe fallback)' : 'none recorded'}</b></li>
        <li><span>Confidence</span><b>{cur ? fmtConf(cur.confidence) : 'n/a'}</b></li>
        <li><span>Decision latency</span><b>{cur?.decision_latency_ms != null ? `${cur.decision_latency_ms} ms (measured)` : 'n/a'}</b></li>
        <li><span>Safety</span><b>{cur ? cur.safety_status : 'n/a'}</b></li>
        <li><span>Executed action</span><b>{cur ? cur.executed_action : 'none recorded'}</b></li>
      </ul>
      {missing && (
        <p className="state">
          No decision recorded yet — run the replay simulation (Step) or the PyBullet
          closed loop. PyBullet execution is BLOCKED on this host (no wheel for Python 3.14.4);
          Jev is BLOCKED (TYPESAFE_API_KEY not set).
        </p>
      )}
      <p className="caption">PyBullet simulation is never described as physical testing. Physical testing: NOT EXECUTED.</p>
    </section>
  );
}
