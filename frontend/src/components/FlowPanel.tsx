import { useEffect, useState } from 'react';
import { api } from '../api/client';

interface FlowStage {
  box: number;
  name: string;
  status: string;
  detail: string;
  evidence: string;
}

const BADGE: Record<string, string> = {
  LIVE: 'ok',
  'LIVE — CARLA SIMULATION': 'ok',
  'IMPLEMENTED (untrained)': 'wait',
  'PARTIAL (safety rule live; no vehicle)': 'wait',
  'BLOCKED (offline replay evaluation instead)': 'bad',
};

/**
 * Methodology / Implementation Flow: the 9 boxes of the flow diagram,
 * each with a live status badge. BLOCKED stages name the missing
 * dependency (CARLA / vehicle hardware) instead of faking capability.
 */
export function FlowPanel() {
  const [stages, setStages] = useState<FlowStage[] | null>(null);
  const [outputs, setOutputs] = useState<string[]>([]);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    let live = true;
    api.getFlowStatus()
      .then((r) => { if (live) { setStages(r.stages); setOutputs(r.system_outputs); } })
      .catch(() => { if (live) setMissing(true); });
    return () => { live = false; };
  }, []);

  return (
    <section className="panel wide" aria-label="Methodology flow">
      <h2>Methodology / Implementation Flow (9 stages)</h2>
      {missing && <p className="state">Flow status unavailable — is the backend running?</p>}
      {!missing && !stages && <p className="state">Loading flow status…</p>}
      {stages && (
        <>
          <div className="flow-grid">
            {stages.map((s) => (
              <div className="flow-box" key={s.box}>
                <b>{s.box}. {s.name}</b>
                <span className={`pill ${BADGE[s.status] ?? 'wait'}`}>{s.status}</span>
                <p>{s.detail}</p>
                <p className="mono">↳ {s.evidence}</p>
              </div>
            ))}
          </div>
          <h3 className="sub">System outputs</h3>
          <p className="caption">{outputs.join(' · ')}</p>
          <p className="caption">
            Resolution correspondence (frozen validated config): High → 0.05 m (diagram 5 cm);
            Medium band → 0.10/0.20 m (covers diagram 20 cm); Low → 0.50 m
            (diagram shows 80 cm — ours is finer per the validated config).
            Perception classes shown as Road / Vehicle / Pedestrian / Static Obstacle / Others.
          </p>
        </>
      )}
    </section>
  );
}
