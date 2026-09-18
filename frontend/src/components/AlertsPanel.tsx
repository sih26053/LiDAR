import { deriveAlerts } from '../utils/terrain';
import type { SessionEvent } from '../hooks/useReplay';
import type { PipelineResult } from '../types/api';

/** Panel 8 — Recent Events / Alerts: backend-derived alerts + session event feed. */
export function AlertsPanel({ result, events }: { result: PipelineResult | null; events: SessionEvent[] }) {
  const alerts = result ? deriveAlerts(result) : [];
  return (
    <section className="panel" id="panel-alerts" aria-label="Recent events and alerts">
      <h2>8. Recent Events / Alerts</h2>
      {!result && events.length === 0 && <p className="state">No events yet. Select a frame and press Run or Play.</p>}
      {result && alerts.map((a, i) => (
        <div className={`alert ${a.severity}`} key={`a-${i}`} role={a.severity === 'high' ? 'alert' : 'status'}>
          <b>{a.text}</b>
          <p>{a.detail}</p>
          <span className="tag">frame {result.frame_id.slice(0, 8)}… · backend-derived</span>
        </div>
      ))}
      {events.length > 0 && (
        <>
          <h3 className="sub">Session events</h3>
          <ul className="events">
            {events.slice(-8).reverse().map((e, i) => (
              <li key={i} className={e.kind}><span className="mono">{e.time}</span> {e.text}</li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
