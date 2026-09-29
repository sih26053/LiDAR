import { useEffect, useState } from 'react';
import type { JSX } from 'react';
import { api } from '../api/client';
import type { LayaCalibration, LayaDiagnostics, LayaServerStatus } from '../types/api';

/** Validation badges: VERIFIED only with artifact evidence, else honest status. */
function ValidationSection() {
  const [v, setV] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    let on = true;
    api.layaValidation()
      .then((r) => { if (on) setV(r as unknown as Record<string, unknown>); })
      .catch(() => undefined);
    return () => { on = false; };
  }, []);
  if (!v) return <p className="state">Loading validation status…</p>;
  const gate = (v.gate ?? {}) as Record<string, unknown>;
  const row = (label: string, value: unknown): JSX.Element => (
    <li key={label}><span>{label}</span><b>{String(value ?? '—')}</b></li>
  );
  return (
    <ul className="kv">
      {row('Calibration', v.calibration)}
      {row('Gate', `${String(gate.threshold ?? '—')} (${String(gate.status ?? '—')}; legacy ${String(gate.legacy ?? '—')})`)}
      {row('Test n', v.test_n)}
      {row('E-category', `${String(v.e_category)} (test n=${String(v.e_test_n)})`)}
      {row('Temperature', v.temperature)}
      {row('Forward behavior', v.forward_behavior)}
      {row('Fine-tuning', v.fine_tuning)}
      {row('OOD', `${String(v.ood)} (n=${String(v.ood_n)})`)}
      {row('Docker', `${String(v.docker)}${v.docker_reason ? ` — ${String(v.docker_reason)}` : ''}`)}
      {row('Offline', v.offline)}
      {row('Emergency executed FORWARD', v.emergency_executed_forward)}
      {row('Promotion', v.promotion)}
      {row('Physical hardware', v.physical_hardware)}
    </ul>
  );
}

/**
 * Laya Navigation Diagnostics + Confidence Calibration + Server panels.
 * Every value is measured (evaluation artifacts / live status endpoints);
 * missing artifacts render as NOT AVAILABLE, never as invented numbers.
 * No PASS badge is shown without an explicit test criterion.
 */
export function LayaDiagnosticsPanel() {
  const [diag, setDiag] = useState<LayaDiagnostics | null>(null);
  const [cal, setCal] = useState<LayaCalibration | null>(null);
  const [srv, setSrv] = useState<LayaServerStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let on = true;
    Promise.all([api.layaDiagnostics(), api.layaCalibration(), api.layaStatus()])
      .then(([d, c, s]) => { if (on) { setDiag(d); setCal(c); setSrv(s); } })
      .catch((e) => { if (on) setError(e instanceof Error ? e.message : 'Unavailable'); });
    return () => { on = false; };
  }, []);

  const pct = (v: number | null | undefined): string =>
    typeof v === 'number' && Number.isFinite(v) ? `${(v * 100).toFixed(1)} %` : '—';
  const num = (v: unknown, digits = 3): string =>
    typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';

  const nav = (diag?.navigation ?? {}) as Record<string, unknown>;
  const byMode = (nav.by_mode ?? {}) as Record<string, Record<string, unknown>>;
  const un = byMode.unconstrained_laya ?? {};
  const co = byMode.constrained_laya ?? {};
  const evalSet = (nav.evaluation_set as string) ?? '—';
  const gateSel = (cal?.gate_selection ?? {}) as Record<string, unknown>;
  const temp = (cal?.temperature ?? {}) as Record<string, unknown>;

  const kv = (label: string, value: string): JSX.Element => (
    <li key={label}><span>{label}</span><b>{value}</b></li>
  );

  return (
    <section className="panel wide" aria-label="Laya navigation diagnostics">
      <h2>Laya Navigation Diagnostics (measured)</h2>
      {error && <p className="state">{error}</p>}
      {!diag && !error && <p className="state">Loading measured diagnostics…</p>}

      <h3 className="sub">Evaluation set</h3>
      <ul className="kv">
        {kv('Evaluation set', nav.evaluation_set ? String(nav.evaluation_set) : 'NOT AVAILABLE')}
        {kv('States', String((nav.n_states as number) ?? '—'))}
        {kv('Frames evaluated', String(un.n ?? co.n ?? '—'))}
        {kv('Raw FORWARD rate (unconstrained)', pct(un.raw_forward_rate as number))}
        {kv('Constrained FORWARD rate', pct((co as Record<string, unknown>).raw_forward_rate as number))}
        {kv('Unsafe raw FORWARD proposals', `${un.unsafe_proposals ?? '—'} (${pct(un.unsafe_proposal_rate as number)})`)}
        {kv('Unsafe executed FORWARD', `${(co as Record<string, unknown>).emergency_executed_forward ?? '—'} emergency-executed (criterion: must be 0)`)}
        {kv('Emergency raw FORWARD (unconstrained)', String(un.emergency_raw_forward ?? '—'))}
        {kv('LEFT / RIGHT / STOP (constrained executed)', `${JSON.stringify((co as Record<string, unknown>).executed_distribution ?? '—')}`)}
        {kv('Last evaluation', String((nav.generated_utc as string) ?? '—'))}
      </ul>
      <p className="caption">Criterion: emergency executed FORWARD must be 0 (deterministic eligibility + safety). Current: {String((co as Record<string, unknown>).emergency_executed_forward ?? 'unknown')}. No other PASS claim is made.</p>

      <h3 className="sub">Confidence Calibration (measured)</h3>
      <ul className="kv">
        {kv('Gate metric', String(cal?.gate_config.metric ?? '—'))}
        {kv('Current threshold', String(cal?.gate_config.threshold ?? '—'))}
        {kv('Threshold source', String(cal?.gate_config.source ?? '—'))}
        {kv('Calibration state', String(cal?.runtime.gate.state ?? 'NOT LOADED'))}
        {kv('Calibration set', String((gateSel.dataset_id as string) ?? (cal?.runtime.gate.dataset as string) ?? '—'))}
        {kv('Selected threshold (artifact)', String((gateSel.selected_threshold as number) ?? '—'))}
        {kv('ECE (calibration split)', num(((((gateSel.split_metrics as Record<string, unknown>) ?? {}).calibration as Record<string, unknown>) ?? {}).ece as number, 4))}
        {kv('Brier (calibration split)', num(((((gateSel.split_metrics as Record<string, unknown>) ?? {}).calibration as Record<string, unknown>) ?? {}).brier as number, 4))}
        {kv('Temperature', `${String(temp.temperature ?? '—')} (state: ${String((cal?.runtime.temperature.state as string) ?? 'NOT LOADED')})`)}
      </ul>
      <p className="caption">Threshold: selected under the documented validation objective; NOT claimed optimal. Coverage/acceptance details: gate_selection artifact.</p>

      <h3 className="sub">Laya Validation (evidence-gated badges)</h3>
      <ValidationSection />
      <h3 className="sub">Laya Server (independent service)</h3>
      <ul className="kv">
        {kv('Status', String(srv?.state ?? '—'))}
        {kv('Host', '127.0.0.1 (loopback-only)')}
        {kv('PID', String(srv?.pid ?? '—'))}
        {kv('Uptime', srv?.uptime_s != null ? `${Math.round(srv.uptime_s)} s (measured)` : '—')}
        {kv('Restarts', String(srv?.restart_count ?? '—'))}
        {kv('Checkpoint', String(srv?.checkpoint ?? '—'))}
        {kv('Revision', String(srv?.checkpoint_revision ?? '—'))}
        {kv('Device', String(srv?.device ?? '—'))}
        {kv('Offline', String(srv?.offline ?? '—'))}
        {kv('Managed by backend', srv?.managed_process ? 'YES (dev convenience)' : srv?.independent ? 'NO (independent process)' : '—')}
      </ul>
      <p className="caption">Evaluation set: {evalSet}. Per-category and A/B tables: /laya/diagnostics endpoint.</p>
    </section>
  );
}
