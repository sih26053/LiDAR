import { useEffect, useState } from 'react';
import { api } from '../api/client';

interface RunInfo {
  run_id: string;
  mode: string;
  scenario: string | null;
  status: string;
  started_at: string;
  frame_count: number;
}

interface RunFrame {
  frame_id: string;
  sequence: number;
  timestamp: number | null;
  point_count: number;
}

interface ReplayResult {
  frame_id: string;
  source: string;
  action: string | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  latency_ms: number | null;
  safety: { status: string | null; override: boolean; reason: string | null };
  stored: boolean;
  error?: string | null;
}

/**
 * Recorded live frames browser (Phases 8/15E).
 * Lists SQLite-recorded runs/frames (origin=live-recorded, never mixed
 * silently with nuScenes replay), loads a frame into the existing replay
 * view, and runs Replay Laya (source=replay-laya) with an original-vs-replay
 * comparison. Descriptive only — neither result is labelled better.
 */
export function RecordedFramesPanel({ onReplayFrame }: { onReplayFrame: (frameId: string) => void }) {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [runId, setRunId] = useState<string>('');
  const [frames, setFrames] = useState<RunFrame[]>([]);
  const [idx, setIdx] = useState(0);
  const [original, setOriginal] = useState<Record<string, unknown> | null>(null);
  const [replayed, setReplayed] = useState<ReplayResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [storeInfo, setStoreInfo] = useState<string | null>(null);

  const refresh = async () => {
    setBusy(true);
    setNotice(null);
    try {
      const [r, st, sys] = await Promise.all([
        api.recordedRuns(),
        api.recordingsStatus(),
        api.systemStatus().catch(() => null),
      ]);
      setRuns(r.runs);
      if (r.runs.length > 0 && !r.runs.some((x) => x.run_id === runId)) {
        setRunId(r.runs[0].run_id);
      }
      const writes = sys?.live?.recorder_writes;
      const recErr = sys?.live?.recorder_error;
      setStoreInfo(
        `DB ${st.db.ok ? 'ok' : 'ERROR'} (${st.db.latest_run ?? 'no runs'}) · ` +
        `recorded runs on disk: ${st.recorded_runs} · ` +
        `live writes this session: ${writes ?? '—'}${recErr ? ` · RECORDER ERROR: ${recErr}` : ''}`,
      );
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Refresh failed.');
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    api.recordedRuns()
      .then((r) => {
        setRuns(r.runs);
        if (r.runs.length > 0) setRunId(r.runs[0].run_id);
      })
      .catch(() => setNotice('Recorded runs unavailable (backend down or DB missing).'));
    api.recordingsStatus()
      .then((st) => setStoreInfo(`DB ${st.db.ok ? 'ok' : 'ERROR'} · recorded runs on disk: ${st.recorded_runs}`))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!runId) return;
    setOriginal(null);
    setReplayed(null);
    api.recordedRunFrames(runId)
      .then((r) => {
        setFrames(r.frames);
        setIdx(0);
      })
      .catch(() => setNotice(`Frames unavailable for run ${runId}.`));
  }, [runId]);

  const cur = frames[idx] ?? null;

  const loadFrame = async (frameId: string) => {
    setBusy(true);
    setNotice(null);
    try {
      const rec = await api.recordedFrame(frameId);
      setOriginal(rec as unknown as Record<string, unknown>);
      onReplayFrame(frameId);
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Frame load failed.');
    } finally {
      setBusy(false);
    }
  };

  const replayLaya = async (all: boolean) => {
    if (!cur && !all) return;
    setBusy(true);
    setNotice(null);
    try {
      if (all) {
        const ids = frames.map((f) => f.frame_id);
        let ok = 0;
        for (const id of ids) {
          const r = await api.replayLayaDecide(id);
          if (r.stored) ok += 1;
          if (id === cur?.frame_id) setReplayed(r);
        }
        setNotice(`Replay Laya stored ${ok}/${ids.length} (source=replay-laya).`);
      } else if (cur) {
        const r = await api.replayLayaDecide(cur.frame_id);
        setReplayed(r);
        if (!original) {
          const rec = await api.recordedFrame(cur.frame_id);
          setOriginal(rec as unknown as Record<string, unknown>);
        }
      }
    } catch (e) {
      setNotice(e instanceof Error ? e.message : 'Replay Laya failed.');
    } finally {
      setBusy(false);
    }
  };

  const fmt = (v: unknown): string =>
    v === null || v === undefined ? '—' : typeof v === 'number' ? (Number.isFinite(v) ? v.toFixed(3) : '—') : String(v);

  const live = original?.live_decision as Record<string, unknown> | null | undefined;

  return (
    <section className="panel wide" aria-label="Recorded live frames">
      <h2>Recorded Live Frames (origin=live-recorded)</h2>
      {storeInfo && <p className="state">{storeInfo}</p>}
      {runs.length === 0 && <p className="state">No recorded runs yet — run the live loop to record frames.</p>}
      {runs.length > 0 && (
        <>
          <div className="btn-row">
            <label>Run:
              <select value={runId} onChange={(e) => setRunId(e.target.value)} disabled={busy}>
                {runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} · {r.scenario ?? 'default'} · {r.frame_count} frames · {r.status}
                  </option>
                ))}
              </select>
            </label>
            <button onClick={() => setIdx((i) => Math.max(0, i - 1))} disabled={busy || idx <= 0}>Previous</button>
            <button onClick={() => setIdx((i) => Math.min(frames.length - 1, i + 1))} disabled={busy || idx >= frames.length - 1}>Next</button>
            <button className="primary" onClick={() => cur && void loadFrame(cur.frame_id)} disabled={busy || !cur}>Load in replay view</button>
            <button onClick={() => void replayLaya(false)} disabled={busy || !cur}>Replay Laya</button>
            <button onClick={() => void replayLaya(true)} disabled={busy || frames.length === 0}>Replay All</button>
            <button onClick={() => void refresh()} disabled={busy}>Refresh</button>
            <span className="pill">{cur ? `${idx + 1} / ${frames.length}` : '—'}</span>
          </div>
          {notice && <p className="state">{notice}</p>}
          {cur && (
            <ul className="kv">
              <li><span>Frame</span><b className="mono">{cur.frame_id} (seq {cur.sequence})</b></li>
              <li><span>Points</span><b>{cur.point_count}</b></li>
            </ul>
          )}
          <div className="btn-row">
            <div style={{ flex: 1 }}>
              <h3 className="sub">Original live</h3>
              {live ? (
                <ul className="kv">
                  <li><span>Action</span><b>{fmt(live.proposed_action)}</b></li>
                  <li><span>Confidence</span><b>{fmt(live.confidence)}</b></li>
                  <li><span>Latency</span><b>{fmt(live.latency_ms)} ms</b></li>
                  <li><span>Source</span><b>{fmt(live.source)}</b></li>
                </ul>
              ) : <p className="state">Load a frame to see its recorded live decision.</p>}
            </div>
            <div style={{ flex: 1 }}>
              <h3 className="sub">Replay Laya (source=replay-laya)</h3>
              {replayed ? (
                <ul className="kv">
                  <li><span>Action</span><b>{fmt(replayed.action)}</b></li>
                  <li><span>Confidence</span><b>{fmt(replayed.confidence)}</b></li>
                  <li><span>Latency</span><b>{fmt(replayed.latency_ms)} ms</b></li>
                  <li><span>Safety</span><b>{replayed.safety.status}{replayed.safety.override ? ' (OVERRIDE)' : ''}</b></li>
                  <li><span>Stored</span><b>{replayed.stored ? 'yes' : 'no'}</b></li>
                </ul>
              ) : <p className="state">Press Replay Laya (runs Stages 1–6 + Laya + safety, no vehicle motion).</p>}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
