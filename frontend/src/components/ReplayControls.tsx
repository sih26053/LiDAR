import { fmtTimestamp, shortId } from '../utils/formatting';
import type { FrameInfo } from '../types/api';
import type { ReplayPhase } from '../hooks/useReplay';

interface Props {
  frames: FrameInfo[];
  framesLoading: boolean;
  framesError: string | null;
  current: FrameInfo | null;
  phase: ReplayPhase;
  actionError: string | null;
  playing: boolean;
  speed: number;
  onSelect: (f: FrameInfo | null) => void;
  onRun: () => void;
  onReset: () => void;
  onPrev: () => void;
  onNext: () => void;
  onPlayPause: () => void;
  onSpeed: (s: number) => void;
}

const PHASE_LABEL: Record<ReplayPhase, string> = {
  idle: 'READY',
  loading: 'LOADING',
  processing: 'PROCESSING',
  ready: 'READY',
  error: 'ERROR',
};

export function ReplayControls(p: Props) {
  return (
    <section className="panel" aria-label="Replay controls">
      <h2>Replay Controls</h2>
      {p.framesLoading && <p className="state">Loading frames…</p>}
      {p.framesError && <p className="error" role="alert">{p.framesError}</p>}
      {!p.framesLoading && !p.framesError && p.frames.length === 0 && (
        <p className="state">No replay frames available.</p>
      )}
      <label className="field">
        Frame
        <select
          value={p.current?.frame_id ?? ''}
          onChange={(e) => {
            const f = p.frames.find((x) => x.frame_id === e.target.value) ?? null;
            p.onSelect(f);
          }}
          disabled={p.frames.length === 0}
        >
          <option value="">— select a frame —</option>
          {p.frames.map((f) => (
            <option key={f.frame_id} value={f.frame_id}>
              {shortId(f.frame_id)} · {f.scene_id ?? '?'}
            </option>
          ))}
        </select>
      </label>
      <div className="btn-row">
        <button onClick={p.onPlayPause} disabled={!p.current || p.frames.length === 0} title="Auto-advance frames through the backend pipeline">
          {p.playing ? '⏸ Pause' : '▶ Play'}
        </button>
        <button onClick={p.onPrev} disabled={!p.current || p.phase === 'processing' || p.phase === 'loading'}>◀ Prev</button>
        <button onClick={p.onNext} disabled={!p.current || p.phase === 'processing' || p.phase === 'loading'}>Next ▶</button>
        <button className="primary" onClick={p.onRun} disabled={!p.current || p.phase === 'processing' || p.phase === 'loading'}>
          {p.phase === 'processing' || p.phase === 'loading' ? 'Running…' : 'Run'}
        </button>
        <button onClick={p.onReset}>Reset</button>
        <label className="field inline">
          Speed
          <select value={p.speed} onChange={(e) => p.onSpeed(Number(e.target.value))} aria-label="replay speed">
            {[0.25, 0.5, 1, 2].map((s) => (
              <option key={s} value={s}>{s}x</option>
            ))}
          </select>
        </label>
      </div>
      <p className="caption">
        Replay: {p.playing ? 'LIVE — auto-advancing through backend pipeline' : 'STOPPED — single frame'} ·
        speed sets display cadence only; measured pipeline latency is unaffected.
      </p>
      <dl className="meta">
        <div><dt>Frame ID</dt><dd>{p.current ? shortId(p.current.frame_id) : '—'}</dd></div>
        <div><dt>Scene ID</dt><dd>{p.current?.scene_id ?? '—'}</dd></div>
        <div><dt>Timestamp</dt><dd className="mono">{fmtTimestamp(p.current?.timestamp)}</dd></div>
        <div><dt>Status</dt><dd><span className={`badge ${p.phase}`}>{PHASE_LABEL[p.phase]}</span></dd></div>
      </dl>
      {p.phase === 'loading' && <p className="state">Loading frame…</p>}
      {p.phase === 'processing' && <p className="state">Processing LiDAR… Building adaptive map…</p>}
      {p.actionError && <p className="error" role="alert">{p.actionError}</p>}
      {!p.current && <p className="state">No frame selected.</p>}
    </section>
  );
}
