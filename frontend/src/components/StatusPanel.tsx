import { fmtTimestamp, shortId } from '../utils/formatting';
import type { BackendState } from '../hooks/useBackendStatus';
import type { DemoStatusInfo, FrameInfo, PipelineResult } from '../types/api';

const SOURCE_BLURB: Record<string, string> = {
  model_prediction: 'Model prediction — network output.',
  lidarseg_annotation: 'LiDARSeg annotation — evaluation reference, never a model prediction.',
  lidarseg: 'LiDARSeg annotation — evaluation reference, never a model prediction.',
  annotation: 'Annotation — evaluation reference, never a model prediction.',
  object_annotation: 'Object annotation — evaluation reference, never a model prediction.',
  fallback: 'Fallback — heuristic label where no annotation applied.',
  unknown: 'Unknown — no semantic information for this frame/source.',
  model: 'Model output.',
};

/** Panel F — Frame / semantic source / status. */
export function StatusPanel({
  frame, result, backend, demo,
}: {
  frame: FrameInfo | null;
  result: PipelineResult | null;
  backend: BackendState;
  demo: DemoStatusInfo | null;
}) {
  const counts = result?.semantic.source_counts ?? {};
  const dominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 'unknown';
  const classes = result ? [...new Set(result.map_cells.slice(0, 2000).map((c) => c.semantic_class))] : [];
  const validSemantic = result ? result.map_cells.filter((c) => c.semantic_source !== 'fallback' && c.semantic_source !== 'unknown').length : 0;
  return (
    <section className="panel" aria-label="Frame and semantic status">
      <h2>7. System Status</h2>
      <dl className="meta">
        <div><dt>Frame ID</dt><dd className="mono">{frame ? shortId(frame.frame_id) : '—'}</dd></div>
        <div><dt>Scene ID</dt><dd>{frame?.scene_id ?? result?.scene_id ?? '—'}</dd></div>
        <div><dt>Timestamp</dt><dd className="mono">{fmtTimestamp(frame?.timestamp ?? result?.timestamp)}</dd></div>
        <div><dt>Backend</dt><dd>{backend === 'connected' ? '● Connected' : backend === 'checking' ? '… Checking' : '● Unavailable'}</dd></div>
        <div><dt>Pipeline</dt><dd>{result?.status ?? '—'}</dd></div>
        <div><dt>Config</dt><dd>{demo?.configuration_loaded ? 'Loaded' : 'Unavailable'}</dd></div>
      </dl>
      <div className="source-box">
        <b>Semantic Source: <span className={`src src-${dominant}`}>{dominant}</span></b>
        <p>{SOURCE_BLURB[dominant] ?? SOURCE_BLURB.unknown}</p>
        {result && (
          <p className="caption">
            mode: {result.semantic.mode} · per-source counts: {Object.entries(counts).map(([k, v]) => `${k}=${v}`).join(', ') || 'none'} ·{' '}
            {validSemantic} cells with valid overlay · classes observed: {classes.slice(0, 8).join(', ') || 'none'}
          </p>
        )}
        {result && validSemantic === 0 && (
          <p className="state">Semantic overlay unavailable for this frame/source.</p>
        )}
        {!result && <p className="state">No result available for this frame.</p>}
      </div>
    </section>
  );
}
