import { API_BASE_URL } from '../api/client';
import type { BackendState } from '../hooks/useBackendStatus';
import type { ConfigInfo, DemoStatusInfo } from '../types/api';

/**
 * Panel 9 — Demo & Data Status. Only truthful, verified prototype facts:
 * public dataset, local replay, frozen config. No security claims beyond
 * what is actually implemented (loopback-only local operation).
 */
export function DemoDataPanel({ backend, demo, config }: {
  backend: BackendState; demo: DemoStatusInfo | null; config: ConfigInfo | null;
}) {
  const rows: [string, string][] = [
    ['Dataset', 'nuScenes Mini (public dataset, local files)'],
    ['Mode', 'Local replay — no live sensor'],
    ['Backend', backend === 'connected' ? `Local (${API_BASE_URL})` : backend === 'checking' ? 'Checking…' : 'Disconnected'],
    ['Model', config ? `Frozen configuration (${config.selection})` : demo?.configuration_loaded ? 'Frozen configuration (loaded)' : 'Unavailable'],
    ['Semantic mode', config?.semantic_source_mode ?? 'annotation+fallback (no trained model)'],
    ['Replay frames', demo ? `${demo.available_frames} validated frames` : 'Unavailable'],
    ['Data classification', 'Public demonstration data'],
    ['Network', 'Offline / local loopback only'],
  ];
  return (
    <section className="panel" id="panel-demo" aria-label="Demo and data status">
      <h2>9. Demo &amp; Data Status</h2>
      <ul className="kv">
        {rows.map(([k, v]) => (
          <li key={k}><span>{k}</span><b>{v}</b></li>
        ))}
      </ul>
      <p className="caption">No external server, upload, or public deployment is used for this demo.</p>
    </section>
  );
}
