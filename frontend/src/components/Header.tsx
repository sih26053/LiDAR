import { useEffect, useState } from 'react';
import type { BackendState } from '../hooks/useBackendStatus';
import type { DemoStatusInfo, FrameInfo } from '../types/api';
import { shortId } from '../utils/formatting';

function useClock(): string {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(now.getDate())}/${p(now.getMonth() + 1)}/${now.getFullYear()} ${p(now.getHours())}:${p(now.getMinutes())}:${p(now.getSeconds())}`;
}

export function Header({ backend, demo, frame, playing }: { backend: BackendState; demo: DemoStatusInfo | null; frame: FrameInfo | null; playing: boolean }) {
  const connected = backend === 'connected';
  const modelLoaded = demo?.configuration_loaded === true;
  const operational = connected && modelLoaded;
  const clock = useClock();
  return (
    <header className="header ops">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">P</div>
      <div>
        <b>PRAGYA</b>
        <span>Perception & Resolution-Adaptive Guidance for Intelligent Autonomous systems</span>
      </div>
    </div>
      <div className="ops-title">
        <h1>Adaptive LiDAR Perception &amp; Mapping System</h1>
        <p className="subtitle">Importance-Driven 2.5D LiDAR Mapping for Dynamic Environment Replay (trained point classifier + tracker on board)</p>
      </div>
      <div className="sys-state" role="status" aria-label="system state">
        <span className={`pill ${operational ? 'ok' : backend === 'checking' ? 'wait' : 'bad'}`}>
          <span className={`dot ${operational ? 'ok' : backend === 'checking' ? 'wait' : 'bad'}`} />
          {operational ? 'SYSTEM OPERATIONAL' : backend === 'checking' ? 'CHECKING…' : 'SYSTEM OFFLINE'}
        </span>
        <span className="ops-meta">{clock}</span>
        <span className="ops-meta">
          Replay: {playing ? 'LIVE' : 'STOPPED'} · Backend: {connected ? 'CONNECTED' : backend === 'checking' ? 'CHECKING' : 'DISCONNECTED'} · Dataset: nuScenes Mini
        </span>
        <span className="ops-meta">
          Scene: {frame?.scene_id ?? '—'} · Frame: {frame ? shortId(frame.frame_id) : '—'} · Model: {connected ? (modelLoaded ? 'Frozen config loaded' : 'Unavailable') : 'Unavailable'}
        </span>
      </div>
    </header>
  );
}
