import { useEffect, useMemo, useRef, useState } from 'react';
import { AdaptiveMapView } from './components/AdaptiveMapView';
import { AlertsPanel } from './components/AlertsPanel';
import { BenchmarkPanel } from './components/BenchmarkPanel';
import { DemoDataPanel } from './components/DemoDataPanel';
import { ExplanationPanel } from './components/ExplanationPanel';
import { Header } from './components/Header';
import { ImportanceView } from './components/ImportanceView';
import { Legend } from './components/Legend';
import { LidarView } from './components/LidarView';
import { MetricsPanel } from './components/MetricsPanel';
import { ObjectsTerrainPanel } from './components/ObjectsTerrainPanel';
import { ReplayControls } from './components/ReplayControls';
import { ResolutionDistancePanel } from './components/ResolutionDistancePanel';
import { ResolutionView } from './components/ResolutionView';
import { SettingsPanel } from './components/SettingsPanel';
import { StatusPanel } from './components/StatusPanel';
import { useBackendStatus } from './hooks/useBackendStatus';
import { useFrames } from './hooks/useFrames';
import { useReplay } from './hooks/useReplay';
import './styles/app.css';

const RAIL: { id: string; label: string }[] = [
  { id: 'panel-live', label: 'Live View' },
  { id: 'panel-map', label: 'Map' },
  { id: 'panel-objects', label: 'Objects' },
  { id: 'panel-alerts', label: 'Alerts' },
  { id: 'panel-system', label: 'System' },
  { id: 'panel-settings', label: 'Settings' },
];

/**
 * Ops-console judge layout. Single source of truth: useReplay() — every
 * panel renders the same frame; all numbers come from the backend result.
 */
export default function App() {
  const backend = useBackendStatus();
  const connected = backend.state === 'connected';
  const frames = useFrames(connected);
  const replay = useReplay();
  const [encoding, setEncoding] = useState<'elevation' | 'semantic' | 'importance'>('elevation');
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const idx = useMemo(
    () => (replay.currentFrame ? frames.frames.findIndex((f) => f.frame_id === replay.currentFrame!.frame_id) : -1),
    [frames.frames, replay.currentFrame],
  );

  const step = (dir: 1 | -1) => {
    if (idx < 0 || frames.frames.length === 0) return;
    const next = frames.frames[(idx + dir + frames.frames.length) % frames.frames.length];
    replay.selectFrame(next);
  };

  const selectAndPrime = (frameId: string | null) => {
    const f = frames.frames.find((x) => x.frame_id === frameId) ?? null;
    replay.selectFrame(f);
  };

  // Live replay loop: each tick runs the NEXT frame through the backend
  // pipeline (load → run → metrics). Ticks never overlap a running pipeline;
  // speed only sets display cadence, never measured latency.
  const tickRef = useRef<() => void>(() => {});
  tickRef.current = () => {
    if (!playing) return;
    if (replay.phase === 'loading' || replay.phase === 'processing') return;
    if (frames.frames.length === 0) return;
    const cur = idx >= 0 ? idx : -1;
    const next = frames.frames[(cur + 1) % frames.frames.length];
    replay.selectFrame(next);
    void replay.run(next);
  };
  useEffect(() => {
    if (!playing) return;
    const ms = Math.round(8000 / speed);
    const t = setInterval(() => tickRef.current(), ms);
    return () => clearInterval(t);
  }, [playing, speed]);

  const stopAndReset = () => {
    setPlaying(false);
    replay.reset();
  };

  return (
    <div className="app ops">
      <Header backend={backend.state} demo={backend.demo} frame={replay.currentFrame} playing={playing} />
      {backend.state === 'checking' && <p className="banner">Checking backend health…</p>}
      {backend.state === 'unavailable' && (
        <div className="banner error" role="alert">
          Backend unavailable. Start the local FastAPI service
          (<code>uvicorn backend.app:app --host 127.0.0.1 --port 8000</code>) and retry.
          {backend.error && <> Detail: {backend.error}</>}
          <button onClick={backend.retry}>Retry</button>
        </div>
      )}
      <div className="ops-body">
        <nav className="rail" aria-label="console sections">
          {RAIL.map((r) => (
            <a key={r.id} href={`#${r.id}`}>{r.label}</a>
          ))}
        </nav>
        <main className="grid ops-grid">
          <div id="panel-live" className="ops-anchor" />
          <ReplayControls
            frames={frames.frames}
            framesLoading={frames.loading}
            framesError={frames.error}
            current={replay.currentFrame}
            phase={replay.phase}
            actionError={replay.error}
            playing={playing}
            speed={speed}
            onSelect={(f) => (f ? selectAndPrime(f.frame_id) : replay.selectFrame(null))}
            onRun={() => void replay.run(replay.currentFrame)}
            onReset={stopAndReset}
            onPrev={() => step(-1)}
            onNext={() => step(1)}
            onPlayPause={() => setPlaying((p) => !p)}
            onSpeed={setSpeed}
          />
          <LidarView result={replay.currentResult} mode={encoding} onMode={setEncoding} title="1. Raw LiDAR Point Cloud" />
          <LidarView result={replay.currentResult} mode="semantic" onMode={() => undefined} title="2. Semantic Perception" hideModeSwitch />
          <ImportanceView result={replay.currentResult} />
          <div id="panel-map" className="ops-anchor" />
          <AdaptiveMapView result={replay.currentResult} />
          <ResolutionView result={replay.currentResult} />
          <ObjectsTerrainPanel result={replay.currentResult} />
          <MetricsPanel result={replay.currentResult} metrics={replay.currentMetrics} />
          <ResolutionDistancePanel result={replay.currentResult} />
          <div id="panel-system" className="ops-anchor" />
          <StatusPanel frame={replay.currentFrame} result={replay.currentResult} backend={backend.state} demo={backend.demo} />
          <AlertsPanel result={replay.currentResult} events={replay.events} />
          <DemoDataPanel backend={backend.state} demo={backend.demo} config={backend.config} />
          <SettingsPanel speed={speed} onSpeed={setSpeed} maxCells={replay.maxCells} onMaxCells={replay.setMaxCells} />
          <Legend />
          <BenchmarkPanel current={replay.currentResult} />
          <ExplanationPanel result={replay.currentResult} />
        </main>
      </div>
      <footer className="footer">
        Paradox Protocol · LOCAL DEMO · PUBLIC DATASET · NUScenes MINI · OFFLINE/LOCAL ·
        semantic annotations are evaluation references, never model predictions
      </footer>
    </div>
  );
}
