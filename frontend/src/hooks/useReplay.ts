/**
 * Single source of truth for the active pipeline result.
 * All panels consume { currentFrame, currentResult, currentMetrics, ... }
 * so metadata, map, metrics can never go stale relative to each other.
 * Also keeps a chronological session event feed (real frontend/backend
 * events only) and the display-only max-cells request setting.
 */
import { useCallback, useState } from 'react';
import { ApiError, api } from '../api/client';
import type { DemoMetrics, FrameInfo, PipelineResult } from '../types/api';
import { isValidResult } from '../utils/validation';

export type ReplayPhase = 'idle' | 'loading' | 'processing' | 'ready' | 'error';

export interface SessionEvent { time: string; kind: 'info' | 'success' | 'error'; text: string; }

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export function useReplay() {
  const [currentFrame, setCurrentFrame] = useState<FrameInfo | null>(null);
  const [currentResult, setCurrentResult] = useState<PipelineResult | null>(null);
  const [currentMetrics, setCurrentMetrics] = useState<DemoMetrics | null>(null);
  const [phase, setPhase] = useState<ReplayPhase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [maxCells, setMaxCells] = useState(2000);

  const push = useCallback((kind: SessionEvent['kind'], text: string) => {
    setEvents((prev) => [...prev.slice(-49), { time: stamp(), kind, text }]);
  }, []);

  const selectFrame = useCallback((frame: FrameInfo | null) => {
    // Selecting a new frame invalidates the previous result immediately.
    setCurrentFrame(frame);
    setCurrentResult(null);
    setCurrentMetrics(null);
    setError(null);
    setPhase('idle');
    if (frame) push('info', `Frame selected: ${frame.frame_id.slice(0, 8)}… (${frame.scene_id ?? 'unknown scene'})`);
  }, [push]);

  const run = useCallback(
    async (frame: FrameInfo | null, cellsOverride?: number) => {
      if (!frame) {
        setError('No frame selected.');
        return;
      }
      const limit = cellsOverride ?? maxCells;
      setPhase('loading');
      setError(null);
      push('info', `Loading frame ${frame.frame_id.slice(0, 8)}…`);
      try {
        const loaded = await api.loadFrame(frame.frame_id);
        setCurrentFrame({ ...frame, point_count: loaded.point_count ?? frame.point_count });
        push('info', `LiDAR frame loaded: ${(loaded.point_count ?? 0).toLocaleString()} points`);
        setPhase('processing');
        const { result } = await api.runFrame(frame.frame_id, limit);
        if (!isValidResult(result)) throw new Error('Invalid response: pipeline result failed validation.');
        setCurrentResult(result);
        push('success', `Map generated: ${result.map_cell_count.toLocaleString()} cells in ${result.timing.total_latency_ms != null ? result.timing.total_latency_ms.toFixed(0) : 'unavailable'} ms (backend-measured)`);
        try {
          setCurrentMetrics(await api.getMetrics(frame.frame_id));
        } catch {
          setCurrentMetrics(null);
        }
        setPhase('ready');
      } catch (e) {
        setPhase('error');
        const msg = e instanceof ApiError || e instanceof Error ? e.message : 'Pipeline failure.';
        setError(msg);
        push('error', `Pipeline failure on ${frame.frame_id.slice(0, 8)}…: ${msg}`);
      }
    },
    [push, maxCells],
  );

  const reset = useCallback(() => {
    setCurrentResult(null);
    setCurrentMetrics(null);
    setError(null);
    setPhase('idle');
    setEvents([{ time: stamp(), kind: 'info', text: 'Session reset: result, metrics, visualizations and events cleared.' }]);
  }, []);

  const semanticSource = currentResult
    ? dominantSource(currentResult.semantic.source_counts)
    : null;

  return { currentFrame, currentResult, currentMetrics, phase, error, semanticSource, events, maxCells, setMaxCells, selectFrame, run, reset };
}

function dominantSource(counts: Record<string, number> | undefined): string {
  if (!counts) return 'unknown';
  let best = 'unknown';
  let bestN = -1;
  for (const [k, v] of Object.entries(counts)) {
    if (v > bestN) { best = k; bestN = v; }
  }
  return best;
}
