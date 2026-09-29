import { useEffect, useRef, useState } from 'react';
import { API_BASE_URL, api } from '../api/client';
import type { LiveSnapshot } from '../types/api';

export type LiveConnection = 'ws' | 'polling' | 'idle' | 'error';

/**
 * Live PyBullet+Laya snapshot hook.
 * Prefers /ws/live; falls back to polling GET /simulation/state every 2 s.
 * Never fabricates values: snapshot is null until the backend produces one.
 */
export function useLiveSimulation(enabled: boolean) {
  const [snapshot, setSnapshot] = useState<LiveSnapshot | null>(null);
  const [connection, setConnection] = useState<LiveConnection>('idle');
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!enabled) {
      wsRef.current?.close();
      wsRef.current = null;
      setConnection('idle');
      return;
    }
    let cancelled = false;
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    const startPolling = () => {
      if (cancelled || pollTimer) return;
      setConnection('polling');
      const poll = async () => {
        try {
          const s = await api.liveState();
          if (!cancelled) {
            setSnapshot(s);
            setError(null);
          }
        } catch (e) {
          if (!cancelled) setError(e instanceof Error ? e.message : 'Live state unavailable.');
        }
      };
      void poll();
      pollTimer = setInterval(() => void poll(), 2000);
    };

    try {
      const wsUrl = `${API_BASE_URL.replace(/^http/, 'ws')}/ws/live`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.onopen = () => { if (!cancelled) { setConnection('ws'); setError(null); } };
      ws.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const msg = JSON.parse(ev.data as string) as LiveSnapshot;
          if (msg.frame_id) setSnapshot(msg);
        } catch { /* ignore malformed frames */ }
      };
      ws.onerror = () => { if (!cancelled) startPolling(); };
      ws.onclose = () => { if (!cancelled) startPolling(); };
    } catch {
      startPolling();
    }

    return () => {
      cancelled = true;
      if (pollTimer) clearInterval(pollTimer);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [enabled]);

  return { snapshot, connection, error };
}

/** Derive the honest engine status pill from a live snapshot. */
export function jevStatusOf(s: LiveSnapshot | null): string {
  if (!s || !s.frame_id) return 'NO DATA';
  if (s.source === 'manual') return 'MANUAL (not Laya)';
  if (s.jev_status === 'OK' && (s.source === 'laya' || s.source === 'jev')) return 'LIVE';
  if (s.jev_status === 'OK') return 'FALLBACK (low confidence)';
  if (s.jev_status === 'UNAVAILABLE') return 'UNAVAILABLE (safe fallback)';
  if (s.jev_status === 'FAILED' || s.jev_status === 'INVALID') return 'ERROR (safe fallback)';
  return 'NO DATA';
}

/** Alias with the active-engine name (identical behavior). */
export const layaStatusOf = jevStatusOf;
