import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { useReplay } from '../hooks/useReplay';
import { sampleResult } from './fixture';

vi.mock('../api/client', () => ({
  api: {
    loadFrame: async (frame_id: string) => ({ frame_id, point_count: 100 }),
    runFrame: async (frame_id: string) => ({
      result: { ...(sampleResult as object), frame_id },
      cache_hit: false,
    }),
    getMetrics: async () => null,
  },
  ApiError: class extends Error {},
}));

describe('useReplay session feed + settings', () => {
  it('records select/load/map events from real hook actions', async () => {
    const { result } = renderHook(() => useReplay());
    const frame = { frame_id: 'abcdef123456', scene_id: 'scene-1', timestamp: 1, source: 'x', point_count: 100 };
    act(() => { result.current.selectFrame(frame); });
    expect(result.current.events.some((e) => e.text.includes('Frame selected'))).toBe(true);
    await act(async () => { await result.current.run(frame); });
    expect(result.current.phase).toBe('ready');
    expect(result.current.events.some((e) => e.text.includes('LiDAR frame loaded'))).toBe(true);
    expect(result.current.events.some((e) => e.text.includes('Map generated'))).toBe(true);
  });

  it('reset clears result and events with a reset notice', async () => {
    const { result } = renderHook(() => useReplay());
    const frame = { frame_id: 'abcdef123456', scene_id: 'scene-1', timestamp: 1, source: 'x', point_count: 100 };
    await act(async () => { await result.current.run(frame); });
    act(() => { result.current.reset(); });
    expect(result.current.currentResult).toBeNull();
    expect(result.current.events).toHaveLength(1);
    expect(result.current.events[0].text).toMatch(/Session reset/);
  });

  it('exposes maxCells display setting (default 2000)', () => {
    const { result } = renderHook(() => useReplay());
    expect(result.current.maxCells).toBe(2000);
    act(() => { result.current.setMaxCells(500); });
    expect(result.current.maxCells).toBe(500);
  });
});
