import { useCallback, useEffect, useState } from 'react';
import { ApiError, api } from '../api/client';
import type { FrameInfo } from '../types/api';

export function useFrames(enabled: boolean) {
  const [frames, setFrames] = useState<FrameInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getFrames();
      setFrames(res.frames);
    } catch (e) {
      setFrames([]);
      setError(e instanceof ApiError ? e.message : 'Cannot load replay frames.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (enabled) void load();
  }, [enabled, load]);

  return { frames, loading, error, reload: load };
}
