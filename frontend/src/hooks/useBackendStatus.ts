import { useCallback, useEffect, useState } from 'react';
import { ApiError, api } from '../api/client';
import type { ConfigInfo, DemoStatusInfo } from '../types/api';

export type BackendState = 'checking' | 'connected' | 'unavailable';

export function useBackendStatus() {
  const [state, setState] = useState<BackendState>('checking');
  const [config, setConfig] = useState<ConfigInfo | null>(null);
  const [demo, setDemo] = useState<DemoStatusInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const check = useCallback(async () => {
    setState('checking');
    setError(null);
    try {
      await api.checkHealth();
      const [cfg, demoStatus] = await Promise.all([api.getConfig(), api.getDemoStatus()]);
      setConfig(cfg);
      setDemo(demoStatus);
      setState('connected');
    } catch (e) {
      setState('unavailable');
      setError(e instanceof ApiError ? e.message : 'Backend unavailable. Start the local FastAPI service and retry.');
    }
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  return { state, config, demo, error, retry: check };
}
