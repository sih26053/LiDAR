/**
 * Central API client. ALL backend communication goes through here --
 * React components must never call fetch() directly.
 *
 * Endpoints (backend/README.md, implemented 17 September):
 *   GET  /health  GET /config  GET /frames
 *   POST /replay/load  POST /replay/run  POST /replay/run-sequence
 *   GET  /results/{frame_id}  GET /metrics/{frame_id}  GET /demo/status
 */
import type {
  BackendStatus,
  BenchmarkAsset,
  ConfigInfo,
  DemoMetrics,
  DemoStatusInfo,
  ErrorResponse,
  FrameList,
  PipelineResult,
  ReplayLoadInfo,
} from '../types/api';

export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ||
  'http://127.0.0.1:8000';

export class ApiError extends Error {
  status: number;
  payload: ErrorResponse | null;
  constructor(status: number, payload: ErrorResponse | null, fallback: string) {
    super(payload?.message || fallback);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

function isErrorResponse(body: unknown): body is ErrorResponse {
  return (
    typeof body === 'object' &&
    body !== null &&
    'error_code' in body &&
    'message' in body
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  } catch {
    throw new ApiError(0, null, 'Backend unavailable. Start the local FastAPI service and retry.');
  }
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    throw new ApiError(
      res.status,
      isErrorResponse(body) ? body : null,
      `Request failed (${res.status}).`,
    );
  }
  return body as T;
}

export const api = {
  checkHealth(): Promise<BackendStatus> {
    return request<BackendStatus>('/health');
  },
  getConfig(): Promise<ConfigInfo> {
    return request<ConfigInfo>('/config');
  },
  getFrames(): Promise<FrameList> {
    return request<FrameList>('/frames');
  },
  loadFrame(frame_id: string): Promise<ReplayLoadInfo> {
    return request<ReplayLoadInfo>('/replay/load', {
      method: 'POST',
      body: JSON.stringify({ frame_id }),
    });
  },
  runFrame(frame_id: string, max_map_cells = 2000): Promise<{ result: PipelineResult; cache_hit: boolean }> {
    return request('/replay/run', {
      method: 'POST',
      body: JSON.stringify({ frame_id, max_map_cells }),
    });
  },
  getResult(frame_id: string): Promise<{ result: PipelineResult }> {
    return request<{ result: PipelineResult }>(
      `/results/${encodeURIComponent(frame_id)}`,
    );
  },
  getMetrics(frame_id: string): Promise<DemoMetrics> {
    return request<DemoMetrics>(`/metrics/${encodeURIComponent(frame_id)}`);
  },
  getDemoStatus(): Promise<DemoStatusInfo> {
    return request<DemoStatusInfo>('/demo/status');
  },
  /** Stored benchmark asset (generated from results/benchmark/, never recalculated). */
  async getBenchmarkResults(): Promise<BenchmarkAsset> {
    const res = await fetch(`${import.meta.env.BASE_URL}benchmark.json`);
    if (!res.ok) throw new Error('Benchmark results are not available.');
    return (await res.json()) as BenchmarkAsset;
  },
};
