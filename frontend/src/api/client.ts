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
  EnvironmentInfo,
  ErrorResponse,
  FrameList,
  LiveSnapshot,
  LiveStatus,
  ModelInfo,
  PipelineResult,
  ReplayLoadInfo,
  ResourceMetrics,
  RLDecision,
  SegmentationMetrics,
  SimStatus,
  SimStep,
  TrackingResult,
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
  runFrame(frame_id: string, max_map_cells = 2000, semantic_mode: 'annotation' | 'model' = 'annotation'): Promise<{ result: PipelineResult; cache_hit: boolean }> {
    return request('/replay/run', {
      method: 'POST',
      body: JSON.stringify({ frame_id, max_map_cells, semantic_mode }),
    });
  },
  runSequence(frame_ids: string[], semantic_mode: 'annotation' | 'model' = 'annotation'): Promise<{ results: PipelineResult[]; errors: unknown[]; ran: number; requested: number }> {
    return request('/replay/run-sequence', {
      method: 'POST',
      body: JSON.stringify({ frame_ids, semantic_mode }),
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
  /** Live-detected hardware/software context (never invented; "Unavailable" where undetectable). */
  getEnvironment(): Promise<EnvironmentInfo> {
    return request<EnvironmentInfo>('/system/environment');
  },
  /** Trained-model card with measured held-out metrics. */
  getModelInfo(): Promise<ModelInfo> {
    return request<ModelInfo>('/model/info');
  },
  /** Stored held-out segmentation evaluation (404 when not generated). */
  getSegmentationMetrics(): Promise<SegmentationMetrics> {
    return request<SegmentationMetrics>('/segmentation/metrics');
  },
  /** Stored fair resource comparison (404 when not generated). */
  getResourceMetrics(): Promise<ResourceMetrics> {
    return request<ResourceMetrics>('/resource/metrics');
  },
  /** 9-stage methodology flow status with per-stage evidence. */
  getFlowStatus(): Promise<{ stages: { box: number; name: string; status: string; detail: string; evidence: string }[]; system_outputs: string[]; perception_taxonomy: Record<string, string> }> {
    return request('/flow/status');
  },
  /** RL state + untrained-DQN decision + safety override for a stored frame. */
  rlDecide(frame_id: string): Promise<RLDecision> {
    return request(`/rl/decide/${encodeURIComponent(frame_id)}`);
  },
  /** RL/DQN descriptor + training status (blocked without simulator). */
  rlStatus(): Promise<{ dqn: Record<string, unknown>; training: string }> {
    return request('/rl/status');
  },
  /** Offline closed-loop simulation controls (replay frames, measured steps). */
  simulationStatus(): Promise<SimStatus> {
    return request('/simulation/status');
  },
  simulationStart(frame_ids?: string[]): Promise<SimStatus> {
    return request('/simulation/start', {
      method: 'POST',
      body: JSON.stringify(frame_ids ? { frame_ids } : {}),
    });
  },
  simulationStep(): Promise<SimStep> {
    return request('/simulation/step', { method: 'POST' });
  },
  simulationStop(): Promise<SimStatus> {
    return request('/simulation/stop', { method: 'POST' });
  },
  simulationReset(): Promise<SimStatus> {
    return request('/simulation/reset', { method: 'POST' });
  },
  /** Multi-frame tracking over replay frames or a scene sequence. */
  runTracking(body: { frame_ids?: string[]; scene_id?: string; scene_samples?: number; semantic_mode?: 'annotation' | 'model' }): Promise<TrackingResult> {
    return request<TrackingResult>('/tracking/run', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },
  getTrackingScenes(): Promise<{ scenes: { scene_id: string; scene_token: string; nbr_samples: number; description: string }[]; count: number }> {
    return request('/tracking/scenes');
  },
  /** Live-stream ingestion (no dataset attached). */
  streamSimulate(frame_id: string, semantic_mode: 'annotation' | 'model'): Promise<{ result: PipelineResult }> {
    return request('/stream/simulate', {
      method: 'POST',
      body: JSON.stringify({ frame_id, semantic_mode }),
    });
  },
  streamPush(points: number[][], semantic_mode: 'annotation' | 'model'): Promise<{ result: PipelineResult }> {
    return request('/stream/push', {
      method: 'POST',
      body: JSON.stringify({ points, semantic_mode }),
    });
  },
  /** Stored benchmark asset (generated from results/benchmark_v2/, never recalculated). */
  async getBenchmarkResults(): Promise<BenchmarkAsset> {
    const res = await fetch(`${import.meta.env.BASE_URL}benchmark.json`);
    if (!res.ok) throw new Error('Benchmark results are not available.');
    return (await res.json()) as BenchmarkAsset;
  },
  /** Latest recorded closed-loop decision (404-shape error when empty). */
  decisionCurrent(): Promise<{ frame_id: string; decision_model: string; proposed_action: string | null; confidence: number | null; safety_status: string; executed_action: string; decision_latency_ms: number | null }> {
    return request('/decision/current');
  },
  /** Live PyBullet/Jev backend state (measured, never assumed). */
  pybulletStatus(): Promise<{ simulator: string; decision_backend: string }> {
    return request('/simulation/pybullet');
  },
  /** Start the live PyBullet loop (autonomous = Jev, manual = human steps). */
  liveStart(mode: 'autonomous' | 'manual', decision_interval_steps = 5): Promise<LiveStatus> {
    return request('/simulation/live/start', {
      method: 'POST',
      body: JSON.stringify({ mode, decision_interval_steps }),
    });
  },
  liveStop(): Promise<LiveStatus> {
    return request('/simulation/live/stop', { method: 'POST' });
  },
  liveReset(): Promise<LiveStatus> {
    return request('/simulation/live/reset', { method: 'POST' });
  },
  /** Current live snapshot (404 when the live loop never stepped). */
  liveState(): Promise<LiveSnapshot> {
    return request('/simulation/state');
  },
  /** One safety-checked manual step (recorded source=manual, never Jev). */
  manualAction(action: 'forward' | 'left' | 'right' | 'stop'): Promise<LiveSnapshot> {
    return request(`/simulation/action/${action}`, { method: 'POST' });
  },
  /** Live system roll-up (simulator, Jev, live loop, physical=NOT EXECUTED). */
  systemStatus(): Promise<{ backend: string; simulator: string; pybullet_importable: boolean; decision_model: string; jev_available: boolean; jev_reason: string | null; live: LiveStatus; physical_testing: string }> {
    return request('/system/status');
  },
};
