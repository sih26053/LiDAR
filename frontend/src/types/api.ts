/**
 * Frontend types mirroring the frozen backend contracts exactly.
 * Sources: backend/schemas/output.py, backend/schemas/errors.py,
 * backend/routes/*.py, backend/config.py (GET /config).
 * No invented fields: every optional field is `| null` capable because the
 * backend returns measured-or-null values.
 */

export type SemanticSource =
  | 'model_prediction'
  | 'lidarseg_annotation'
  | 'lidarseg'
  | 'annotation'
  | 'object_annotation'
  | 'fallback'
  | 'unknown'
  | 'model';

export interface MapCell {
  x: number;
  y: number;
  elevation: number;
  occupancy: number;
  resolution: number;
  importance: number;
  semantic_class: string;
  semantic_source: string;
  confidence: number | null;
  point_count: number | null;
  region_id: number | null;
}

export interface ImportanceSummary {
  mean: number | null;
  min: number | null;
  max: number | null;
  count: number;
}

export interface ResolutionSummary {
  fine_cells: number;
  medium_cells: number;
  coarse_cells: number;
  res_20cm_cells: number;
  res_50cm_cells: number;
  average_resolution: number | null;
  distribution: Record<string, number>;
}

export interface SemanticSummary {
  mode: string;
  source_counts: Record<string, number>;
  note: string;
}

export interface TimingInfo {
  preprocessing_latency_ms: number | null;
  perception_latency_ms: number | null;
  feature_extraction_latency_ms: number | null;
  importance_latency_ms: number | null;
  resolution_latency_ms: number | null;
  mapping_latency_ms: number | null;
  total_latency_ms: number | null;
  serialization_latency_ms: number | null;
  wall_clock_ms: number | null;
  fps: number | null;
}

export interface PipelineResult {
  frame_id: string;
  scene_id: string | null;
  timestamp: number | null;
  status: string;
  input_point_count: number | null;
  processed_point_count: number | null;
  map_cells: MapCell[];
  map_cell_count: number;
  importance: ImportanceSummary;
  resolution: ResolutionSummary;
  semantic: SemanticSummary;
  timing: TimingInfo;
}

export interface FrameInfo {
  frame_id: string;
  scene_id: string | null;
  timestamp: number | null;
  source: string | null;
  point_count: number | null;
}

export interface FrameList {
  frames: FrameInfo[];
  count: number;
}

export interface ReplayLoadInfo {
  frame_id: string;
  scene_id: string | null;
  timestamp: number | null;
  point_count: number | null;
  load_status: string;
}

export interface DemoMetrics {
  frame_id: string;
  input_point_count: number | null;
  processed_point_count: number | null;
  map_cell_count: number | null;
  fine_cells: number | null;
  medium_cells: number | null;
  coarse_cells: number | null;
  average_resolution: number | null;
  latency_ms: number | null;
  fps: number | null;
}

export interface BackendStatus {
  status: string;
  service: string;
}

export interface ConfigInfo {
  final_version: string;
  selection: string;
  resolution_levels: [number, number][];
  resolution_levels_m: Record<string, number>;
  resolution_thresholds: Record<string, number>;
  max_mapping_distance_m: number;
  integration_cell_size_m: number;
  semantic_source_mode: string;
  random_seed: number;
  config_path: string;
}

export interface DemoStatusInfo {
  backend: string;
  configuration_loaded: boolean;
  replay_available: boolean;
  available_frames: number;
  last_frame_id: string | null;
  last_status: string | null;
}

export interface ErrorResponse {
  stage: string;
  error_code: string;
  message: string;
  frame_id: string | null;
}

export interface BenchmarkMethod {
  method: string;
  label: string;
  frames: number;
  successful: number;
  mean_mapping_latency_ms: number | null;
  median_mapping_latency_ms: number | null;
  mean_end_to_end_latency_ms: number | null;
  mean_cells: number | null;
  median_cells: number | null;
  mean_resolution_m: number | null;
  mean_coverage: number | null;
  failure_rate: number | null;
  resolution_totals: Record<string, number>;
  resolution_share: Record<string, number>;
}

export interface BenchmarkAsset {
  provenance: { source: string; files: string[]; note: string };
  task: string;
  methods: BenchmarkMethod[];
  per_frame: {
    frame_id: string;
    method: string;
    map_cells: number;
    mapping_latency_ms: number | null;
    end_to_end_latency_ms: number | null;
    mean_resolution_m: number | null;
    mean_importance: number | null;
    status: string;
  }[];
}
