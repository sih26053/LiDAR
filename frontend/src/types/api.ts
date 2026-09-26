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

export interface ModelBlock {
  model_name: string | null;
  architecture: string | null;
  classes: string[] | null;
  trained: boolean;
  weights_present: boolean;
  active: boolean;
  inference_latency_ms: number | null;
  note: string;
}

export interface ModelEvalBlock {
  available: boolean;
  accuracy_vs_annotation_reference: number | null;
  n_reference_points: number | null;
  mean_model_confidence: number | null;
  eval_only_ms: number | null;
  reason?: string | null;
  note?: string | null;
}

export interface PipelineResult {
  frame_id: string;
  scene_id: string | null;
  timestamp: number | null;
  input_source?: string | null;
  status: string;
  input_point_count: number | null;
  processed_point_count: number | null;
  map_cells: MapCell[];
  map_cell_count: number;
  importance: ImportanceSummary;
  resolution: ResolutionSummary;
  semantic: SemanticSummary;
  timing: TimingInfo;
  model?: ModelBlock | null;
  model_eval?: ModelEvalBlock | null;
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
  load_latency_ms?: number | null;
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
  importance_weights?: Record<string, number>;
  uncertainty_lambda?: number;
  resolution_levels: [number, number][];
  resolution_levels_m: Record<string, number>;
  resolution_thresholds: Record<string, number>;
  max_mapping_distance_m: number;
  integration_cell_size_m: number;
  semantic_source_mode: string;
  trained_model_available?: boolean;
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
  std_mapping_latency_ms: number | null;
  mean_end_to_end_latency_ms: number | null;
  mean_cells: number | null;
  median_cells: number | null;
  mean_resolution_m: number | null;
  mean_coverage: number | null;
  failure_rate: number | null;
  resolution_totals: Record<string, number>;
  resolution_share: Record<string, number>;
}

export interface EnvironmentInfo {
  cpu_model: string;
  cpu_count: number | string;
  ram_gb: number | string;
  gpu: string;
  os: string;
  python_version: string;
  packages: Record<string, string>;
  note: string;
}

export interface BenchmarkAsset {
  provenance: { source: string; files: string[]; note: string };
  task: string;
  hardware_summary?: string | null;
  methodology?: {
    timer?: string;
    repetitions_measured?: number;
    warmup_runs_discarded?: number;
    excluded?: string[];
    end_to_end_definition?: string;
    resolution_levels?: [number, number][];
  } | null;
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

export interface ModelInfo {
  model_name: string;
  model_file: string;
  framework: string;
  input_features: string[];
  classes: string[];
  supervision: string;
  not_supervised: string[];
  train_frames: string[];
  test_frames: string[];
  n_train_points: number;
  n_test_points: number;
  metrics_held_out: {
    accuracy: number | null;
    f1_macro: number | null;
    f1_per_class: Record<string, number | null>;
    confusion_matrix_rows_true_cols_pred: number[][];
    confusion_labels: string[];
    mean_predicted_confidence: number | null;
    mean_iou?: number | null;
    n_test_points?: number | null;
    reference?: string;
  } | null;
  inference_stage: string;
}

export interface SegmentationMetrics {
  reference: string;
  eval_frames: string[];
  n_eval_points: number;
  overall_accuracy_vs_reference: number | null;
  mean_iou_vs_reference: number | null;
  mean_confidence: number | null;
  per_class: { class: string; support: number; predicted: number; precision: number | null; recall: number | null; f1: number | null; iou: number | null }[];
  distance_bins: { distance_bin_m: string; n_points: number; status: string; accuracy_vs_reference: number | null; mean_iou: number | null; mean_confidence: number | null }[];
  classes_without_support: string[];
}

export interface RLDecision {
  frame_id: string;
  semantic_mode: string | null;
  state: {
    state_dim: number;
    state_vector: number[];
    sector_ranges_m: number[];
    cells_in_radius: number;
    cells_total: number;
    obstacle_density: number;
    moving_share: number;
    static_share: number;
    terrain_share: number;
    safety: { emergency_stop: boolean; nearest_forward_obstacle_m: number | null; rule: string };
  };
  decision: {
    actions: string[];
    q_values: number[];
    network_action: string;
    final_action: string;
    safety_override: boolean;
    trained: boolean;
    model: string;
    warning: string;
  };
  decide_latency_ms: number;
}

export interface SimStatus {
  active: boolean;
  episode_id: string | null;
  semantic_mode: string | null;
  frames_total: number;
  frames_done: number;
  stops: number;
  safety_overrides: number;
  mean_decide_latency_ms: number | null;
  carla?: { installed: boolean; connected: boolean; lidar_simulated: boolean; closed_loop: boolean };
}

export interface SimStep {
  frame_id: string;
  stages_executed: string[];
  map_cells: number;
  q_values: number[];
  network_action: string;
  final_action: string;
  safety_verdict: string;
  safety_override: boolean;
  rewards: Record<string, number | string | null>;
  latency_ms: Record<string, number>;
  decide_latency_ms: number;
  dqn_trained: boolean;
  done?: boolean;
}

/** Live PyBullet+Jev snapshot from GET /simulation/state or /ws/live. */
export interface LiveSnapshot {
  frame_id: string | null;
  timestamp: number | null;
  run_id: string | null;
  mode: string;
  simulator: string;
  lidar_points: number | null;
  map_cells: number | null;
  rl_state: number[] | null;
  jev_action: string | null;
  jev_confidence: number | null;
  jev_probabilities: Record<string, number> | null;
  jev_status: string | null;
  safety_status: string | null;
  executed_action: string | null;
  source: string | null;
  vehicle_position: { x: number; y: number } | null;
  vehicle_heading: number | null;
  collision: boolean | null;
  jev_latency_ms: number | null;
  loop_latency_ms: number | null;
  simulation_time: number | null;
  system_status: string;
  pipeline?: Record<string, { state: string; verified: boolean }>;
  lidar?: { point_count: number | null; frame_count: number | null; fps: number | null; status: string };
  map?: { cell_count: number | null; importance: ImportanceSummary; semantic_classes: string[]; status: string };
  decision?: { model: string; action: string | null; confidence: number | null; probabilities: Record<string, number> | null; latency_ms: number | null; status: string };
  safety?: { status: string | null; override: boolean; reason: string | null };
  vehicle?: { x: number | null; y: number | null; z: number | null; yaw_deg: number | null; speed_mps: number | null; yaw_rate: number | null };
  execution?: { proposed_action: string | null; executed_action: string | null; source: string | null };
  metrics?: {
    jev_latency_ms: number | null; perception_latency_ms: number | null;
    safety_latency_ms: number | null; action_execution_latency_ms: number | null;
    loop_latency_ms: number | null; vehicle_speed_mps: number | null;
    distance_m: number | null; collision: boolean | null; collisions_total: number;
    jev_calls: number; jev_successful: number; jev_failed: number;
    manual_calls: number; directional_actions: number; stop_actions: number;
    safety_overrides: number;
  };
  pipeline_result?: PipelineResult | null;
}

export interface LiveStatus {
  active: boolean;
  mode: string;
  run_id: string | null;
  steps: number;
  simulator: string;
  jev_calls: number;
  jev_successful: number;
  jev_failed: number;
  manual_calls: number;
  safety_overrides: number;
  collisions: number;
  error: string | null;
}

export interface ResourceMetrics {
  provenance: string;
  methods: { method: string; frames: number; mean_cells: number; mean_mapping_latency_ms: number; mean_total_latency_ms: number; mean_fps: number; mean_memory_bytes: number; mean_resolution_m: number; mean_input_points: number }[];
  cell_reduction_proposed_vs_uniform_pct: number;
  memory_reduction_proposed_vs_uniform_pct: number;
  memory_note: string;
}

export interface TrackState {
  frame_id: string;
  x: number;
  y: number;
  vx_m_s: number | null;
  vy_m_s: number | null;
  speed_m_s: number | null;
  cells: number;
}

export interface Track {
  id: number;
  class: string;
  closed: boolean;
  misses: number;
  hits: number;
  age_frames: number;
  states: TrackState[];
}

export interface TrackingResult {
  tracks: Track[];
  n_tracks: number;
  n_associations: number;
  n_new_tracks: number;
  n_scene_breaks: number;
  note: string;
  frames: {
    frame_id: string;
    scene_id: string | null;
    timestamp: number | null;
    detections: { class: string; x: number; y: number; cells: number }[];
    map_cell_count: number;
    total_latency_ms: number | null;
  }[];
  frame_ids: string[];
  semantic_mode: string;
  parameters: Record<string, number>;
  errors: { frame_id: string; stage: string; error_code: string }[];
}
