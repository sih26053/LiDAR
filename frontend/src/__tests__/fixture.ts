import type { PipelineResult } from '../types/api';

/** Fixture shaped exactly like POST /replay/run -> { result } (real run, truncated). */
export const sampleResult: PipelineResult = {
  frame_id: '5991fad3280c4f84b331536c32001a04',
  scene_id: 'scene-0655',
  timestamp: 1535385092150099.0,
  status: 'success',
  input_point_count: 34359,
  processed_point_count: 34359,
  map_cell_count: 533,
  map_cells: [
    {
      x: -34.98, y: -64.92, elevation: -0.01, occupancy: 1.0,
      resolution: 0.2, importance: 0.38, semantic_class: 'unknown',
      semantic_source: 'fallback', confidence: 0.0, point_count: 1, region_id: 0,
    },
    {
      x: 5.2, y: 3.1, elevation: 0.4, occupancy: 1.0,
      resolution: 0.05, importance: 0.84, semantic_class: 'vehicle',
      semantic_source: 'object_annotation', confidence: 0.0, point_count: 42, region_id: 7,
    },
    {
      x: 6.1, y: 4.2, elevation: 0.2, occupancy: 1.0,
      resolution: 0.1, importance: 0.55, semantic_class: 'static_manmade',
      semantic_source: 'lidarseg_annotation', confidence: 0.0, point_count: 12, region_id: 8,
    },
  ],
  importance: { mean: 0.565, min: 0.361, max: 0.846, count: 533 },
  resolution: {
    fine_cells: 52, medium_cells: 429, coarse_cells: 52,
    res_20cm_cells: 52, res_50cm_cells: 0,
    average_resolution: 0.104, distribution: { '0.05': 52, '0.1': 429, '0.2': 52, '0.5': 0 },
  },
  semantic: {
    mode: 'annotation+fallback (no trained model)',
    source_counts: { fallback: 474, annotation: 59 },
    note: 'Annotation-derived labels are evaluation references, never model predictions.',
  },
  timing: {
    preprocessing_latency_ms: 15.3, perception_latency_ms: 571.3,
    feature_extraction_latency_ms: 1169.7, importance_latency_ms: 49.9,
    resolution_latency_ms: 0.8, mapping_latency_ms: 109.8,
    total_latency_ms: 1866.1, serialization_latency_ms: 151.1,
    wall_clock_ms: 8899.4, fps: 0.54,
  },
};
