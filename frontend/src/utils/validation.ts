/** Runtime guards for backend payloads (invalid-response state). */
import type { PipelineResult } from '../types/api';

export function isValidResult(v: unknown): v is PipelineResult {
  if (typeof v !== 'object' || v === null) return false;
  const r = v as Record<string, unknown>;
  return (
    typeof r.frame_id === 'string' &&
    Array.isArray(r.map_cells) &&
    typeof r.importance === 'object' &&
    typeof r.resolution === 'object' &&
    typeof r.timing === 'object'
  );
}

export const KNOWN_SEMANTIC_SOURCES = new Set([
  'model_prediction',
  'lidarseg_annotation',
  'lidarseg',
  'annotation',
  'object_annotation',
  'fallback',
  'unknown',
  'model',
]);
