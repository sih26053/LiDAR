import { describe, expect, it } from 'vitest';
import type { MapCell, PipelineResult } from '../types/api';
import { cellDistance, deriveAlerts, groupObjects, resolutionByDistance, terrainSplit } from '../utils/terrain';
import { sampleResult } from './fixture';

function cell(patch: Partial<MapCell>): MapCell {
  return {
    x: 0, y: 0, elevation: 0, occupancy: 1, resolution: 0.1, importance: 0.5,
    semantic_class: 'unknown', semantic_source: 'fallback', confidence: null,
    point_count: 1, region_id: 0, ...patch,
  };
}

describe('terrain utils (display derivations only)', () => {
  it('computes Euclidean distance from sensor origin', () => {
    expect(cellDistance(3, 4)).toBe(5);
  });

  it('groups annotation cells by class with centroid and distance', () => {
    const groups = groupObjects([
      cell({ x: 10, y: 0, semantic_class: 'vehicle', semantic_source: 'annotation', importance: 0.8 }),
      cell({ x: 20, y: 0, semantic_class: 'vehicle', semantic_source: 'annotation', importance: 0.6 }),
      cell({ x: 0, y: 5, semantic_class: 'unknown', semantic_source: 'fallback' }),
    ]);
    expect(groups).toHaveLength(1);
    expect(groups[0].semantic_class).toBe('vehicle');
    expect(groups[0].cells).toBe(2);
    expect(groups[0].centroidX).toBe(15);
    expect(groups[0].distanceM).toBe(15);
    expect(groups[0].maxImportance).toBe(0.8);
  });

  it('splits terrain honestly: fallback counts as unclassified', () => {
    const s = terrainSplit([
      cell({ semantic_class: 'road_driveable', semantic_source: 'annotation' }),
      cell({ semantic_class: 'static_manmade', semantic_source: 'annotation' }),
      cell({ semantic_class: 'vehicle', semantic_source: 'annotation' }),
      cell({ semantic_class: 'unknown', semantic_source: 'fallback' }),
    ]);
    expect(s).toMatchObject({ drivable: 1, nonDrivable: 1, dynamic: 1, unclassified: 1, total: 4 });
  });

  it('bands backend resolution by distance without changing it', () => {
    const bands = resolutionByDistance([
      cell({ x: 10, y: 0, resolution: 0.05 }),
      cell({ x: 100, y: 0, resolution: 0.2 }),
    ]);
    expect(bands[0]).toMatchObject({ label: '0–20 m', cells: 1, meanResolution: 0.05 });
    expect(bands[3]).toMatchObject({ label: '60 m+', cells: 1, meanResolution: 0.2 });
  });

  it('derives alerts from the real sample result without inventing confidences', () => {
    const alerts = deriveAlerts(sampleResult as PipelineResult);
    expect(alerts.length).toBeGreaterThanOrEqual(1);
    for (const a of alerts) expect(JSON.stringify(a)).not.toMatch(/0\.93|confidence.*0\.\d/);
  });

  it('reports ok-state when nothing triggers', () => {
    const r = {
      ...(sampleResult as PipelineResult),
      map_cells: [cell({ x: 10, y: 0, semantic_class: 'vehicle', semantic_source: 'annotation', importance: 0.5 })],
    };
    expect(deriveAlerts(r)[0].severity).toBe('ok');
  });

  it('flags limited annotation coverage as info, not an error', () => {
    const r = { ...(sampleResult as PipelineResult), map_cells: [cell({})] };
    expect(deriveAlerts(r)[0].severity).toBe('info');
  });
});
