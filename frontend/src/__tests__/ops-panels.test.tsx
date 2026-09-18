import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AlertsPanel } from '../components/AlertsPanel';
import { ObjectsTerrainPanel } from '../components/ObjectsTerrainPanel';
import { ResolutionDistancePanel } from '../components/ResolutionDistancePanel';
import type { PipelineResult } from '../types/api';
import { sampleResult } from './fixture';

const R = sampleResult as unknown as PipelineResult;

describe('ops-console panels (backend values only)', () => {
  it('ObjectsTerrainPanel lists annotation objects with distance, never confidence', () => {
    const { container } = render(<ObjectsTerrainPanel result={R} />);
    expect(screen.getByText('5. Detected Objects & Terrain')).toBeTruthy();
    expect(container.textContent).toMatch(/Distance/);
    expect(container.textContent).not.toMatch(/0\.93|confidence.*%/i);
  });
  it('ObjectsTerrainPanel shows empty state without fabricating', () => {
    render(<ObjectsTerrainPanel result={null} />);
    expect(screen.getByText(/No result available/)).toBeTruthy();
  });
  it('AlertsPanel derives alerts from the result', () => {
    render(<AlertsPanel result={R} events={[]} />);
    expect(screen.getByText('8. Recent Events / Alerts')).toBeTruthy();
  });
  it('AlertsPanel shows session events honestly', () => {
    render(<AlertsPanel result={null} events={[{ time: '12:00:00', kind: 'info', text: 'Frame selected: abc…' }]} />);
    expect(screen.getByText(/Frame selected/)).toBeTruthy();
  });
  it('ResolutionDistancePanel bands measured resolution by distance', () => {
    const { container } = render(<ResolutionDistancePanel result={R} />);
    expect(screen.getByText(/Resolution vs Distance/)).toBeTruthy();
    expect(container.textContent).toMatch(/0–20 m/);
    expect(container.textContent).toMatch(/allocated by importance/);
  });
});
