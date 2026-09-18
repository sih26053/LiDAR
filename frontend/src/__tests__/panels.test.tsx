import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { MetricsPanel } from '../components/MetricsPanel';
import { StatusPanel } from '../components/StatusPanel';
import { ReplayControls } from '../components/ReplayControls';
import { ExplanationPanel } from '../components/ExplanationPanel';
import { sampleResult } from './fixture';

vi.mock('../api/client', () => ({
  api: { getBenchmarkResults: async () => ({ provenance: { source: 'results/benchmark/' }, methods: [], per_frame: [] }) },
  API_BASE_URL: 'http://127.0.0.1:8000',
  ApiError: class extends Error {},
}));

describe('MetricsPanel', () => {
  it('renders measured backend values with units', () => {
    render(<MetricsPanel result={sampleResult} metrics={null} />);
    expect(screen.getAllByText('34,359').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('109.8 ms')).toBeTruthy();
    expect(screen.getByText('Mapping latency')).toBeTruthy();
    expect(screen.getByText('API wall clock')).toBeTruthy();
  });
  it('shows empty state without fabricating values', () => {
    render(<MetricsPanel result={null} metrics={null} />);
    expect(screen.getByText(/No result available/)).toBeTruthy();
  });
});

describe('StatusPanel semantic source', () => {
  it('shows the exact backend source, never "AI prediction"', () => {
    const { container } = render(
      <StatusPanel frame={null} result={sampleResult} backend="connected" demo={null} />,
    );
    expect(container.textContent).toMatch(/fallback/);
    expect(container.textContent).not.toMatch(/AI prediction/);
  });
  it('declares unavailable overlay instead of hiding it', () => {
    const noSem = { ...sampleResult, map_cells: sampleResult.map_cells.filter((c: { semantic_source: string }) => c.semantic_source === 'fallback') };
    render(<StatusPanel frame={null} result={noSem} backend="connected" demo={null} />);
    expect(screen.getByText(/Semantic overlay unavailable/)).toBeTruthy();
  });
});

describe('ReplayControls states', () => {
  const base = {
    frames: [{ frame_id: 'a', scene_id: 's', timestamp: 1, source: 'x', point_count: 10 }],
    framesLoading: false, framesError: null, current: null, phase: 'idle' as const,
    actionError: null, playing: false, speed: 1,
    onSelect: () => {}, onRun: () => {}, onReset: () => {},
    onPrev: () => {}, onNext: () => {}, onPlayPause: () => {}, onSpeed: () => {},
  };
  it('shows empty state with no frame selected', () => {
    render(<ReplayControls {...base} />);
    expect(screen.getByText(/No frame selected/)).toBeTruthy();
  });
  it('shows processing state during run', () => {
    render(<ReplayControls {...base} phase="processing" current={base.frames[0]} />);
    expect(screen.getByText(/Processing LiDAR/)).toBeTruthy();
  });
  it('shows backend errors explicitly', () => {
    render(<ReplayControls {...base} framesError="Backend unavailable. Start the local FastAPI service and retry." />);
    expect(screen.getByText(/Backend unavailable/)).toBeTruthy();
  });
});

describe('ExplanationPanel', () => {
  it('renders the decision summary for the current frame', () => {
    render(<ExplanationPanel result={sampleResult} />);
    expect(screen.getByText(/Fine regions/)).toBeTruthy();
    expect(screen.getByText(/Highest importance observed/)).toBeTruthy();
  });
});
