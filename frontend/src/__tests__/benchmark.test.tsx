import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { BenchmarkPanel } from '../components/BenchmarkPanel';
import { sampleResult } from './fixture';

const asset = {
  provenance: {
    source: 'results/benchmark/ (stored validated results, NOT recalculated)',
    files: ['results/benchmark/benchmark_summary.csv'],
    note: 'not recalculated',
  },
  task: 'test',
  methods: [
    {
      method: 'proposed', label: 'Proposed Adaptive', frames: 10, successful: 10,
      mean_mapping_latency_ms: 104.8, median_mapping_latency_ms: 112.5,
      mean_end_to_end_latency_ms: 1184.2, mean_cells: 873.9, median_cells: 1028,
      mean_resolution_m: 0.125, mean_coverage: 18620.0, failure_rate: 0,
      resolution_totals: {}, resolution_share: {},
    },
    {
      method: 'uniform_5cm', label: 'Uniform 5 cm', frames: 10, successful: 10,
      mean_mapping_latency_ms: 0.2, median_mapping_latency_ms: 0.04,
      mean_end_to_end_latency_ms: 1079.5, mean_cells: 7453396.6, median_cells: 7492925,
      mean_resolution_m: 0.05, mean_coverage: 18620.0, failure_rate: 0,
      resolution_totals: {}, resolution_share: {},
    },
  ],
  per_frame: [],
};

vi.mock('../api/client', () => ({
  api: { getBenchmarkResults: async () => asset },
  API_BASE_URL: 'http://127.0.0.1:8000',
  ApiError: class extends Error {},
}));

describe('BenchmarkPanel', () => {
  it('separates stored benchmarks from the current replay and shows the fairness note', async () => {
    render(<BenchmarkPanel current={sampleResult} />);
    await waitFor(() => expect(screen.getAllByText('Proposed Adaptive').length).toBeGreaterThanOrEqual(1));
    expect(screen.getAllByText('Stored benchmark result', { exact: false }).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Current replay result', { exact: false })).toBeTruthy();
    expect(screen.getByText(/not[\s\S]*recalculated by the frontend/)).toBeTruthy();
  });
  it('handles missing benchmark data without fake values', async () => {
    const { api } = await import('../api/client');
    vi.spyOn(api, 'getBenchmarkResults').mockRejectedValueOnce(new Error('missing'));
    render(<BenchmarkPanel current={null} />);
    await waitFor(() => expect(screen.getByText(/not available/)).toBeTruthy());
  });
});
