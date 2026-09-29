import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { DecisionPanel } from '../components/DecisionPanel';
import { FlowPanel } from '../components/FlowPanel';
import { RLPanel } from '../components/RLPanel';
import { SimulationPanel } from '../components/SimulationPanel';
import { sampleResult } from './fixture';

vi.mock('../api/client', () => {
  let simActive = false;
  const simStatus = () => ({
    active: simActive, episode_id: simActive ? 'e1' : null, semantic_mode: 'model',
    frames_total: 7, frames_done: simActive ? 1 : 0, stops: 0, safety_overrides: 0,
    mean_decide_latency_ms: null,
    carla: { installed: false, connected: false, lidar_simulated: false, closed_loop: false },
  });
  return {
  api: {
    getFlowStatus: async () => ({
      stages: [
        { box: 1, name: 'Data Acquisition', status: 'LIVE', detail: 'replay', evidence: '/frames' },
        { box: 7, name: 'RL Decision Making (DQN)', status: 'IMPLEMENTED (untrained)', detail: 'dqn', evidence: 'src/rl_agent.py' },
        { box: 9, name: 'Real-World Testing', status: 'BLOCKED (offline replay evaluation instead)', detail: 'blocked', evidence: 'metrics' },
      ],
      system_outputs: ['High-detail mapping where needed'],
      perception_taxonomy: { road_driveable: 'Road' },
    }),
    rlDecide: async () => ({
      frame_id: 'abc', semantic_mode: 'model',
      state: {
        state_dim: 13, state_vector: Array(13).fill(0.5), sector_ranges_m: Array(8).fill(30),
        cells_in_radius: 10, cells_total: 20, obstacle_density: 0.1, moving_share: 0,
        static_share: 0, terrain_share: 0.9,
        safety: { emergency_stop: false, nearest_forward_obstacle_m: null, rule: 'r' },
      },
      decision: {
        actions: ['forward', 'turn_left', 'turn_right', 'stop'], q_values: [0.1, 0, 0, 0],
        network_action: 'forward', final_action: 'forward', safety_override: false,
        trained: false, model: 'DQN', warning: 'Untrained.',
      },
      decide_latency_ms: 3.5,
    }),
    simulationStatus: async () => simStatus(),
    simulationStart: async () => { simActive = true; return simStatus(); },
    simulationStep: async () => ({
      frame_id: 'abc', stages_executed: ['perception+map', 'safety+execute'],
      map_cells: 100, q_values: [0.1, 0, 0, 0], network_action: 'forward',
      final_action: 'forward', safety_verdict: 'SAFE_TO_EXECUTE', safety_override: false,
      rewards: { safe_clearance: 0.5 }, latency_ms: { dqn: 0.1 }, decide_latency_ms: 5,
      dqn_trained: false,
    }),
    simulationStop: async () => ({
      active: false, episode_id: 'e1', semantic_mode: 'model', frames_total: 7,
      frames_done: 1, stops: 0, safety_overrides: 0, mean_decide_latency_ms: 5,
    }),
    simulationReset: async () => ({
      active: true, episode_id: 'e2', semantic_mode: 'model', frames_total: 7,
      frames_done: 0, stops: 0, safety_overrides: 0, mean_decide_latency_ms: null,
    }),
    decisionCurrent: async () => ({
      frame_id: 'abc', decision_model: 'jev', proposed_action: 'forward',
      confidence: 0.82, safety_status: 'SAFE_TO_EXECUTE', executed_action: 'forward',
      decision_latency_ms: 410.2,
    }),
    pybulletStatus: async () => ({
      simulator: 'pybullet (BLOCKED: not importable here)',
      decision_backend: 'jev (BLOCKED) / dqn-untrained baseline',
    }),
  },
  API_BASE_URL: 'http://127.0.0.1:8000',
  ApiError: class extends Error {},
  };
});

describe('Methodology flow + RL panels', () => {
  it('FlowPanel renders stages with honest blocked/untrained badges', async () => {
    render(<FlowPanel />);
    await waitFor(() => expect(screen.getByText(/1\. Data Acquisition/)).toBeTruthy());
    expect(screen.getByText('LIVE')).toBeTruthy();
    expect(screen.getByText('IMPLEMENTED (untrained)')).toBeTruthy();
    expect(screen.getByText(/BLOCKED \(offline replay evaluation instead\)/)).toBeTruthy();
  });
  it('RLPanel asks for a frame first and warns the net is untrained', async () => {
    const { unmount } = render(<RLPanel result={null} />);
    expect(screen.getByText(/Run a frame first/)).toBeTruthy();
    unmount();
    render(<RLPanel result={sampleResult as never} />);
    screen.getByText(/Decide from/).click();
    await waitFor(() => expect(screen.getByText(/Untrained network/)).toBeTruthy());
  });
  it('SimulationPanel reports CARLA blocked and steps honestly', async () => {
    render(<SimulationPanel />);
    await waitFor(() => expect(screen.getByText(/NOT INSTALLED/)).toBeTruthy());
    screen.getByText('Start').click();
    await waitFor(() => expect(screen.getByText('RUNNING')).toBeTruthy());
    screen.getByText('Step').click();
    await waitFor(() => expect(screen.getByText(/SAFE_TO_EXECUTE/)).toBeTruthy());
  });
  it('DecisionPanel shows recorded action, confidence and blocked backends', async () => {
    render(<DecisionPanel />);
    await waitFor(() => expect(screen.getAllByText('forward').length).toBeGreaterThanOrEqual(1));
    expect(screen.getByText('0.820')).toBeTruthy();
    expect(screen.getByText(/BLOCKED: not importable/)).toBeTruthy();
  });
});
