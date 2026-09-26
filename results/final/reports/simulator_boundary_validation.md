# Simulator Boundary Validation (2026-09-25)

Mandate: representation fixes WITHOUT modifying Stages 1-6.
Statuses unchanged (7 BLOCKED, 8 PARTIAL, 9 PARTIAL).

## Phase 1 — Boundary data inspection (measured)
LiDAR: 3 rings x 180 rays; origin at vehicle center +1.1 m (z=1.8).
Per-180-ray census on default traffic (seed 42):
- -6° ring: 118 ground hits @16.8-17.4 m + 62 wall hits @7.8-15.5 m
- 0° ring: 2 obstacle hits @29.1-29.2 m (TRUE range) + 178 sky
  (0° flies over the 1 m walls from 1.8 m origin)
- +6° ring: 180 sky — dropped, never invented (182 pts reach pipeline)
- empty space: no return -> no cell -> sectors saturate to 1.0 (clear)
- obstacle returns: in-lane boxes at true range; ego vehicle never
  self-hits (origin above body, verified: no <7 m ego returns)

Replay parity (3 real nuScenes frames, model channel):
- density = 1.000 on ALL replay frames too — evidence-presence
  occupancy (mapper_2_5d.py: occupancy=1.0 wherever LiDAR evidence
  falls, documented prototype semantics) is SHARED, not a sim defect
- replay sectors 0.2-3.8 m (dense urban); sim sectors 8-12.5 m —
  sim states are MORE OPEN than the validated replay distribution

## Phase 2 — Ground-return verdict: NO filtering change
Ground IS mapped as occupied (structurally, same as replay). Filtering
it at the boundary was REJECTED on evidence: (a) sector minima are
wall-driven (7.8 m), not ground-driven (17 m) — filtering would not
change clearances; (b) it would DIVERGE sim input from replay input
the frozen stages were validated on; (c) it would be sensor-tuning
toward motion. Documented instead + regression tests:
- test_lidar_ring_attribution_ground_wall_obstacle (PASS live)
- test_explicit_empty_scenario_gives_open_road (PASS live)
- test_state_geometry_correspondence (PASS live)

## Phase 3 — Corridor geometry + repairs
- Road 120x12 m; wall inner faces y=+/-6.25 m: REAL lateral
  constraints, correctly read at ~7.8 m. Forward path genuinely open.
- Repair 1 (code, boundary): env.reset() treated explicit
  scenario={'obstacles': []} as falsy -> silently loaded default
  traffic. Now None=default traffic, []=open road (1-line change,
  src/simulation/pybullet_env.py).
- Repair 2 (config): scenario obstacles were authored near origin
  while the vehicle starts at x=-50 (60 m away, invisible). All six
  translated to reachable world coords; side blockers at ~4.5 m
  ahead so they read INSIDE the 7.8 m wall floor (a 10 m side
  obstacle is occluded by nearer wall returns — proven by probe:
  left-sector min 4.32 m vs 7.80 m).
- Files: config/pybullet_scenarios.json (coords + geometry doc),
  config/simulation_config.json (stale-install comment corrected).

## Phase 4 — Controlled states (NO Jev): 12/12 PASS
results/pybullet/controlled_states.json (+ scripts/probe_controlled_states.py):
- straight_road: nearest_fwd 12.53 m (open) | obstacle_ahead: 8.51 m
- left_blocked: left 4.6 m < right 8.0 m | right_blocked mirrored
- both_blocked: 4.6/5.4 m | emergency_close: 2.54 m + emergency flag
Every 13-D vector valid; named states + criteria recorded per scenario.
