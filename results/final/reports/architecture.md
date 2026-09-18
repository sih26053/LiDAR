# Architecture — Paradox Protocol (frozen prototype-final-v1)

```
LiDAR Replay (nuScenes Mini, local .npy + metadata)
     ↓
FastAPI Backend (backend/app.py — wiring only, local loopback 127.0.0.1:8000)
     ↓
Replay Service (backend/services/replay_service.py — frame index + Nx4 load)
     ↓
Preprocessing (src/preprocessing.py — finite-mask + ROI filter)
     ↓
Perception / Annotation (annotation lookup via nuScenes devkit + fallback)
     ↓
Feature Extraction (src/feature_adapter.py — region features)
     ↓
Importance Engine (src/importance_engine.py — frozen weights W_BASE)
     ↓
Resolution Engine (src/resolution_engine.py — frozen thresholds THRESH_C)
     ↓
Adaptive 2.5D Mapper (src/mapper_2_5d.py — 2 m integration cells)
     ↓
Pipeline Result (structured dict: cells + importance + resolution + semantic + timing)
     ↓
React Dashboard (frontend/src — visualization only, never recomputes ML)
```

Frozen configuration: `results/final/config/final_config.json`
(selection `W_BASE+THRESH_C`, seed 42). No trained weight files exist (CASE B —
heuristic engines with frozen weights; see `results/final/code_state.json`).
Engines are rebuilt from the frozen config on backend start and parity-asserted
(`backend/services/pipeline_service.py::_engines`).

Related: `pipeline_flow.md` (data flow), `input_output_contract.md` (schemas),
`metric_definitions.md` (metric semantics).
