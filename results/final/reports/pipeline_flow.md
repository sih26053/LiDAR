# Pipeline Flow — Paradox Protocol

```
Raw LiDAR (Nx4: x, y, z, intensity, replayed from nuScenes Mini)
   ↓  finite-mask + ROI filter → counts (n_raw / removed / n_processed)
Preprocessing (src/preprocessing.py)
   ↓  annotation lookup + geometric fallback
Semantic / Geometric Features (src/semantic_adapter.py, src/feature_adapter.py)
   ↓  weighted combination (distance 0.30 / semantic 0.30 / terrain 0.15 /
Importance (src/importance_engine.py)  dynamic 0.15 / uncertainty 0.10)
   ↓  thresholds fine ≥ 0.70 / medium ≥ 0.45 / coarse ≥ 0.20
Adaptive Resolution (src/resolution_engine.py)  → 0.05 / 0.10 / 0.20 / 0.50 m
   ↓  2 m integration cells with elevation + occupancy
2.5D Map (src/mapper_2_5d.py)
   ↓  cells + summaries + per-stage timing + fps
Metrics + Visualization (backend PipelineResult → React panels)
```

Each stage records its own latency (`timing.*_latency_ms`); the sum is
`total_latency_ms` (compute only). Failures raise `PipelineStageError`
(stage + code) and surface as clean judge-facing errors; details stay in
server logs. Failsafe policy: `results/final/config/failsafe_config.json`.
