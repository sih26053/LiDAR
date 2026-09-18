# Final Application Architecture — Paradox Protocol dashboard

```
            REAL / REPLAYED LiDAR  (nuScenes Mini, local .npy)
                    │
                    ▼
            ┌─────────────────┐
            │ Simulation /    │  frame index + Nx4 load
            │ Replay Engine   │  (backend/services/replay_service.py)
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ FastAPI Backend │  127.0.0.1:8000, wiring only
            └────────┬────────┘  (backend/app.py + routes/)
                     │
                     ▼
        ┌─────────────────────────────┐
        │ Existing Frozen src/        │  W_BASE + THRESH_C, seed 42
        │ Preprocessing               │  finite-mask + ROI
        │ Perception                  │  annotation lookup + fallback
        │ Importance Engine           │  frozen weights
        │ Resolution Engine           │  frozen thresholds
        │ Adaptive 2.5D Mapper        │  2 m integration cells
        └──────────────┬──────────────┘
                       │
                       ▼
              Structured Results      PipelineResult + DemoMetrics
                       │              (cells, importance, resolution,
                       ▼               semantic+source, per-stage timing)
            ┌─────────────────┐
            │ React Dashboard │  visualization only, never recomputes ML
            └────────┬────────┘
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
       Raw LiDAR   Importance   Adaptive
       +semantic                2.5D Map
       (canvas 2D, pan/zoom)    + objects/terrain + metrics + alerts
```

## Live replay loop (Play)

```
Play tick (display cadence 8 s / speed)
  → select next frame (clears stale result)
  → POST /replay/load → POST /replay/run → GET /metrics
  → single useReplay state update → ALL panels re-render same frame
```

Speed (0.25x–2x) and max-cells settings are display/request parameters only;
frozen weights/thresholds are not exposed anywhere in the UI.

## Rendering decisions (documented, PART 23)

- Canvas 2D (not Three.js): map outputs are 509–1257 cells; canvas rect
  rendering is GPU-composited and sufficient on a judge laptop. No new heavy
  dependency was introduced, per the minimal-architecture rule.
- Payloads capped via `max_map_cells` (default 2000, adjustable 500–4000);
  full raw clouds are never fetched — views render backend `map_cells`.
- Camera: per-view 2D pan (drag) / zoom (wheel) / reset (double-click, button,
  new frame, session Reset). True oblique-3D rendering is NOT implemented
  (documented limitation; top-down + elevation encoding instead).
- React: `useMemo`/`useCallback` for frame index and handlers; single source
  of truth (`useReplay`) prevents cross-frame staleness; no fake animation —
  views update only from actual backend results.

## Truthfulness rules enforced

- Every number from backend responses; annotation sources preserved per cell;
  no ML confidence shown (no trained model); no security/classification
  claims beyond "LOCAL DEMO · PUBLIC DATASET · OFFLINE/LOCAL".
