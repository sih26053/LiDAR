# Metric Definitions — Paradox Protocol (existing project definitions)

| Metric | Definition | Unit | Source |
|---|---|---|---|
| Input points | Raw points loaded for the frame (`len(raw_points)`) | points | backend |
| Processed points | Points surviving preprocessing (finite + ROI) | points | backend |
| Map cells | Rows of the adaptive 2.5D map (`len(map_df)`) | cells | backend |
| Fine cells | Cells with resolution exactly 0.05 m | cells | backend |
| Medium cells | Cells with resolution exactly 0.10 m | cells | backend |
| Coarse cells | Cells with resolution 0.20 m **or** 0.50 m | cells | backend |
| Avg resolution | Mean of cell resolutions | m | backend |
| Mapping latency | Mapper stage timer only | ms | backend |
| Total latency | Sum of stage timers (preprocessing + perception + feature + importance + resolution + mapping); compute only | ms | backend |
| Serialization latency | Time to serialize map cells to JSON | ms | backend |
| API wall clock | Load + pipeline + serialize, measured around the request | ms | backend |
| FPS / Throughput | `1000 / total_latency_ms` (`src/robustness.safe_fps`; NaN on zero/invalid — never invented) | FPS | backend |

Frontend rendering time is not measured and is never part of any latency metric.
The dashboard shows Mapping latency (`core ML time`), Total latency
(`compute only`) and API wall clock (`incl. overhead`) as separate cards.
