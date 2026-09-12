# Data Interface Contract — Perception ↔ Importance Engine

Stable data contract between the data/perception side and the existing
Stage-1 core. The producer may change internally without forcing the
consumer to change, as long as this contract stays valid.

```text
DATA / PERCEPTION SIDE PRODUCES:
- LiDARFrame
- PerceptionResult

FEATURE ADAPTER PRODUCES:
- RegionFeatures

IMPORTANCE ENGINE CONSUMES:
- RegionFeatures (via RegionFeatures.to_legacy_region())

RESOLUTION ENGINE CONSUMES:
- ImportanceResult (safe_importance)

2.5D MAPPER CONSUMES:
- legacy region (points + semantic_class + confidence) + ImportanceResult
```

Ownership:

```text
Rajashree / data-perception side
        ↓
standard interface (this document + src/)
        ↓
Anik / Importance Engine
```

## 0. LiDAR contract

Project contract expects:

```text
N × 4
[x, y, z, intensity]
```

* `x, y, z` in metres, `intensity` = sensor intensity value.
* No fifth field. `points.shape == (N, 4)`, all finite, `N > 0`.

## 1. Structures

* `LiDARFrame(frame_id, timestamp, points)` — raw input.
* `PerceptionResult(frame_id, points, semantic_labels, confidence, distance,
  elevation, roughness, point_density)` — point-level arrays, all length N:
  `points (N×4)`, `semantic_labels (N)`, `confidence (N)`,
  `distance (N)`, `elevation (N)`, `roughness (N)`, `point_density (N)`.
* `RegionFeatures(...)` — region-level aggregation, the main engine input.

## 2. Field contract (RegionFeatures)

| Field | Meaning | Type | Unit / Range |
|---|---|---|---|
| region_id | unique region ID | int | — (>= 0) |
| x | region X centroid | float | metres |
| y | region Y centroid | float | metres |
| distance | distance from sensor (centroid range) | float | metres (>= 0) |
| elevation | region height (mean z) | float | metres |
| roughness | local terrain/geometric variation | float | [0,1] normalized |
| point_density | LiDAR point density | float | points/region (count broadcast) |
| semantic_label | semantic class | str | class name (see vocab) |
| semantic_importance | class importance | float | [0,1] |
| confidence | perception confidence | float | [0,1] |
| dynamic_relevance | dynamic-object relevance | float | [0,1] |
| uncertainty | uncertainty score/proxy | float | [0,1] |
| point_count | number of points | int | count (>= 0) |

Class vocabulary: `pedestrian, bicycle, motorcycle, vehicle, traffic_cone,
barrier, building, vegetation, rough_terrain, road, unknown_obstacle,
unknown`. Mapping in `src/data_types.py` (`SEMANTIC_IMPORTANCE`).

## 3. Normalization contract

Already normalized `[0.0, 1.0]`: `semantic_importance, confidence,
roughness / terrain complexity, dynamic_relevance, uncertainty`.

Physical/raw (never silently normalized by the interface):
`distance → metres`, `elevation → metres`,
`point_density → points/region`. The Importance Engine normalizes distance
via `D = clip(1 - d/100, 0, 1)`. Do not double-normalize.

Engine formula (unchanged Stage-1):
`I_base = 0.30·D + 0.30·S + 0.15·T + 0.15·M + 0.10·U`,
`I_safe = min(1, I_base + 0.5·U)`, resolution
`≥0.75→0.05 m, ≥0.50→0.10 m, ≥0.25→0.20 m, else 0.50 m`.

## 4. Missing-data policy

* Missing semantic prediction: `semantic_label = "unknown"`,
  `confidence = 0.0`, `semantic_importance = 0.0`. Never invent a class.
* Missing dynamic info: documented semantic proxy (`DYNAMIC_PROXY`;
  `unknown → 0.20`). Never claim measured velocity.
* Missing uncertainty: documented proxy `uncertainty = 1 − confidence`
  (`unknown/conf 0 → 0.5`). Marked as proxy, never real sensor uncertainty.
* Region aggregation: majority vote wins; confidence = mean of winning
  class points; no fake probabilities.

## 5. Adapter

`src/feature_adapter.py`: deterministic 4 m XY grid (Stage-1 bin size),
`distance = hypot(cx, cy)`, `elevation = mean(z)`,
`roughness = min(1, std(z)/0.5)`, `point_density = count`.
Replaceable: a real-data adapter only reimplements
`PerceptionResult → List[RegionFeatures]`.

## 6. Future real-data compatibility

```text
Real LiDAR (.bin via a future loader, NOT this task)
   ↓ LiDARFrame (N×4)
   ↓ PerceptionResult (point-level)
   ↓ RegionFeatures (adapter)
   ↓ Existing Importance Engine (unchanged)
```

The engine never knows the dataset source, file layout, or label encoding.

## 7. What this task does NOT do

No dataset download, no Kaggle credentials, no Stage 2, no segmentation
model, no engine redesign, no dashboard/database, no real-LiDAR or
accuracy/FPS/IoU claims. All numbers are synthetic and labeled as such.
