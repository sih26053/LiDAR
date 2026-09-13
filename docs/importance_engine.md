# Importance Engine v1 + Resolution Engine v1

Stage-1 adaptive-resolution decision layer for the Paradox Protocol pipeline.
This is an **initial prototype / engineering heuristic**, subject to validation
and benchmark testing. Nothing here is claimed to be optimal, production-ready,
or a formal safety guarantee.

```text
RegionFeatures (9-Sept contract, unchanged)
        ↓ normalized inputs
Importance Engine v1  →  final importance ∈ [0,1]
        ↓
Resolution Engine v1  →  cell size in {0.05, 0.10, 0.20, 0.50} m
        ↓
Adaptive 2.5D Mapper (existing interface, unchanged)
```

## 1. Purpose

Decide **how much spatial detail each region deserves**. High-importance
regions (nearby pedestrians, complex/uncertain geometry) are mapped finely;
low-importance regions (smooth empty road far away) are mapped coarsely,
saving compute while preserving information where it matters.

## 2. Input fields

Consumes the existing `RegionFeatures` contract (`src/data_types.py`) unchanged:

| Field | Meaning | Unit / range |
|---|---|---|
| `distance` | centroid range from sensor | metres, `>= 0` |
| `semantic_importance` | class importance (pedestrian 1.0 … road 0.1) | [0,1] |
| `roughness` (= terrain complexity `T`) | local geometric variation `min(1, std(z)/0.5)` | [0,1] |
| `dynamic_relevance` | dynamic-object relevance (documented semantic proxy) | [0,1] |
| `uncertainty` | uncertainty score/proxy (`1 − confidence`) | [0,1] |
| `confidence` | perception confidence (validated, passed through) | [0,1] |

The engine also accepts legacy Stage-1 regions (`distance_m`,
`terrain_complexity`) and plain dicts via `extract_features()`; no caller
was rewritten to accommodate v1.

## 3. Units and normalization

- `distance` stays a **physical quantity (metres)** until distance
  normalization inside the engine. The interface never pre-normalizes it.
- All other factors are already normalized [0,1] by the producer; the engine
  validates (rejects NaN/inf/out-of-range/negative distance with `ValueError`)
  rather than silently hiding bad data, then defensively clips to [0,1].

## 4. Formula

Distance score:

$$D = \mathrm{clip}_{[0,1]}\left(1 - \frac{d}{100}\right)$$

Base importance:

$$I_{base} = 0.30\,D + 0.30\,S + 0.15\,T + 0.15\,M + 0.10\,U$$

Uncertainty-aware safety heuristic:

$$I_{safe} = \min\left(1,\, I_{base} + 0.15\,U\right)$$

`confidence` is validated as part of the interface contract but does not
enter the v1 formula; `uncertainty` carries the perception-quality term.

## 5. Initial weights

```python
WEIGHTS = {"distance": 0.30, "semantic": 0.30, "terrain": 0.15,
           "dynamic": 0.15, "uncertainty": 0.10}  # sum = 1.0
MAX_DISTANCE = 100.0
UNCERTAINTY_LAMBDA = 0.15
```

> The weights are initial engineering assumptions and are not claimed to be
> optimized.

## 6. Distance normalization

`normalize_distance(d, max_distance=100.0)` is a standalone, independently
tested function (not buried in `calculate()`):

| distance | score |
|---|---|
| 0 m | 1.00 |
| 10 m | 0.90 |
| 25 m | 0.75 |
| 50 m | 0.50 |
| 75 m | 0.25 |
| 100 m | 0.00 |
| >100 m | 0.00 (clipped) |

Negative / non-finite distances and non-positive `max_distance` raise
`ValueError` (no division by zero, no silent acceptance).

## 7. Uncertainty modifier

`apply_uncertainty_modifier()` adds `λ·U` (λ = 0.15, configurable) and clips
to [0,1]. Effect: high uncertainty preserves additional detail and reduces
the risk of excessive simplification. This is a **safety-oriented heuristic,
not a formal probabilistic safety guarantee**. It can never reduce the final
score below the base score.

Note: the canonical Stage-1 core (`src/stage1_engines.py`) uses λ = 0.5 and
is preserved untouched; v1 (λ = 0.15) is the 10-September prototype policy.
Ablation studies will decide the tuned value.

## 8. Output range

`ImportanceEngine.calculate(region)` returns an `ImportanceResult` with
`base_importance` and `final_importance` (plus a `safe_importance` alias for
Stage-1 consumer compatibility), both guaranteed ∈ [0,1]. The Resolution
Engine consumes only the final score.

## 9. Resolution mapping

Initial prototype resolution thresholds, subject to later tuning and
benchmark validation:

| Importance | Cell size |
|---|---|
| `0.75 ≤ I ≤ 1.00` | `0.05 m` (5 cm, fine) |
| `0.50 ≤ I < 0.75` | `0.10 m` (10 cm, medium-fine) |
| `0.25 ≤ I < 0.50` | `0.20 m` (20 cm, medium-coarse) |
| `0.00 ≤ I < 0.25` | `0.50 m` (50 cm, coarse) |

Smaller cell = finer detail. Thresholds live in
`config/resolution_config.py`; `ResolutionEngine.select_resolution()`
validates its input ∈ [0,1].

## 10. Why weights/thresholds are configurable

They are prototype values. Tuning them later (ablation studies, real-data
benchmarks) must not require code changes, so they live in `config/` with
import-time validation, and both engines accept overrides via constructors.

## 11. Limitations

- Scalar region-level heuristic; no spatial context between regions.
- `dynamic_relevance`/`uncertainty` are documented proxies, not measured
  velocity / sensor uncertainty.
- `confidence` validated but unused in the v1 formula.
- Prototype policy only validated on synthetic regions so far.

## 12. Future ablation / tuning plan

1. Run on real nuScenes-backed `RegionFeatures` via the same adapter interface.
2. Grid-search weights and thresholds against map-quality vs cell-count metrics.
3. Compare λ = 0.15 (v1) vs λ = 0.5 (canonical Stage-1) on uncertain regions.
4. Benchmark end-to-end (accuracy, cell count, latency) before any
   production or deployment claim.

## 13. Verification (synthetic, computed from the actual engine)

Same-distance novelty demo (both @70 m): pedestrian
(base 0.575, final 0.605 → 0.10 m) vs empty road
(base 0.145, final 0.160 → 0.50 m) — same distance, different
semantic/dynamic context, different importance and resolution.

Full 8-scenario matrix and monotonicity checks (distance, semantic,
uncertainty, resolution) are exercised in `tests/test_importance_engine.py`,
`tests/test_resolution_engine.py`, and the Stage-1 path
`tests/test_stage1_integration.py` (synthetic → adapter → v1 engines →
existing mapper). No real-dataset dependency.
