# Importance Engine v1 + Resolution Engine v1 (10 September 2026)

> Status: initial prototype / engineering heuristic. Weights and thresholds are
> configurable prototype values — **not** scientifically optimized, **not**
> benchmarks, **not** a production-safety claim. All validation below uses
> controlled synthetic region features (no nuScenes download, no trained model).

Pipeline position (9-September contract preserved, not redesigned):

```text
RegionFeatures
      │
      ├── distance
      ├── semantic_importance
      ├── terrain_complexity (RegionFeatures.roughness)
      ├── dynamic_relevance
      ├── uncertainty
      └── confidence (validated, passed through; no v1 weight)
              │
              ▼
       Normalization (D = 1 - d/dmax, clip01)
              │
              ▼
      Importance Engine v1 → Final Importance [0,1]
              │
              ▼
      Resolution Engine v1 → {0.05, 0.10, 0.20, 0.50} m
              │
              ▼
      Adaptive 2.5D Mapper (existing interface, unchanged)
```

Implementation: `src/importance_engine.py`, `src/resolution_engine.py`.
Configuration: `config/importance_config.py`, `config/resolution_config.py`.
Tests: `tests/test_importance_engine.py`, `tests/test_resolution_engine.py`,
`tests/test_stage1_integration.py`. Notebook: `notebooks/01_Model_Development.ipynb`.

## 1. Purpose

Convert region-level perception/geometric features into a normalized
**importance score**, then convert that score into an **adaptive spatial
resolution** (2.5D map cell size). High-importance regions (nearby, semantically
critical, geometrically complex, dynamic, or uncertain) receive finer cells;
low-importance regions (far, smooth, static, certain background) receive coarse
cells, saving compute while preserving detail where it matters.

## 2. Inputs

Consumes the existing 9-September `RegionFeatures` contract
(`src/data_types.py`) unchanged. Primary v1 inputs:

| Field | Meaning | Unit / range |
|---|---|---|
| `distance` | centroid range from sensor | metres, `>= 0` (physical until normalized inside the engine) |
| `semantic_importance` | class importance (e.g. pedestrian 1.0 … road 0.1) | [0,1] |
| `terrain_complexity` (`RegionFeatures.roughness`) | local geometric variation, `min(1, std(z)/0.5)` | [0,1] |
| `dynamic_relevance` | dynamic-object relevance (documented semantic proxy, not measured velocity) | [0,1] |
| `uncertainty` | uncertainty score / proxy (`1 − confidence`) | [0,1] |
| `confidence` | perception confidence | [0,1], validated + passed through (see §6) |

Full region record also carries `region_id, x, y, elevation, point_density,
semantic_label, point_count`. The engine accepts `RegionFeatures`, legacy
Stage-1 regions (`distance_m` / `terrain_complexity`), and plain dicts via
`extract_features()` — no caller was rewritten. Invalid inputs (negative
distance, out-of-range / NaN / inf normalized fields) raise `ValueError`;
they are never silently clipped or hidden.

## 3. Mathematical model

Distance normalization (`normalize_distance`, standalone + tested):

```text
D = clip01(1 − d / dmax),   dmax = 100 m
```

| distance | D |
|---|---|
| 0 m | 1.00 |
| 10 m | 0.90 |
| 50 m | 0.50 |
| 75 m | 0.25 |
| 100 m | 0.00 |
| >100 m | 0.00 (clipped) |

Base importance (weighted sum, weights sum to 1.0 within 1e-6):

```text
I_base = wd·D + ws·S + wt·T + wm·M + wu·U
```

Uncertainty-aware safety heuristic (prototype fallback, not a formal guarantee):

```text
I_safe = min(1, I_base + λ·U),   λ = 0.15
```

`I_safe` is exposed as `final_importance` (with a `safe_importance`
compatibility alias). Guaranteed `0.0 ≤ importance ≤ 1.0`. The engine exposes
`distance_score`, `base_importance`, `final_importance` separately — no hidden
intermediate math.

## 4. Initial weights

```python
MAX_DISTANCE = 100.0
WEIGHTS = {
    "distance": 0.30,
    "semantic": 0.30,
    "terrain": 0.15,
    "dynamic": 0.15,
    "uncertainty": 0.10,
}  # sum == 1.0
UNCERTAINTY_BOOST = 0.15  # λ (config name UNCERTAINTY_LAMBDA / LAMBDA_UNCERTAINTY)
```

These are **initial prototype engineering parameters only** — documented
assumptions, not optimized or benchmarked values. They live in
`config/importance_config.py` with import-time validation so tuning never
requires code changes. No weight optimization was performed in this task.
Note: the canonical Stage-1 core (`src/stage1_engines.py`) uses λ = 0.5 and is
preserved untouched; v1 (λ = 0.15) is the 10-September prototype policy.

## 5. Resolution mapping

Initial engineering thresholds (`config/resolution_config.py`, configurable):

```text
importance >= 0.75 → 0.05 m (very_high / fine, 5 cm)
importance >= 0.50 → 0.10 m (high / medium-fine, 10 cm)
importance >= 0.25 → 0.20 m (medium / medium-coarse, 20 cm)
otherwise          → 0.50 m (low / coarse, 50 cm)
```

| Importance | Resolution |
|---|---|
| [0.75, 1.00] | 0.05 m |
| [0.50, 0.75) | 0.10 m |
| [0.25, 0.50) | 0.20 m |
| [0.00, 0.25) | 0.50 m |

Smaller cell = finer detail. Boundaries are inclusive of the finer bin,
deterministic, and unit-tested at
`0.00 → 0.50, 0.24 → 0.50, 0.25 → 0.20, 0.49 → 0.20, 0.50 → 0.10, 0.74 → 0.10,
0.75 → 0.05, 1.00 → 0.05`. API: `select_resolution()` with spec-name alias
`assign_resolution()` (+ module function and `ResolutionResult` struct).

## 6. Confidence

> **Confidence is part of the project interface but is not given an
> independent weight in Importance Engine v1. Future versions may incorporate
> confidence explicitly after evaluation.**

Concretely: `confidence` is a validated input (must be [0,1], finite), it is
preserved on the region record and on `ImportanceResult`, and it informs
safety/validation discussion — but it does **not** enter the v1 formula.
`uncertainty` carries the perception-quality term, so adding confidence as a
second quality weight without an explicit design decision would double-count
the same signal. A confidence-aware extension is deferred, not silently added.

## 7. Uncertainty

Uncertainty is used as an **importance-preservation mechanism** in v1: the
`λ·U` term can only raise (never lower) the final score, so high-uncertainty
regions keep finer detail when thresholds justify it instead of being
aggressively simplified. Documented as a **prototype uncertainty-aware
fallback mechanism**, not a validated safety system. Verified behavior:
identical regions differing only in `uncertainty` (0.1 vs 0.9) score higher
when uncertainty is higher, monotonically.

## 8. Limitations

* Weights are initial engineering assumptions — not optimized, no tuning claim.
* Thresholds are initial engineering values — subject to later ablation study.
* All v1 validation uses controlled **synthetic** regions — sanity/distribution
  checks only, never real-world benchmarks or FPS claims.
* Semantic quality depends entirely on the upstream perception system (v1 was
  validated with hand-defined synthetic features, no trained segmentation).
* `dynamic_relevance` / `uncertainty` here are documented proxies
  (semantic prior / `1 − confidence`), not measured velocity or calibrated
  sensor uncertainty.
* Scalar region-level heuristic: no inter-region spatial context.
* No production-safety claim is made; the uncertainty modifier is a heuristic.
* No dataset, credentials, models, frontend, or database were built in this task.

## Verification pointers (all computed from the actual engine)

* Same-distance test (@70 m): pedestrian (final ~0.605 → 0.10 m) scores finer
  than road (final ~0.16 → 0.50 m) — resolution is not distance-only.
* 500-region synthetic distribution (`seed 42`): counts/percentages per level
  plus min/max/mean importance, saved to
  `results/resolution_distribution.csv`.
* 5-region controlled table saved to
  `results/importance_engine_synthetic_results.csv`; figures under
  `results/figures/`.
* Run `pytest -q` (88 tests: normalization, weights, monotonicity,
  boundaries, edge cases, integration). Notebook ends with an assertion-gated
  `[PASS]/[FAIL]` validation summary.
