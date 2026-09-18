# Simulation Handoff — Adaptive Variable-Resolution 2.5D LiDAR Mapping

Frozen prototype handoff (`prototype-final-v1`, 15 September 2026) +
16 September simulation integration (7-frame replay validated, all MATCH frozen).
Independently executable: `python run_demo.py` needs only this folder,
its `requirements.txt`, the nuScenes Mini dataset, and one bundled sample.
Full integration record: `notebooks/08_Simulation_Integration_and_Demo.ipynb`
(clean + `.executed.` top-to-bottom run); demo frame list:
`config/demo_config.json` (references frozen `config/final_config.json`,
no separate weights/thresholds).

## 1. Project purpose

Convert one nuScenes LiDAR sweep into an adaptive 2.5D map whose cell size
follows measured region importance (close / critical / uncertain regions
stay fine at 5 cm; far / uniform regions coarsen up to 50 cm).

## 2. Input format

- Bundled demo sample: `sample_data/<token>_LIDAR_TOP_xyzi.npy`
  (float64 N×4 `[x, y, z, intensity]`, all finite) + `<token>_metadata.csv`
  (`frame_id, sample_token, lidar_token, timestamp, source_file, n_raw,
  n_processed`).
- Live runs additionally need nuScenes Mini (`--nusc-root`) for the
  annotation reference of the same sample token.

## 3. Required dependencies

See `requirements.txt` (exact pinned versions, all queried from the
validated environment):

```text
numpy==2.4.4
pandas==3.0.2
nuscenes-devkit==1.2.0
pyquaternion==0.9.9
matplotlib==3.11.1
scipy==1.17.1
```

Install with `pip install -r requirements.txt`.
torch / open3d are NOT required (the pipeline never imports them).
matplotlib/scipy are required only because the shipped `src/__init__.py`
eagerly imports the full package (visualization/baselines); the mapping
path itself uses numpy/pandas only.

## 4. How to load configuration

```python
import json
final_cfg = json.load(open("config/final_config.json"))
```

`config/` also carries `importance_config.json`, `resolution_config.json`,
`failsafe_config.json`, `semantic_mapping.json`. The `.py` files next to
them are the shipped validators/defaults; the JSON files are authoritative
for the frozen selection **W_BASE+THRESH_C**.

## 5. How to load model artifact

There is none (CASE B). `models/` contains only `NO_TRAINED_WEIGHTS.txt`.
The prototype is deterministic/annotation-driven; do not add fake weights.
`run_demo.py` asserts that no `*.pth / *.pt / *.onnx / *.pkl` file exists.

## 6. How to run one LiDAR frame

```bash
python run_demo.py [--nusc-root ../data/raw/nuscenes] [--sample-token TOKEN] [--seed 42]
```

`--nusc-root` defaults to the sibling development dataset
(`../data/raw/nuscenes`); override it wherever your copy of nuScenes Mini
lives. Prints staged progress plus one JSON summary line; saves the map to
`sample_outputs/adaptive_map_<token>.csv`.

## 7. Expected output

One row per 2 m integration region (prototype: one region → one cell):

`x, y, elevation, occupancy, semantic_class, confidence, importance,
resolution, point_count, region_id, semantic_source`

plus the stdout JSON (`map_cells, mean_importance, mean_resolution,
resolution_distribution, latency_ms, fps`).

## 8. Adaptive-map fields

- `x, y, elevation`: measured geometry (metres).
- `occupancy`: 1.0 evidence-presence indicator (NOT probabilistic occupancy).
- `semantic_class` + `semantic_source`: project class and its provenance
  (`annotation` inside a projected box, else `fallback`/`unknown`).
- `confidence`: 0.0 placeholder for non-ML regions (never ML confidence).
- `importance`: frozen Importance Engine output in [0, 1].
- `resolution`: one of {0.05, 0.10, 0.20, 0.50} m from THRESH_C bands
  (≥0.70 / ≥0.45 / ≥0.20 / else, boundaries inclusive).

## 9. Configuration files

`config/final_config.json` (authoritative) + the four split files above.
Frozen selection: weights {distance 0.30, semantic 0.30, terrain 0.15,
dynamic 0.15, uncertainty 0.10}, max distance 100 m, λ 0.15, seed 42,
integration cell 2.0 m.

## 10. Dataset requirements

nuScenes Mini (`v1.0-mini`) at `--nusc-root` — needed only for the
annotation reference of the demo token. Without it the demo exits with
code 2 and a JSON `BLOCKED`-style error (it never fabricates semantics).

## 11. Known limitations

- Annotation-derived labels are references, never neural predictions.
- No trained model: no predictions, no confidence (NaN / 0.0 placeholder).
- `dynamic_relevance` is a heuristic prior, not measured velocity.
- Uncertainty 0.5 is a documented fallback proxy, not calibrated.
- FPS is measured CPU throughput, not a real-time claim.
- Full statement: `results/final/reports/limitations.md` (dev repo).

## 12. Example execution

```bash
pip install -r requirements.txt
python run_demo.py
# config_load: OK ...
# handoff_import: OK
# artifact_load: OK (CASE B: no trained weights; config-only)
# sample_load: OK (5991fad3280c4f84b331536c32001a04, n=34359)
# pipeline_execution: OK
# output_generation: OK (adaptive_map_....csv, cells=533)
# {"status": "PASS", ...}
```

16-Sep replay evidence (measured, same frozen pipeline): 7/7 frames PASS,
cells 533/759/1132/1257/1059/509/1055 — identical to the frozen benchmark;
handoff `sample_outputs/adaptive_map_....csv` is byte-identical (sha256)
to the simulation replay map for the demo frame. Representative 4-view figure:
`sample_outputs/representative_demo_5991fad3280c4f84b331536c32001a04.png`
(input LiDAR + adaptive 2.5D map + resolution view + measured statistics).
Sim wall-clock latencies are recorded separately and differ honestly from the
frozen compute-only timings (timing-scope difference, documented in the dev
repo issue log SIM-001); frozen results were not overwritten.

Pipeline recap:

```text
LiDAR input -> preprocessing -> perception (annotation semantics) ->
feature extraction -> Importance Engine -> Resolution Engine ->
Adaptive 2.5D Mapper -> 2.5D map output
```

Reproducibility: `environment_info.json`, `final_manifest.json`,
`file_hashes.json` are copied from the validated `results/final/` run.
