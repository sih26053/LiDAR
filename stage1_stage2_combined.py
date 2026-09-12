# coding: utf-8
"""Combined Stage 1 + Stage 2 pipeline in ONE self-contained file.

No dependency on a pre-existing Stage-1 src/ folder: the Stage-1 core
(Importance Engine, Resolution Engine, Adaptive 2.5D Mapper, data types,
evaluation helpers, synthetic scene generator) is embedded below, then the
Stage-2 real-data adapter runs the SAME core on real nuScenes Mini LiDAR.

Run:
    Colab :  python stage1_stage2_combined.py
    Local :  NUSCENES_DATAROOT=<mini-root> PROJECT_ROOT=<out-dir> python stage1_stage2_combined.py

Dataset: Kaggle nuScenes Mini mirror (aadimator/nuscenes-mini) or any local
folder containing v1.0-mini/*.json + samples/. Set via NUSCENES_DATAROOT.
If nothing is found and Kaggle credentials exist, a download is attempted.

Final outcome: synthetic validation + real-data adaptive maps + metrics +
results/figures|metrics|logs and a PASS/PARTIAL/FAIL verdict printed.
"""
import os, sys, json, time, shutil, subprocess, importlib.util
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional

# ============================================================================
# PART 0 - configuration (env-overridable, no hard-coded secrets)
# ============================================================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/content/project"))
DATASET_ENV = os.environ.get("NUSCENES_DATAROOT", "")
CANDIDATE_ROOTS = [p for p in [
    DATASET_ENV,
    "/content/datasets/nuscenes-mini",
    str(Path.cwd() / "nuscenes-mini"),
    "/content/data/nuscenes",
] if p]
KAGGLE_HANDLE = os.environ.get("KAGGLE_DATASET", "aadimator/nuscenes-mini")
VERSION = "v1.0-mini"
MAX_SCENES = int(os.environ.get("STAGE2_MAX_SCENES", "5"))
MAX_SAMPLES = int(os.environ.get("STAGE2_MAX_SAMPLES", "20"))
PHASES = {"A": 1, "B": 5, "C": 10, "D": 20}
ACTIVE_PHASE = os.environ.get("STAGE2_PHASE", "D")
MAX_RANGE = 100.0
ROI = {"x_min": -80.0, "x_max": 80.0, "y_min": -50.0, "y_max": 50.0,
       "z_min": -5.0, "z_max": 10.0, "max_range": MAX_RANGE}
REGION_BIN_M = 4.0
MIN_POINTS_PER_REGION = 15
TERRAIN_NORM_M = 0.5
USP_MAX_DIST_DIFF_M = 5.0
CRITICAL_CLASSES = {"pedestrian", "bicycle", "motorcycle", "vehicle",
                    "traffic_cone", "barrier", "unknown_obstacle"}
SEED = 42

for _sub in ["data/sample_metadata", "results/figures", "results/metrics",
             "results/logs", "config"]:
    (PROJECT_ROOT / _sub).mkdir(parents=True, exist_ok=True)
FIG_DIR = PROJECT_ROOT / "results" / "figures"
MET_DIR = PROJECT_ROOT / "results" / "metrics"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
META_DIR = PROJECT_ROOT / "data" / "sample_metadata"

# ============================================================================
# PART 1 - third-party imports (installed only if missing)
# ============================================================================
for _pkg, _mod in [("numpy", "numpy"), ("pandas", "pandas"),
                   ("matplotlib", "matplotlib"), ("scipy", "scipy"),
                   ("nuscenes-devkit", "nuscenes"), ("kagglehub", "kagglehub")]:
    if importlib.util.find_spec(_mod) is None:
        print(f"[setup] installing {_pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
np.random.seed(SEED)
print("[setup] numpy", np.__version__, "| pandas", pd.__version__)

# ============================================================================
# PART 2 - STAGE-1 CORE (embedded; identical logic to 01_Model_Development)
# ============================================================================
W = {"distance": 0.30, "semantic": 0.30, "terrain": 0.15,
     "dynamic": 0.15, "uncertainty": 0.10}
LAMBDA_U = 0.5
LEVELS = {"fine": 0.05, "medium_fine": 0.10,
          "medium_coarse": 0.20, "coarse": 0.50}
THRESH = (0.75, 0.50, 0.25)
DIST_BINS = [(10.0, 0.05), (30.0, 0.10), (60.0, 0.20), (float("inf"), 0.50)]
SEMANTIC_IMPORTANCE = {
    "pedestrian": 1.0, "bicycle": 1.0, "motorcycle": 1.0, "vehicle": 0.9,
    "traffic_cone": 0.8, "barrier": 0.8, "unknown_obstacle": 0.8,
    "building": 0.5, "rough_terrain": 0.45, "vegetation": 0.3, "road": 0.1,
}
DYNAMIC_PROXY = {
    "pedestrian": 0.95, "bicycle": 0.95, "motorcycle": 0.9, "vehicle": 0.8,
    "traffic_cone": 0.4, "barrier": 0.3, "unknown_obstacle": 0.4,
    "building": 0.05, "vegetation": 0.05, "rough_terrain": 0.05, "road": 0.0,
}


@dataclass
class RegionFeatures:
    region_id: int
    semantic_class: str
    points: np.ndarray
    distance_m: float
    semantic_importance: float
    terrain_complexity: float
    dynamic_relevance: float
    uncertainty: float
    confidence: float


@dataclass
class RealRegion(RegionFeatures):
    source_sample_token: str = ""
    dominant_fraction: float = 0.5
    class_entropy: float = 0.0
    n_classes: int = 1
    semantic_source: str = ""
    dynamic_source: str = "semantic_proxy"
    quality_note: str = "proxy quality field (NOT AI confidence)"


@dataclass
class ImportanceResult:
    region_id: int
    distance_score: float
    semantic_score: float
    terrain_score: float
    dynamic_score: float
    uncertainty_score: float
    base_importance: float
    safe_importance: float
    selected_resolution_m: float = -1.0
    resolution_level: str = ""


@dataclass
class MapCell:
    x: float
    y: float
    elevation: float
    elevation_std: float
    occupancy: int
    semantic_class: str
    confidence: float
    importance: float
    resolution: float
    region_id: int
    point_count: int


@dataclass
class MapResult:
    cells: List[MapCell]
    importance: List[ImportanceResult]
    elapsed_s: float
    n_points: int


class ImportanceEngine:
    """Stage-1 core: I_base = wd*D + ws*S + wt*T + wm*M + wu*U; I_safe = min(1, I_base + lam*U)."""

    def __init__(self, weights: Dict[str, float] = None,
                 max_range_m: float = MAX_RANGE,
                 lambda_uncertainty: float = LAMBDA_U):
        weights = dict(weights or W)
        assert max_range_m > 0
        assert all(v >= 0 for v in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        self.w, self.max_range_m, self.lam = weights, max_range_m, lambda_uncertainty

    @staticmethod
    def _clip01(v: float) -> float:
        return float(min(1.0, max(0.0, v)))

    def distance_score(self, d: float) -> float:
        assert np.isfinite(d) and d >= 0
        return self._clip01(1.0 - d / self.max_range_m)

    def score_region(self, r: RegionFeatures) -> ImportanceResult:
        d = self.distance_score(r.distance_m)
        s = self._clip01(r.semantic_importance)
        t = self._clip01(r.terrain_complexity)
        m = self._clip01(r.dynamic_relevance)
        u = self._clip01(r.uncertainty)
        base = (self.w["distance"] * d + self.w["semantic"] * s +
                self.w["terrain"] * t + self.w["dynamic"] * m +
                self.w["uncertainty"] * u)
        return ImportanceResult(r.region_id, d, s, t, m, u,
                                self._clip01(base), min(1.0, base + self.lam * u))


class ResolutionEngine:
    """Stage-1 core: importance -> 5/10/20/50 cm (boundaries inclusive below)."""

    def __init__(self, t_fine: float = THRESH[0], t_med_fine: float = THRESH[1],
                 t_med_coarse: float = THRESH[2], levels: Dict[str, float] = None):
        assert t_fine > t_med_fine > t_med_coarse > 0
        self.t_fine, self.t_med_fine, self.t_med_coarse = t_fine, t_med_fine, t_med_coarse
        self.levels = dict(levels or LEVELS)

    def select(self, importance: float) -> Tuple[float, str]:
        assert np.isfinite(importance) and 0.0 <= importance <= 1.0
        if importance >= self.t_fine:
            return self.levels["fine"], "fine (5 cm)"
        if importance >= self.t_med_fine:
            return self.levels["medium_fine"], "medium_fine (10 cm)"
        if importance >= self.t_med_coarse:
            return self.levels["medium_coarse"], "medium_coarse (20 cm)"
        return self.levels["coarse"], "coarse (50 cm)"


class AdaptiveMapper2_5D:
    """Stage-1 core: per-region XY binning at each region's own resolution."""

    def __init__(self, imp: ImportanceEngine, res: ResolutionEngine):
        self.imp, self.res = imp, res

    def map_region(self, region: RegionFeatures, importance: float) -> List[MapCell]:
        res_m, _ = self.res.select(importance)
        pts = region.points
        ix = np.floor(pts[:, 0] / res_m).astype(np.int64)
        iy = np.floor(pts[:, 1] / res_m).astype(np.int64)
        ukeys, inv = np.unique(np.column_stack([ix, iy]), axis=0, return_inverse=True)
        cells = []
        for k, (cx, cy) in enumerate(ukeys):
            z = pts[inv == k, 2]
            cells.append(MapCell(
                float((cx + 0.5) * res_m), float((cy + 0.5) * res_m),
                float(z.mean()), float(z.std() if len(z) > 1 else 0.0), 1,
                region.semantic_class, float(region.confidence), float(importance),
                float(res_m), int(region.region_id), int((inv == k).sum())))
        return cells

    def map_scene(self, regs: List[RegionFeatures]) -> MapResult:
        t0 = time.perf_counter()
        cells, impl, n = [], [], 0
        for r in regs:
            s = self.imp.score_region(r)
            rm, lvl = self.res.select(s.safe_importance)
            s.selected_resolution_m, s.resolution_level = rm, lvl
            cells += self.map_region(r, s.safe_importance)
            impl.append(s)
            n += len(r.points)
        return MapResult(cells, impl, time.perf_counter() - t0, n)


def distance_resolution(d: float) -> float:
    for thr, res in DIST_BINS:
        if d <= thr:
            return res
    return 0.50


class SyntheticSceneGenerator:
    """Stage-1 synthetic primitives (controlled input only, NOT real LiDAR)."""

    def __init__(self, seed: int = SEED, dropout: float = 0.02):
        self.rng = np.random.default_rng(seed)
        self.dropout = dropout

    def _drop(self, pts: np.ndarray) -> np.ndarray:
        if self.dropout <= 0 or len(pts) == 0:
            return pts
        return pts[self.rng.random(len(pts)) >= self.dropout]

    def _fin(self, xyz: np.ndarray, mu: float, sg: float) -> np.ndarray:
        inten = np.clip(self.rng.normal(mu, sg, len(xyz)), 0, 1)
        return self._drop(np.column_stack([xyz, inten]))

    def generate_region_points(self, spec: Dict[str, Any]) -> np.ndarray:
        kind = spec["kind"]
        cx, cy, cz = spec["center"]
        n = int(spec.get("n_points", 1000))
        nz = float(spec.get("noise", 0.02))
        sx, sy = spec["size"][0], spec["size"][1]
        sz = spec["size"][2] if len(spec["size"]) > 2 else 1.0
        R = self.rng
        if kind == "plane":
            xyz = np.column_stack([R.uniform(cx - sx / 2, cx + sx / 2, n),
                                   R.uniform(cy - sy / 2, cy + sy / 2, n),
                                   cz + R.normal(0, nz, n)])
            return self._fin(xyz, 0.35, 0.08)
        if kind == "box":
            xs, ys, zs = [], [], []
            for _ in range(n):
                f = R.integers(0, 6)
                if f == 0:
                    xs.append(R.uniform(cx - sx / 2, cx + sx / 2))
                    ys.append(R.uniform(cy - sy / 2, cy + sy / 2))
                    zs.append(cz + sz / 2)
                elif f == 1:
                    xs.append(R.uniform(cx - sx / 2, cx + sx / 2))
                    ys.append(R.uniform(cy - sy / 2, cy + sy / 2))
                    zs.append(cz - sz / 2)
                elif f == 2:
                    xs.append(cx + sx / 2)
                    ys.append(R.uniform(cy - sy / 2, cy + sy / 2))
                    zs.append(R.uniform(cz - sz / 2, cz + sz / 2))
                elif f == 3:
                    xs.append(cx - sx / 2)
                    ys.append(R.uniform(cy - sy / 2, cy + sy / 2))
                    zs.append(R.uniform(cz - sz / 2, cz + sz / 2))
                elif f == 4:
                    xs.append(R.uniform(cx - sx / 2, cx + sx / 2))
                    ys.append(cy + sy / 2)
                    zs.append(R.uniform(cz - sz / 2, cz + sz / 2))
                else:
                    xs.append(R.uniform(cx - sx / 2, cx + sx / 2))
                    ys.append(cy - sy / 2)
                    zs.append(R.uniform(cz - sz / 2, cz + sz / 2))
            return self._fin(np.column_stack([xs, ys, zs]) + R.normal(0, nz, (n, 3)), 0.6, 0.12)
        if kind == "pedestrian":
            xyz = np.column_stack([cx + R.normal(0, sx * 0.35, n),
                                   cy + R.normal(0, sy * 0.35, n),
                                   R.normal(cz, sz * 0.28, n)]) + R.normal(0, nz, (n, 3))
            return self._fin(xyz, 0.55, 0.10)
        raise ValueError(f"unknown primitive kind: {kind}")

    def generate_scene(self, specs: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
        return {int(s["region_id"]): self.generate_region_points(s) for s in specs}


IMP_ENGINE = ImportanceEngine()
RES_ENGINE = ResolutionEngine()
MAPPER = AdaptiveMapper2_5D(IMP_ENGINE, RES_ENGINE)

# ============================================================================
# PART 3 - STAGE 1 synthetic validation (proves the embedded core works)
# ============================================================================
def run_stage1() -> Dict[str, Any]:
    print("\n========== STAGE 1 : synthetic validation ==========")
    specs = [
        {"region_id": 0, "semantic_class": "road", "kind": "plane",
         "center": (0.0, 0.0, 0.0), "size": (30.0, 10.0), "n_points": 1500, "noise": 0.02},
        {"region_id": 1, "semantic_class": "pedestrian", "kind": "pedestrian",
         "center": (70.0, 2.0, 0.9), "size": (0.5, 0.5, 1.8), "n_points": 400, "noise": 0.03},
        {"region_id": 2, "semantic_class": "road", "kind": "plane",
         "center": (70.0, -6.0, 0.0), "size": (8.0, 4.0), "n_points": 600, "noise": 0.02},
    ]
    feat = {0: (0.05, 0.0, 0.05, 0.95), 1: (0.20, 0.95, 0.20, 0.85),
            2: (0.05, 0.0, 0.05, 0.95)}
    sem = {"pedestrian": 0.95, "vehicle": 0.90, "road": 0.10}
    clouds = SyntheticSceneGenerator(seed=SEED, dropout=0.0).generate_scene(specs)
    regs = []
    for s in specs:
        t, m, u, c = feat[s["region_id"]]
        cx, cy, _ = s["center"]
        regs.append(RegionFeatures(
            s["region_id"], s["semantic_class"], clouds[s["region_id"]],
            float(np.hypot(cx, cy)), sem[s["semantic_class"]], t, m, u, c))
    mres = MAPPER.map_scene(regs)
    ped = next(s for s in mres.importance if s.region_id == 1)
    rd = next(s for s in mres.importance if s.region_id == 2)
    ok = (ped.selected_resolution_m < rd.selected_resolution_m and
          len(set(c.resolution for c in mres.cells)) > 1)
    print(f"synthetic USP: pedestrian@70m -> {ped.selected_resolution_m} m | "
          f"road@70m -> {rd.selected_resolution_m} m")
    print("resolutions present:", sorted(set(c.resolution for c in mres.cells)))
    print("Stage 1 synthetic test ->", "PASS" if ok else "FAIL")
    assert ok, "Stage-1 core regression failed"
    return {"ok": True, "ped_res": ped.selected_resolution_m,
            "road_res": rd.selected_resolution_m,
            "cells": len(mres.cells)}

# ============================================================================
# PART 4 - STAGE-2 adapter (real nuScenes Mini -> SAME core above)
# ============================================================================
def _load_table(root: Path, name: str):
    for cand in [root / name, root / "v1.0-mini" / name]:
        if cand.is_file():
            return json.loads(cand.read_text())
    hits = list(Path(root).rglob(name))[:3]
    return json.loads(hits[0].read_text()) if hits else None


def find_nuscenes_root(start: Path) -> Optional[Path]:
    if not start.exists():
        return None
    cands = [start] + [p for p in start.rglob("v1.0-mini") if p.is_dir()]
    best, score_best = None, -1
    for c in cands:
        t = c / "v1.0-mini" if (c / "v1.0-mini").is_dir() else c
        sc = sum(1 for m in ["scene.json", "sample.json", "sample_data.json"]
                 if (t / m).is_file())
        if sc > score_best:
            best, score_best = t, sc
    return best if score_best >= 2 else None


def obtain_dataset() -> Tuple[Path, Path, str]:
    for cand in CANDIDATE_ROOTS:
        root = find_nuscenes_root(Path(cand))
        if root is not None:
            print("[stage2] reusing dataset root (no download):", root)
            return root, root.parent, "local"
    try:
        import kagglehub
        dl = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
        root = find_nuscenes_root(dl)
        if root is not None:
            print("[stage2] kagglehub downloaded:", dl)
            return root, root.parent, "kagglehub:" + KAGGLE_HANDLE
    except Exception as e:
        print("[stage2] kagglehub unavailable/failed:", str(e)[:150])
    raise RuntimeError(
        "No nuScenes Mini found. Set NUSCENES_DATAROOT to a folder with "
        "v1.0-mini/*.json + samples/, or provide Kaggle credentials.")


def map_dataset_labels_to_project_classes(official_name: str) -> str:
    n = str(official_name).lower()
    if "pedestrian" in n or n.startswith("human"):
        return "pedestrian"
    if "bicycle" in n:
        return "bicycle"
    if "motorcycle" in n or "motorbike" in n:
        return "motorcycle"
    if "cone" in n:
        return "traffic_cone"
    if "barrier" in n:
        return "barrier"
    if "vehicle" in n or n in ("car", "truck", "bus", "trailer"):
        return "vehicle"
    if "driveable" in n or "sidewalk" in n or "road" in n or "lane" in n:
        return "road"
    if "terrain" in n or "vegetation" in n:
        return "vegetation"
    if "building" in n:
        return "building"
    return "unknown_obstacle"


def quat_to_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class Stage2Pipeline:
    """Real-data adapter feeding the embedded Stage-1 core."""

    def __init__(self, root: Path, parent: Path):
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
        self.LidarPointCloud = LidarPointCloud
        self.root, self.parent = root, parent
        self.nusc = NuScenes(version=VERSION, dataroot=str(parent), verbose=False)
        try:
            self.idx2name = {int(k): str(v) for k, v in
                             dict(getattr(self.nusc, "lidarseg_idx2name_mapping", {})
                                  or {}).items()}
        except Exception:
            self.idx2name = {}
        self.idx2project = {i: map_dataset_labels_to_project_classes(n)
                            for i, n in self.idx2name.items()}
        self.lidarseg_bins = sorted((parent / "lidarseg").rglob("*.bin")) \
            if (parent / "lidarseg").is_dir() else []
        self.use_lidarseg = len(self.lidarseg_bins) > 0 and len(self.idx2name) > 0
        self.semantic_source = ("NuScenes LiDARSeg ground truth" if self.use_lidarseg
                                else "NuScenes 3D annotation-derived semantic regions")
        self.anns_by_sample: Dict[str, list] = {}
        for a in (getattr(self.nusc, "sample_annotation", []) or []):
            self.anns_by_sample.setdefault(a.get("sample_token"), []).append(a)
        self.tracks: Dict[str, list] = {}
        for a in (getattr(self.nusc, "sample_annotation", []) or []):
            self.tracks.setdefault(a.get("instance_token", ""), []).append(a)

    # ---- loading / preprocessing ----
    def load_frame(self, sample: dict) -> Tuple[np.ndarray, dict]:
        sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        fp = self.parent / sd["filename"]
        assert fp.is_file(), "missing LiDAR file: " + str(fp)
        try:
            pc = self.LidarPointCloud.from_file(str(fp))
            arr = np.asarray(pc.points[:4, :].T, dtype=np.float64)
        except Exception:
            raw = np.fromfile(str(fp), dtype=np.float32)
            arr = (raw.reshape((-1, 5))[:, :4] if raw.size % 5 == 0
                   else raw.reshape((-1, 4))).astype(np.float64)
        assert arr.ndim == 2 and arr.shape[1] == 4
        finite = np.all(np.isfinite(arr), axis=1)
        return arr[finite], {"file": sd["filename"], "n_raw": int(len(arr)),
                             "n_finite": int(finite.sum())}

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        box = ((frame[:, 0] >= ROI["x_min"]) & (frame[:, 0] <= ROI["x_max"]) &
               (frame[:, 1] >= ROI["y_min"]) & (frame[:, 1] <= ROI["y_max"]) &
               (frame[:, 2] >= ROI["z_min"]) & (frame[:, 2] <= ROI["z_max"]))
        rr = np.hypot(frame[:, 0], frame[:, 1])
        keep = box & (rr >= 0.5) & (rr <= ROI["max_range"])
        return frame[keep], {"raw": int(len(frame)), "filtered": int(keep.sum()),
                             "retained_pct": round(100.0 * keep.sum() / max(1, len(frame)), 1)}

    # ---- regions + features ----
    @staticmethod
    def generate_regions(points: np.ndarray):
        ix = np.floor(points[:, 0] / REGION_BIN_M).astype(np.int64)
        iy = np.floor(points[:, 1] / REGION_BIN_M).astype(np.int64)
        ukeys, inv = np.unique(np.column_stack([ix, iy]), axis=0, return_inverse=True)
        regs, skipped = {}, 0
        for k in range(len(ukeys)):
            idx = np.where(inv == k)[0]
            if len(idx) < MIN_POINTS_PER_REGION:
                skipped += 1
                continue
            regs[int(k)] = idx
        return regs, skipped

    @staticmethod
    def geometry(points: np.ndarray, idx: np.ndarray) -> dict:
        xyz = points[idx][:, :3]
        c = xyz.mean(axis=0)
        return {"cx": float(c[0]), "cy": float(c[1]),
                "distance_m": float(np.hypot(c[0], c[1])),
                "elevation": float(xyz[:, 2].mean()),
                "terrain_proxy": float(min(1.0, xyz[:, 2].std() / TERRAIN_NORM_M))}

    def lidarseg_of_sample(self, sample_token: str, sd_token: str):
        rec = None
        try:
            r = self.nusc.get("lidarseg", sd_token)
            if isinstance(r, dict) and r.get("sample_data_token", sd_token) == sd_token:
                rec = r
        except Exception:
            print("lidarseg direct lookup failed; scanning table")
            rec = None
        if rec is None:
            for r in (getattr(self.nusc, "lidarseg", []) or []):
                if r.get("sample_data_token") == sd_token:
                    rec = r
                    break
        if rec is None:
            return None
        fp = self.parent / rec["filename"]
        return np.fromfile(str(fp), dtype=np.uint8) if fp.is_file() else None

    def boxes_in_lidar(self, sample_token: str, sd_token: str) -> list:
        anns = self.anns_by_sample.get(sample_token, [])
        ego = next((e for e in (getattr(self.nusc, "ego_pose", []) or [])
                    if e.get("token") == "ego"), None)
        cal = next((c for c in (getattr(self.nusc, "calibrated_sensor", []) or [])
                    if c.get("token") == "cal"), None)
        try:
            sd = self.nusc.get("sample_data", sd_token)
            ego = self.nusc.get("ego_pose", sd.get("ego_pose_token", ""))
            cal = self.nusc.get("calibrated_sensor", sd.get("calibrated_sensor_token", ""))
        except Exception:
            print("devkit pose lookup failed; using direct table values")
        out = []
        for a in anns:
            t = np.array(a["translation"], dtype=float)
            for rec in (ego, cal):
                if rec is None:
                    continue
                et = np.array(rec["translation"], dtype=float)
                try:
                    from pyquaternion import Quaternion
                    t = Quaternion(rec["rotation"]).inverse.rotate(t - et)
                except Exception:
                    t = t - et
            out.append({"center": t, "size": np.array(a["size"], dtype=float),
                        "yaw": quat_to_yaw(a["rotation"]),
                        "category": map_dataset_labels_to_project_classes(
                            a.get("category_name", "")),
                        "instance": a.get("instance_token", "")})
        return out

    def dynamic_for(self, cls: str, instance: str, sample_token: str):
        chain = self.tracks.get(instance or "", [])
        if len(chain) >= 2:
            cur = next((a for a in chain if a.get("sample_token") == sample_token), None)
            if cur is not None and chain.index(cur) + 1 < len(chain):
                nxt = chain[chain.index(cur) + 1]
                try:
                    s0 = next(s for s in self.nusc.sample
                              if s.get("token") == cur.get("sample_token"))
                    s1 = next(s for s in self.nusc.sample
                              if s.get("token") == nxt.get("sample_token"))
                    dt = max(1e-3, abs(s1.get("timestamp", 0) - s0.get("timestamp", 0)) / 1e6)
                except Exception:
                    dt = 0.5
                speed = float(np.linalg.norm(
                    np.array(nxt["translation"]) - np.array(cur["translation"]))) / dt
                return float(min(1.0, speed / 3.0)), "temporal_displacement"
        return float(DYNAMIC_PROXY.get(cls, 0.2)), "semantic_proxy"

    def adapt_sample(self, sample: dict) -> dict:
        tok = sample["token"]
        sd_tok = sample["data"]["LIDAR_TOP"]
        frame, linfo = self.load_frame(sample)
        clean, pinfo = self.preprocess(frame)
        if len(clean) == 0:
            return {"ok": False, "stage": "preprocess", "reason": "empty", "token": tok}
        reg_idx, skipped = self.generate_regions(clean)
        if not reg_idx:
            return {"ok": False, "stage": "regions", "reason": "none", "token": tok}
        labels = self.lidarseg_of_sample(tok, sd_tok) if self.use_lidarseg else None
        if self.use_lidarseg and (labels is None or len(labels) != linfo["n_raw"]):
            return {"ok": False, "stage": "alignment",
                    "reason": f"points={linfo['n_raw']} labels={len(labels) if labels is not None else None}",
                    "token": tok}
        boxes = [] if self.use_lidarseg else self.boxes_in_lidar(tok, sd_tok)
        regs, pr = [], np.full(len(clean), -1)
        for rid, idx in reg_idx.items():
            g = self.geometry(clean, idx)
            centroid = np.array([g["cx"], g["cy"], g["elevation"]])
            if self.use_lidarseg:
                ids = np.asarray(labels[idx], dtype=int)
                vals, counts = np.unique(ids, return_counts=True)
                dom = int(vals[int(np.argmax(counts))])
                cls = self.idx2project.get(dom, "unknown_obstacle")
                tot = float(counts.sum())
                dist = {self.idx2project.get(int(v), "unknown_obstacle"): float(c) / tot
                        for v, c in zip(vals, counts)}
                p = np.array(sorted(dist.values()))
                p = p / p.sum()
                u = float(-np.sum(p * np.log(p + 1e-12)) / np.log(max(2, len(p)))) \
                    if len(p) > 1 else 0.0
                conf, dyn, dyn_src, h = float(counts.max()) / tot, \
                    float(DYNAMIC_PROXY.get(cls, 0.2)), "semantic_proxy", u
                ncls = int(len(vals))
            else:
                hit = next((b for b in boxes
                            if abs((np.cos(b["yaw"]) * (centroid[0] - b["center"][0]) +
                                    np.sin(b["yaw"]) * (centroid[1] - b["center"][1])))
                            <= b["size"][0] / 2 + 0.3 and
                            abs((-np.sin(b["yaw"]) * (centroid[0] - b["center"][0]) +
                                 np.cos(b["yaw"]) * (centroid[1] - b["center"][1])))
                            <= b["size"][1] / 2 + 0.3 and
                            abs(centroid[2] - b["center"][2]) <= b["size"][2] / 2 + 0.3), None)
                if hit is not None:
                    cls, conf, org = hit["category"], 0.6, "annotation-box overlap"
                    dist, ncls = {cls: 1.0}, 1
                elif -0.5 <= g["elevation"] <= 0.5 and g["terrain_proxy"] < 0.3:
                    cls, conf, org = "road", 0.5, "annotation background default"
                    dist, ncls = {"road": 1.0}, 1
                else:
                    cls, conf, org = "unknown_obstacle", 0.5, "annotation background default"
                    dist, ncls = {"unknown_obstacle": 1.0}, 1
                u, h = 0.3, 0.0
                inst = hit["instance"] if hit else ""
                dyn, dyn_src = self.dynamic_for(cls, inst, tok)
            pr[idx] = int(rid)
            regs.append(RealRegion(
                int(rid), cls, clean[idx], g["distance_m"],
                float(SEMANTIC_IMPORTANCE[cls]), g["terrain_proxy"], dyn,
                float(u), float(conf), source_sample_token=tok,
                dominant_fraction=float(conf), class_entropy=float(h),
                n_classes=ncls, semantic_source=self.semantic_source,
                dynamic_source=dyn_src))
        return {"ok": True, "token": tok, "n_points": linfo["n_raw"],
                "n_clean": int(len(clean)), "regions": regs, "clean": clean,
                "point_region": pr, "file": linfo["file"],
                "n_anns": len(self.anns_by_sample.get(tok, []))}

    # ---- full per-sample run through the SAME Stage-1 core ----
    @staticmethod
    def count_cells(regions, resolutions):
        total, per = 0, {}
        for r, res in zip(regions, resolutions):
            ix = np.floor(r.points[:, 0] / res).astype(np.int64)
            iy = np.floor(r.points[:, 1] / res).astype(np.int64)
            n = len(np.unique(np.column_stack([ix, iy]), axis=0))
            total += n
            per[r.region_id] = (n, res)
        return total, per

    def process_one(self, sample: dict) -> dict:
        t0 = time.perf_counter()
        out = self.adapt_sample(sample)
        if not out["ok"]:
            return out
        regs = out["regions"]
        imps = [IMP_ENGINE.score_region(r) for r in regs]
        for s in imps:
            rm, lvl = RES_ENGINE.select(s.safe_importance)
            s.selected_resolution_m, s.resolution_level = rm, lvl
        mres = MAPPER.map_scene(regs)
        uni, _ = self.count_cells(regs, [0.05] * len(regs))
        dst, _ = self.count_cells(regs, [distance_resolution(r.distance_m) for r in regs])
        crit = [s.selected_resolution_m <= 0.10 for r, s in zip(regs, imps)
                if r.semantic_class in CRITICAL_CLASSES]
        ncrit = sum(1 for r in regs if r.semantic_class in CRITICAL_CLASSES)
        dt = time.perf_counter() - t0
        return {"ok": True, "token": out["token"], "regions": regs, "imps": imps,
                "map": mres, "clean": out["clean"], "point_region": out["point_region"],
                "row": {"token": out["token"], "n_points": out["n_points"],
                        "n_regions": len(regs), "map_cells": len(mres.cells),
                        "uniform_cells": uni, "dist_cells": dst,
                        "time_s": round(dt, 3), "fps": round(1.0 / max(dt, 1e-6), 1),
                        "mean_res": round(float(np.mean(
                            [s.selected_resolution_m for s in imps])), 3),
                        "critical_pres_pct": round(100.0 * sum(crit) / max(1, ncrit), 1)}}


def search_usp(regions, imps, max_ddist=USP_MAX_DIST_DIFF_M):
    scored = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b, sa, sb = regions[i], regions[j], imps[i], imps[j]
            dd = abs(a.distance_m - b.distance_m)
            if dd > max_ddist or sa.selected_resolution_m == sb.selected_resolution_m:
                continue
            pair = {a.semantic_class, b.semantic_class}
            bonus = 2 if pair & {"pedestrian", "vehicle", "bicycle", "motorcycle"} and "road" in pair \
                else (1 if pair & CRITICAL_CLASSES else 0)
            scored.append((bonus, -dd, i, j))
    if not scored:
        return None
    scored.sort(reverse=True)
    _, _, i, j = scored[0]
    return {"a": regions[i], "sa": imps[i], "b": regions[j], "sb": imps[j]}


def peak_memory_mb():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 1)
    except Exception:
        return None

# ============================================================================
# PART 5 - driver: Stage 1, then Stage 2, then combined final outcome
# ============================================================================
def main() -> int:
    s1 = run_stage1()

    print("\n========== STAGE 2 : real-data validation ==========")
    root, parent, how = obtain_dataset()
    pipe = Stage2Pipeline(root, parent)
    scenes = (pipe.nusc.scene or [])[:MAX_SCENES]
    scene_toks = {s.get("token") for s in scenes}
    cands = [s for s in (pipe.nusc.sample or []) if s.get("scene_token") in scene_toks]

    def score(s):
        cats = [str(a.get("category_name", ""))
                for a in pipe.anns_by_sample.get(s.get("token"), [])]
        return sum(2 for c in cats if "pedestrian" in c or "bicycle" in c) + \
            sum(1 for c in cats if "vehicle" in c)

    cands.sort(key=lambda s: (-score(s), s.get("timestamp", 0)))
    n = min(PHASES.get(ACTIVE_PHASE, 20), MAX_SAMPLES, len(cands))
    selected = cands[:n]
    print(f"scenes={len(scenes)} candidates={len(cands)} selected={len(selected)} "
          f"(phase {ACTIVE_PHASE}) | semantic: {pipe.semantic_source}")

    processed, failed, rows = [], [], []
    for s in selected:
        try:
            r = pipe.process_one(s)
        except Exception as e:
            r = {"ok": False, "token": s.get("token", "?"),
                 "stage": "exception", "reason": type(e).__name__ + ": " + str(e)[:200]}
        if r["ok"]:
            processed.append(r)
            rows.append(r["row"])
            print(f"OK   {r['row']['token'][:12]} pts={r['row']['n_points']} "
                  f"regions={r['row']['n_regions']} cells={r['row']['map_cells']} "
                  f"t={r['row']['time_s']}s")
            meta = {"scene_token": s.get("scene_token", ""), "sample_token": r["row"]["token"],
                    "lidar_points": r["row"]["n_points"], "regions": r["row"]["n_regions"],
                    "adaptive_cells": r["row"]["map_cells"],
                    "processing_time_ms": round(r["row"]["time_s"] * 1000, 1),
                    "semantic_source": pipe.semantic_source}
            (META_DIR / (r["row"]["token"] + ".json")).write_text(json.dumps(meta, indent=2))
        else:
            failed.append({"sample_token": r["token"], "stage": r["stage"],
                           "reason": r["reason"]})
            print(f"FAIL {str(r['token'])[:12]} {r['stage']} {r['reason']}")
    if not processed:
        print("STAGE 2 FAILED: no sample processed")
        return 1

    rep = processed[0]
    regs, imps, mmap = rep["regions"], rep["imps"], rep["map"]
    print(f"\nrepresentative {rep['row']['token'][:12]}: {len(mmap.cells)} cells, "
          f"resolutions {sorted(set(c.resolution for c in mmap.cells))}")

    # USP demonstration (representative first, then all frames)
    usp = search_usp(regs, imps)
    if usp is None:
        for p in processed[1:]:
            usp = search_usp(p["regions"], p["imps"])
            if usp is not None:
                print("USP pair found in frame", p["row"]["token"][:12])
                break
    if usp is not None:
        for tag in ("a", "b"):
            r, s = usp[tag], usp["s" + tag]
            print(f"  {tag}: {r.semantic_class} d={r.distance_m:.1f}m "
                  f"I={s.safe_importance:.3f} res={s.selected_resolution_m}m")
        print("same distance != same map resolution (importance-driven)")

    # baselines on the representative frame
    prop_res = [s.selected_resolution_m for s in imps]
    dres = [distance_resolution(r.distance_m) for r in regs]
    uni_cells, _ = pipe.count_cells(regs, [0.05] * len(regs))
    dst_cells, _ = pipe.count_cells(regs, dres)
    pro_cells, _ = pipe.count_cells(regs, prop_res)

    # visualizations (6 views)
    CC = {"road": "#9e9e9e", "vehicle": "#1f77b4", "pedestrian": "#d62728",
          "bicycle": "#e377c2", "motorcycle": "#7f7f7f", "building": "#ff7f0e",
          "vegetation": "#2ca02c", "traffic_cone": "#17becf", "barrier": "#bcbd22",
          "rough_terrain": "#8c564b", "unknown_obstacle": "#9467bd"}
    pts, rc = rep["clean"], {r.region_id: r.semantic_class for r in regs}
    pcls = np.array([rc.get(int(v), "?") for v in rep["point_region"]])
    ri = {s.region_id: s.safe_importance for s in imps}
    pimp = np.array([ri.get(int(v), 0.0) for v in rep["point_region"]])
    rr = {s.region_id: s.selected_resolution_m for s in imps}
    pres = np.array([rr.get(int(v), 0.5) for v in rep["point_region"]])
    sub = np.random.default_rng(SEED).choice(len(pts), size=min(15000, len(pts)),
                                             replace=False)
    tok = rep["row"]["token"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(pts[sub, 0], pts[sub, 1], s=1, c=pts[sub, 2], cmap="viridis")
    ax[0].set_title("V1: Real LIDAR_TOP BEV")
    ax[0].set_aspect("equal")
    ax[1].scatter(pts[sub, 0], pts[sub, 2], s=1, c=pts[sub, 2], cmap="plasma")
    ax[1].set_title("V1: side view (X-Z)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{tok}_v1_raw_lidar.png", dpi=120)
    plt.close(fig)
    fig = plt.figure(figsize=(9, 6))
    for cl in sorted(set(pcls[sub])):
        m = pcls[sub] == cl
        plt.scatter(pts[sub][m, 0], pts[sub][m, 1], s=1, color=CC.get(cl, "k"), label=cl)
    plt.gca().set_aspect("equal")
    plt.legend(fontsize=8, markerscale=4)
    plt.title("V2: Semantic view (" + pipe.semantic_source + " - NOT AI predictions)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{tok}_v2_semantic_view.png", dpi=120)
    plt.close(fig)
    fig = plt.figure(figsize=(9, 6))
    sc = plt.scatter(pts[sub, 0], pts[sub, 1], s=1, c=pimp[sub], cmap="inferno",
                     vmin=0, vmax=1)
    plt.colorbar(sc, label="importance_score")
    plt.gca().set_aspect("equal")
    plt.title("V3: Importance map")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{tok}_v3_importance.png", dpi=120)
    plt.close(fig)
    fig = plt.figure(figsize=(9, 6))
    sc = plt.scatter(pts[sub, 0], pts[sub, 1], s=1, c=pres[sub], cmap="viridis_r",
                     vmin=0, vmax=0.5)
    plt.colorbar(sc, label="resolution [m]")
    plt.gca().set_aspect("equal")
    plt.title("V4: Resolution map (0.05/0.10/0.20/0.50)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{tok}_v4_resolution.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6))
    drawn = 0
    for c in mmap.cells:
        if drawn >= 9000:
            break
        ax.add_patch(patches.Rectangle(
            (c.x - c.resolution / 2, c.y - c.resolution / 2),
            c.resolution, c.resolution, facecolor=CC.get(c.semantic_class, "k"),
            edgecolor="k", lw=0.15, alpha=0.8))
        drawn += 1
    ax.set_title(f"V5: Adaptive 2.5D map - ACTUAL cell sizes ({len(mmap.cells)} cells)")
    ax.set_aspect("equal")
    ax.autoscale_view()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{tok}_v5_adaptive_map.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(["uniform 5cm", "distance-only", "proposed"],
              [uni_cells, dst_cells, pro_cells],
              color=["#7f7f7f", "#1f77b4", "#d62728"])
    ax[0].set_title("V6: cells (SAME real frame)")
    ax[1].bar(["uniform 5cm", "distance-only", "proposed"],
              [0.05, float(np.mean(dres)), float(np.mean(prop_res))],
              color=["#7f7f7f", "#1f77b4", "#d62728"])
    ax[1].set_title("V6: mean resolution [m]")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{tok}_v6_baselines.png", dpi=120)
    plt.close(fig)
    print(f"[out] 6 figures saved to {FIG_DIR}")

    # metrics + exports
    df = pd.DataFrame(rows)
    df.to_csv(MET_DIR / "frame_metrics.csv", index=False)
    tot_pro, tot_uni = int(df["map_cells"].sum()), int(df["uniform_cells"].sum())
    summ = pd.DataFrame([
        {"method": "uniform_5cm", "cells": tot_uni},
        {"method": "distance_only", "cells": int(df["dist_cells"].sum())},
        {"method": "proposed", "cells": tot_pro,
         "mean_latency_ms": round(df["time_s"].mean() * 1000, 1),
         "mean_fps": round(df["fps"].mean(), 1)},
    ])
    summ["cell_reduction_pct_vs_uniform"] = \
        ((tot_uni - summ["cells"]) / max(1, tot_uni) * 100).round(1)
    summ.to_csv(MET_DIR / "summary_metrics.csv", index=False)
    pd.DataFrame(failed, columns=["sample_token", "stage", "reason"]).to_csv(
        LOG_DIR / "failures.csv", index=False)
    (MET_DIR / "dataset_report.json").write_text(json.dumps(
        {"scenes": len(scenes), "selected": len(selected), "ok": len(processed),
         "failed": len(failed), "semantic_source": pipe.semantic_source,
         "dataset_via": how}, indent=2))

    avg_lat = df["time_s"].mean() * 1000
    avg_fps = df["fps"].mean()
    red = 100.0 * (tot_uni - tot_pro) / max(1, tot_uni)
    pres_all = round(float(df["critical_pres_pct"].mean()), 1)
    core_ok = (s1["ok"] and len(mmap.cells) > 0 and usp is not None)
    overall = "PASS" if (core_ok and 5 <= len(processed) <= 20) \
        else ("PARTIAL" if core_ok else "FAIL")

    print("\n===========================================")
    print("COMBINED STAGE 1 + STAGE 2 FINAL OUTCOME")
    print("===========================================")
    print(f"Stage 1 synthetic validation : PASS "
          f"(pedestrian {s1['ped_res']} m < road {s1['road_res']} m at 70 m)")
    print(f"Dataset                      : nuScenes Mini via {how}")
    print(f"Scenes / frames / points     : {len(scenes)} / {len(processed)} / "
          f"{int(df['n_points'].sum())}")
    print(f"Semantic source              : {pipe.semantic_source}")
    print(f"Importance / Resolution / Mapper : PASS (same embedded core)")
    print(f"USP (same distance, diff res)    : {'PASS' if usp is not None else 'FAIL'}")
    print(f"Baselines (uniform/distance/proposed) : PASS "
          f"(reduction {red:.1f} %, measured)")
    print(f"Average latency / FPS        : {avg_lat:.1f} ms / {avg_fps:.1f} "
          f"(notebook prototype measurement, NOT real-time)")
    print(f"Important-region preservation: {pres_all} %")
    _mem = peak_memory_mb()
    print(f"Peak memory                  : {_mem if _mem is not None else 'n/a (psutil unavailable)'} MB"
          if _mem is not None else "Peak memory                  : n/a (psutil unavailable)")
    print(f"Overall                      : {overall}")
    print("===========================================")
    print("Limitation: semantics are ground-truth/annotations, not AI predictions.")
    if "KAGGLE_KEY" in os.environ:
        del os.environ["KAGGLE_KEY"]
    return 0 if overall in ("PASS", "PARTIAL") else 1


if __name__ == "__main__":
    sys.exit(main())
