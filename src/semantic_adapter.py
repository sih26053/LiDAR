"""Semantic + geometric integration adapter (11 September 2026 task).

ADDITIVE module. Consumes processed N x 4 LiDAR + a semantic source
(lidarseg / annotation / model) and produces:

    PerceptionResult (point-level, existing 9-September class)
    RegionFeatures   (region-level, existing 9-September class)

which enter the UNCHANGED Importance Engine v1 + Resolution Engine v1.

Provenance (every output carries it):
- semantic_source: "lidarseg" | "annotation" | "model" | "fallback"
- confidence: actual model confidence ONLY for source == "model";
  NaN (= not applicable) for lidarseg / annotation / fallback at the
  point level. RegionFeatures.confidence uses the numeric 0.0 interface
  placeholder for non-ML regions (documented, never presented as ML
  confidence) because the existing engine contract requires [0,1].
- dominant_class_ratio: label-consistency measure, NOT ML confidence.
- terrain_complexity (RegionFeatures.roughness): geometric
  roughness-derived proxy, NOT a trained terrain classifier.
- dynamic_relevance: PROJECT_DYNAMIC_PRIOR heuristic unless genuine
  motion evidence is supplied; never measured velocity.
- uncertainty: documented fallback (0.5) unless a real estimator exists.

Grid: 2.0 m XY integration grid (same convention as the 10-September
first handoff), NOT the final adaptive mapping representation.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from .data_types import PerceptionResult, RegionFeatures
from .interface_validator import (
    validate_perception_result,
    validate_region_features,
)
from .semantic_mapping import (
    DEFAULT_UNCERTAINTY_FALLBACK,
    PROJECT_DYNAMIC_PRIOR,
    PROJECT_SEMANTIC_IMPORTANCE,
    VALID_SEMANTIC_SOURCES,
    map_nuscenes_label_to_project,
)

CELL_SIZE_M = 2.0
MIN_HEIGHT_VARIATION = 1e-6

# Region-level numeric placeholder when no ML confidence exists.
# Interface-compatible ([0,1]) but documented as "no ML confidence".
REGION_CONFIDENCE_NON_ML = 0.0


# ---------------------------------------------------------------------------
# Geometry (spec Sections 19/21)
# ---------------------------------------------------------------------------

def compute_point_geometry(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point distance (3D Euclidean) and elevation (= z).

    Region-level distance separately uses the planar centroid range
    hypot(cx, cy) per the region contract (Section 21).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError(f"points must be N x 4. Received shape {pts.shape!r}")
    if pts.shape[0] == 0:
        raise ValueError("points must contain at least 1 point.")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must be all finite (NaN/inf not allowed).")
    distance = np.linalg.norm(pts[:, :3], axis=1).astype(np.float64)
    elevation = pts[:, 2].astype(np.float64)
    return distance, elevation


# ---------------------------------------------------------------------------
# Annotation fallback: oriented-box point assignment (spec Section 15)
# ---------------------------------------------------------------------------

def _yaw_from_quaternion(rotation: Sequence[float]) -> float:
    """Extract yaw (rad) from a nuScenes (w, x, y, z) quaternion.

    nuScenes boxes are upright (rotation about z only in practice); this
    extracts the z-yaw. Falls back to 0.0 for malformed input (documented,
    box then treated as axis-aligned).
    """
    try:
        w, x, y, z = (float(v) for v in list(rotation)[:4])
    except (TypeError, ValueError):
        return 0.0
    norm = float(np.sqrt(w * w + x * x + y * y + z * z))
    if norm <= 0 or not np.isfinite(norm):
        return 0.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _resolve_lidar_sd_token(nusc, sample, sensor: str = "LIDAR_TOP") -> str:
    """Resolve the LiDAR sample_data token for a sample (full + mini schema).

    Full nuScenes samples carry ``sample["data"][sensor]``. The v1.0-mini
    mirror ships ``sample.json`` records with only
    ``token/timestamp/prev/next/scene_token`` (no ``data`` key), so fall
    back to matching ``sample_data.sample_token == sample["token"]`` with a
    ``channel == sensor`` (or ``sensor`` in filename) filter.
    """
    data = sample.get("data") if isinstance(sample, dict) else None
    if isinstance(data, dict) and data.get(sensor):
        return str(data[sensor])
    candidates = []
    for rec in (getattr(nusc, "sample_data", None) or []):
        if not isinstance(rec, dict):
            continue
        if rec.get("sample_token") != sample.get("token"):
            continue
        if rec.get("channel") == sensor or sensor in str(rec.get("filename", "")):
            candidates.append(rec)
    if not candidates:  # last resort: any record for this sample
        for rec in (getattr(nusc, "sample_data", None) or []):
            if isinstance(rec, dict) and rec.get("sample_token") == sample.get("token"):
                candidates.append(rec)
    if not candidates:
        raise KeyError(
            f"cannot resolve {sensor} sample_data for sample {sample.get('token')!r} "
            "(no sample['data'] link and no sample_token match in sample_data table)."
        )
    # Prefer keyframes / exact channel matches first.
    candidates.sort(
        key=lambda r: (
            0 if r.get("channel") == sensor else 1,
            0 if r.get("is_key_frame") else 1,
        )
    )
    return str(candidates[0]["token"])


def _resolve_annotation_tokens(nusc, sample) -> List[str]:
    """Resolve annotation tokens (full + mini schema).

    Full schema: ``sample["anns"]``. Mini schema: no ``anns`` key, so
    collect ``sample_annotation.sample_token == sample["token"]`` matches.
    """
    if isinstance(sample.get("anns"), list):
        return [str(t) for t in sample["anns"]]
    tokens = [
        str(rec["token"])
        for rec in (getattr(nusc, "sample_annotation", None) or [])
        if isinstance(rec, dict) and rec.get("sample_token") == sample.get("token")
    ]
    return tokens


def _category_name_for_annotation(nusc, rec: Dict[str, Any]) -> str:
    """Return the nuScenes category name for an annotation record.

    Full schema carries ``rec["category_name"]``. Mini ``sample_annotation``
    records carry only ``instance_token``; resolve via
    ``instance -> category`` tables (unknown when unresolvable, never
    invented).
    """
    if rec.get("category_name"):
        return str(rec["category_name"])
    try:
        inst = nusc.get("instance", rec.get("instance_token", ""))
        cat = nusc.get("category", (inst or {}).get("category_token", ""))
        name = (cat or {}).get("name", "")
        return str(name or "unknown")
    except Exception:
        return "unknown"


def annotations_to_sensor_frame(nusc, sample, sensor: str = "LIDAR_TOP") -> List[Dict[str, Any]]:
    """Map sample annotations from the global frame to a LiDAR sensor frame.

    ``sample_annotation`` records live in the global/map frame while the
    processed points are in the sensor frame; comparing them directly yields
    zero overlap (real-data bug caught 16 Sept 2026). This applies the
    devkit global -> ego -> sensor chain (ego_pose + calibrated_sensor of
    the sample_data record) and returns annotation dicts ready for
    :func:`assign_annotation_semantics` (translation in sensor frame,
    rotation as a (w, x, y, z) quaternion in sensor frame, size unchanged).
    Requires ``pyquaternion`` (a nuScenes-devkit dependency).

    Handles both the full nuScenes schema (``sample["data"]`` /
    ``sample["anns"]`` / ``rec["category_name"]``) and the v1.0-mini mirror
    schema (token-only sample records, category via instance join).
    """
    from pyquaternion import Quaternion

    sd = nusc.get("sample_data", _resolve_lidar_sd_token(nusc, sample, sensor))
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    ego_t = np.asarray(ego["translation"], dtype=np.float64)
    ego_q = Quaternion(ego["rotation"])
    cs_t = np.asarray(cs["translation"], dtype=np.float64)
    cs_q = Quaternion(cs["rotation"])
    ego_q_inv = ego_q.inverse
    cs_q_inv = cs_q.inverse

    out: List[Dict[str, Any]] = []
    for tok in _resolve_annotation_tokens(nusc, sample):
        rec = nusc.get("sample_annotation", tok)
        c = np.asarray(rec["translation"], dtype=np.float64) - ego_t
        c = np.asarray(ego_q_inv.rotate(c), dtype=np.float64) - cs_t
        c = np.asarray(cs_q_inv.rotate(c), dtype=np.float64)
        q = cs_q_inv * ego_q_inv * Quaternion(rec["rotation"])
        out.append({
            "category_name": _category_name_for_annotation(nusc, rec),
            "translation": [float(v) for v in c.tolist()],
            "size": [float(v) for v in list(rec["size"])],
            "rotation": [float(q.w), float(q.x), float(q.y), float(q.z)],
            "instance_token": str(rec.get("instance_token", "")),
        })
    return out


def points_in_box(
    points_xyz: np.ndarray,
    translation: Sequence[float],
    size: Sequence[float],
    rotation: Sequence[float] | float | None,
) -> np.ndarray:
    """Boolean mask of points inside one 3D box (box frame, inclusive).

    size = (width, length, height) per nuScenes convention; in the box
    local frame the x extent is length/2 and the y extent is width/2
    (matches devkit ``Box.corners``). rotation may be a quaternion
    (w,x,y,z), a yaw float, or None (= 0 yaw).
    """
    xyz = np.asarray(points_xyz, dtype=np.float64)
    cx, cy, cz = (float(v) for v in list(translation)[:3])
    w, l, h = (float(v) for v in list(size)[:3])
    if isinstance(rotation, (int, float)):
        yaw = float(rotation)
    elif rotation is None:
        yaw = 0.0
    else:
        try:
            seq = list(rotation)
            yaw = float(seq[0]) if len(seq) == 1 else _yaw_from_quaternion(seq)
        except (TypeError, ValueError):
            yaw = 0.0
    dx = xyz[:, 0] - cx
    dy = xyz[:, 1] - cy
    dz = xyz[:, 2] - cz
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    # Rotate points into the box frame (inverse yaw rotation).
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return (
        (np.abs(lx) <= l / 2.0)
        & (np.abs(ly) <= w / 2.0)
        & (np.abs(dz) <= h / 2.0)
    )


def assign_annotation_semantics(
    points: np.ndarray,
    annotations: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign per-point project labels from nuScenes object annotations.

    Honest fallback: points inside a 3D box get the box's mapped class with
    source "annotation"; ALL other points get "unknown" with source
    "fallback". Larger boxes are applied first so smaller (usually more
    specific) boxes win overlaps. No point-level segmentation is faked and
    no ML confidence is invented (confidence = NaN for every point here;
    the caller keeps the distinction between "annotation" and "fallback"
    via the returned source array).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    # NOTE: dtype=object (not dtype=str) so later assignment of longer
    # strings such as "annotation" is never truncated (numpy fixed-width
    # '<U8' from "fallback" would otherwise clip to "annotati").
    labels = np.array(["unknown"] * n, dtype=object)
    sources = np.array(["fallback"] * n, dtype=object)
    originals = np.array([""] * n, dtype=object)

    def _box_volume(ann: Dict[str, Any]) -> float:
        try:
            w, l, h = (float(v) for v in list(ann["size"])[:3])
            return float(w * l * h)
        except (KeyError, TypeError, ValueError):
            return float("inf")

    ordered = sorted(list(annotations or []), key=_box_volume, reverse=True)
    for ann in ordered:
        try:
            cat = str(ann.get("category_name", ""))
            trans = ann["translation"]
            size = ann["size"]
        except KeyError:
            continue
        rot = ann.get("rotation", None)
        try:
            mask = points_in_box(pts[:, :3], trans, size, rot)
        except (TypeError, ValueError):
            continue
        project_label = map_nuscenes_label_to_project(cat)
        labels[mask] = project_label
        sources[mask] = "annotation"
        originals[mask] = cat
    confidence = np.full(n, np.nan, dtype=np.float64)  # not applicable
    return labels, sources, confidence


def build_perception_from_semantics(
    frame_id: str,
    points: np.ndarray,
    semantic_labels: Sequence[str],
    semantic_source: Sequence[str] | str,
    confidence: Sequence[float] | np.ndarray | None = None,
    original_labels: Sequence[str] | None = None,
) -> Tuple[PerceptionResult, np.ndarray, np.ndarray]:
    """Build PerceptionResult + parallel provenance arrays.

    - semantic_labels: project-class strings (N,).
    - semantic_source: single source str or per-point array (N,).
    - confidence: REQUIRED for source "model" (finite [0,1]); must be
      None/NaN for lidarseg/annotation/fallback (never fabricated).
    - Returns (perception_result, source_array, original_array).

    Point-level roughness = zeros (region step computes real roughness);
    point_density = ones placeholder, overwritten by region broadcast in
    aggregate_to_regions (same precedent as build_perception_from_labeled_points).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError(f"points must be N x 4. Received shape {pts.shape!r}")
    n = pts.shape[0]
    # dtype=object throughout: avoids numpy fixed-width truncation of
    # variable-length label/source strings (see note above).
    labels = np.asarray([str(s) for s in list(semantic_labels)], dtype=object)
    if labels.shape[0] != n:
        raise ValueError(f"len(semantic_labels)={labels.shape[0]} != N={n}")
    if isinstance(semantic_source, str):
        sources = np.array([str(semantic_source)] * n, dtype=object)
    else:
        sources = np.asarray([str(s) for s in list(semantic_source)], dtype=object)
    if sources.shape[0] != n:
        raise ValueError(f"len(semantic_source)={sources.shape[0]} != N={n}")
    for s in sources.tolist():
        if s not in VALID_SEMANTIC_SOURCES:
            raise ValueError(
                f"semantic_source must be one of {sorted(VALID_SEMANTIC_SOURCES)}. "
                f"Received: {s!r}"
            )
    if original_labels is None:
        originals = np.array([""] * n, dtype=object)
    else:
        originals = np.asarray([str(s) for s in list(original_labels)], dtype=object)

    unique_sources = set(sources.tolist())
    if unique_sources == {"model"} or "model" in unique_sources:
        if confidence is None:
            raise ValueError(
                'confidence array is required when semantic_source includes "model".'
            )
        conf = np.asarray(list(confidence), dtype=np.float64)
        if conf.shape[0] != n:
            raise ValueError(f"len(confidence)={conf.shape[0]} != N={n}")
        model_mask = sources == "model"
        if not np.all(np.isfinite(conf[model_mask])):
            raise ValueError('Model points must carry finite confidence in [0,1].')
        if np.any((conf[model_mask] < 0.0) | (conf[model_mask] > 1.0)):
            raise ValueError('Model confidence must be within [0,1].')
        non_model = ~model_mask
        if np.any(np.isfinite(conf[non_model])):
            raise ValueError(
                "Non-model points must use NaN (not applicable) confidence; "
                "do not fabricate ML confidence for lidarseg/annotation/fallback."
            )
    else:
        if confidence is not None:
            conf = np.asarray(list(confidence), dtype=np.float64)
            if np.any(np.isfinite(conf)):
                raise ValueError(
                    "Non-model sources must use NaN (not applicable) confidence; "
                    "received finite values."
                )
        conf = np.full(n, np.nan, dtype=np.float64)

    distance, elevation = compute_point_geometry(pts)
    roughness = np.zeros(n, dtype=np.float64)
    density = np.ones(n, dtype=np.float64)
    pr = PerceptionResult(
        frame_id=str(frame_id),
        points=pts,
        semantic_labels=labels,
        confidence=conf,
        distance=distance,
        elevation=elevation,
        roughness=roughness,
        point_density=density,
    )
    validate_perception_result(pr)  # allows NaN confidence (documented N/A)
    return pr, sources, originals


def aggregate_to_regions(
    pr: PerceptionResult,
    semantic_source: np.ndarray,
    cell_size: float = CELL_SIZE_M,
    semantic_importance_map: Dict[str, float] | None = None,
    dynamic_map: Dict[str, float] | None = None,
    default_uncertainty: float = DEFAULT_UNCERTAINTY_FALLBACK,
) -> Tuple[List[RegionFeatures], List[Dict[str, Any]]]:
    """Aggregate point-level semantics + geometry into RegionFeatures.

    Grid: floor(x / cell) x floor(y / cell) (2.0 m default).
    Per region: centroid, planar distance hypot(cx,cy), elevation mean(z),
    roughness std(z) normalized by max over frame (proxy, clipped [0,1]),
    density count / cell^2, dominant project class by majority vote,
    dominant_class_ratio (consistency measure, NOT confidence),
    source = majority source among points voting for the dominant class,
    RegionFeatures.confidence = 0.0 placeholder for non-ML regions
    (mean model confidence when the dominant source is "model"),
    dynamic = PROJECT_DYNAMIC_PRIOR heuristic, uncertainty = 0.5 fallback
    (1 - confidence proxy only when dominant source is "model").

    Returns (region_features, region_detail_rows) where each detail row
    carries the extra audit fields (semantic_source, dominant_class_ratio,
    uncertainty_source, dynamic_source) that do not fit the fixed
    RegionFeatures contract.
    """
    validate_perception_result(pr)
    if not isinstance(cell_size, (int, float)) or not np.isfinite(float(cell_size)):
        raise ValueError(f"cell_size must be finite. Received: {cell_size!r}")
    if float(cell_size) <= 0:
        raise ValueError(f"cell_size must be > 0. Received: {cell_size!r}")
    cell = float(cell_size)
    sem_map = dict(semantic_importance_map or PROJECT_SEMANTIC_IMPORTANCE)
    dyn_map = dict(dynamic_map or PROJECT_DYNAMIC_PRIOR)

    pts = pr.points
    n = pts.shape[0]
    sources = np.asarray([str(s) for s in list(semantic_source)], dtype=object)
    if sources.shape[0] != n:
        raise ValueError(f"len(semantic_source)={sources.shape[0]} != N={n}")

    grid_x = np.floor(pts[:, 0] / cell).astype(np.int64)
    grid_y = np.floor(pts[:, 1] / cell).astype(np.int64)
    unique_regions = np.unique(np.column_stack([grid_x, grid_y]), axis=0)

    records: List[Dict[str, Any]] = []
    for region_id, (gx, gy) in enumerate(unique_regions):
        mask = (grid_x == int(gx)) & (grid_y == int(gy))
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        region_points = pts[idx]
        xyz = region_points[:, :3]
        x_mean = float(xyz[:, 0].mean())
        y_mean = float(xyz[:, 1].mean())
        z_vals = xyz[:, 2]
        elevation = float(z_vals.mean())
        height_std = float(z_vals.std())
        region_distance = float(np.sqrt(x_mean ** 2 + y_mean ** 2))
        point_count = int(len(idx))
        point_density = float(point_count / (cell ** 2))

        region_labels = [str(s) for s in pr.semantic_labels[idx].tolist()]
        region_sources = sources[idx].tolist()
        counts = Counter(region_labels)
        dominant_label, dominant_count = counts.most_common(1)[0]
        dominant_ratio = float(dominant_count / len(idx))
        # Majority source among points voting for the dominant class.
        dom_sources = [
            s for lab, s in zip(region_labels, region_sources) if lab == dominant_label
        ]
        dominant_source = Counter(dom_sources).most_common(1)[0][0]

        if dominant_label not in sem_map:
            dominant_label = "unknown"
        sem_imp = float(sem_map.get(dominant_label, 0.5))

        if dominant_source == "model":
            dom_conf = np.asarray(pr.confidence[idx], dtype=np.float64)
            voter_conf = np.array(
                [c for lab, c in zip(region_labels, dom_conf.tolist())
                 if lab == dominant_label],
                dtype=np.float64,
            )
            finite = voter_conf[np.isfinite(voter_conf)]
            region_conf = float(finite.mean()) if len(finite) else 0.0
            uncertainty = float(max(0.0, min(1.0, 1.0 - region_conf)))
            uncertainty_source = "one_minus_model_confidence (prototype proxy)"
        else:
            region_conf = float(REGION_CONFIDENCE_NON_ML)
            uncertainty = float(default_uncertainty)
            uncertainty_source = "documented fallback 0.5 (no estimator)"

        dynamic = float(dyn_map.get(dominant_label, dyn_map.get("unknown", 0.2)))
        dynamic_source = (
            "measured motion" if False
            else "PROJECT_DYNAMIC_PRIOR heuristic (not measured velocity)"
        )

        records.append({
            "region_id": int(region_id),
            "x": x_mean,
            "y": y_mean,
            "distance": region_distance,
            "elevation": elevation,
            "height_std": height_std,
            "point_density": point_density,
            "point_count": point_count,
            "semantic_label": str(dominant_label),
            "semantic_importance": sem_imp,
            "confidence": region_conf,
            "dynamic_relevance": dynamic,
            "uncertainty": uncertainty,
            "semantic_source": str(dominant_source),
            "dominant_class_ratio": dominant_ratio,
            "uncertainty_source": uncertainty_source,
            "dynamic_source": dynamic_source,
        })

    if not records:
        raise ValueError("No spatial regions generated from the input points.")

    # Terrain-complexity proxy: max-normalized height variation, [0,1].
    max_std = max(
        float(max(r["height_std"] for r in records)),
        float(MIN_HEIGHT_VARIATION),
    )
    region_features: List[RegionFeatures] = []
    detail_rows: List[Dict[str, Any]] = []
    for r in records:
        terrain = float(max(0.0, min(1.0, r["height_std"] / max_std)))
        feature = RegionFeatures(
            region_id=int(r["region_id"]),
            x=float(r["x"]),
            y=float(r["y"]),
            distance=float(r["distance"]),
            elevation=float(r["elevation"]),
            roughness=terrain,
            point_density=float(r["point_density"]),
            semantic_label=str(r["semantic_label"]),
            semantic_importance=float(r["semantic_importance"]),
            confidence=float(r["confidence"]),
            dynamic_relevance=float(r["dynamic_relevance"]),
            uncertainty=float(r["uncertainty"]),
            point_count=int(r["point_count"]),
        )
        validate_region_features(feature)
        region_features.append(feature)
        detail_rows.append({**r, "terrain_complexity": terrain})
    return region_features, detail_rows
