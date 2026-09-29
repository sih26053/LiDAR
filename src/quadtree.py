"""Hierarchical quadtree over adaptive-map regions (flow box 4 detail).

Builds a genuine pointer-based subdivision tree from region centroids:

  root region -> subdivision -> higher-detail child cells

Subdivision rule (computationally practical, no imitation subdivision):

  subdivide a node while depth < max_depth AND mean importance of its
  regions >= split_importance AND node width > min_size_m.

Consequence, matching the reference behavior:

  High importance   -> subdivide more (deeper leaves)
  Medium importance -> moderate subdivision
  Low importance    -> retain larger cell (leaf at shallow depth)

Resolution assignment itself stays in ``src/resolution_engine.py``; each
leaf records the engine-assigned resolution of its regions for audit.
Leaf bands HIGH/MEDIUM/LOW use the caller-supplied thresholds (frozen
runtime config) and are reporting labels only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

HIGH = "HIGH"
MEDIUM = "MEDIUM"
LOW = "LOW"


def band(importance: float, high: float = 0.70, medium: float = 0.45,
         low: float = 0.20) -> str:
    if importance >= high:
        return HIGH
    if importance >= medium:
        return MEDIUM
    return LOW


def build_quadtree(
    regions: Sequence[Dict[str, Any]],
    max_depth: int = 4,
    split_importance: float = 0.20,
    min_size_m: float = 2.0,
    high: float = 0.70,
    medium: float = 0.45,
    low: float = 0.20,
) -> Dict[str, Any]:
    """Region dicts (x, y, importance, resolution) -> quadtree root node."""
    regs = [
        {"x": float(r["x"]), "y": float(r["y"]),
         "importance": float(r["importance"]),
         "resolution": float(r.get("resolution", 0.5))}
        for r in regions
    ]
    if not regs:
        raise ValueError("build_quadtree received no regions.")
    xs = [r["x"] for r in regs]
    ys = [r["y"] for r in regs]
    pad = 1.0
    root = _node(min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad,
                 0, regs, max_depth, split_importance, min_size_m,
                 high, medium, low)
    return root


def _node(x_min, x_max, y_min, y_max, depth, regs, max_depth,
          split_importance, min_size_m, high, medium, low):
    imps = [r["importance"] for r in regs]
    mean_imp = sum(imps) / len(imps)
    node: Dict[str, Any] = {
        "bounds": [x_min, x_max, y_min, y_max],
        "depth": depth,
        "n_regions": len(regs),
        "mean_importance": mean_imp,
        "band": band(mean_imp, high, medium, low),
        "children": None,
    }
    width = max(x_max - x_min, y_max - y_min)
    if depth < max_depth and mean_imp >= split_importance and width > min_size_m and len(regs) > 1:
        mx, my = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        quads = [[] for _ in range(4)]
        for r in regs:
            qi = (0 if r["x"] < mx else 1) + (0 if r["y"] < my else 2)
            quads[qi].append(r)
        bounds = [(x_min, mx, y_min, my), (mx, x_max, y_min, my),
                  (x_min, mx, my, y_max), (mx, x_max, my, y_max)]
        children = []
        for (bx0, bx1, by0, by1), q in zip(bounds, quads):
            if not q:
                continue
            children.append(_node(bx0, bx1, by0, by1, depth + 1, q,
                                  max_depth, split_importance, min_size_m,
                                  high, medium, low))
        if children:
            node["children"] = children
    return node


def stats(root: Dict[str, Any]) -> Dict[str, Any]:
    """Measured tree statistics (node counts, depth, band mix of leaves)."""
    nodes = [root]
    leaves: List[Dict[str, Any]] = []
    max_depth = 0
    while nodes:
        n = nodes.pop()
        max_depth = max(max_depth, int(n["depth"]))
        if n["children"]:
            nodes.extend(n["children"])
        else:
            leaves.append(n)
    total_nodes = 1 + sum(1 for _ in _walk(root))
    band_mix = {HIGH: 0, MEDIUM: 0, LOW: 0}
    for lf in leaves:
        band_mix[lf["band"]] += 1
    depths = [lf["depth"] for lf in leaves]
    return {
        "total_nodes": total_nodes,
        "n_leaves": len(leaves),
        "max_depth": max_depth,
        "mean_leaf_depth": sum(depths) / len(depths) if depths else 0.0,
        "leaf_band_mix": band_mix,
        "mean_leaf_importance": (sum(lf["mean_importance"] for lf in leaves) / len(leaves)) if leaves else 0.0,
    }


def _walk(root):
    stack = list(root.get("children") or [])
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.get("children") or [])
