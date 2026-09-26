"""Compatibility entry point for point-classifier training.

The model of record is the MLP point segmenter trained by
``scripts/train_segmentation_mlp.py`` (weights in
``models/segmentation/weights/``). This script delegates to it so older
references keep working. Run::

    python scripts/train_semantic_model.py   # == train_segmentation_mlp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_segmentation_mlp import main  # noqa: E402


def extract_features(points):
    """Legacy import path; single owner is src/semantic_model."""
    from src.semantic_model import extract_features as _extract

    return _extract(points)


if __name__ == "__main__":
    main()
