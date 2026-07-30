"""A_RED label provider that answers from COCO-derived tile GT (no GUI)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .coco_index import CocoBox
from .tile_labeler import BBoxTileLabeler, TileLabelResult


class AutoAnnotationLabelProvider:
    """
    Callable label provider for AREDAdapter.

    Signature matches the interactive path:
        (emb, tile_img, meta) -> (label, relevant)

    Uses bbox intersection GT. Does not open any GUI.
    Optional ``on_label`` callback for side effects (stats, last-query preview).
    """

    def __init__(
        self,
        labeler: BBoxTileLabeler,
        boxes_for_meta: Callable[[Dict[str, Any]], Sequence[CocoBox]],
        on_label: Optional[Callable[[TileLabelResult, Dict[str, Any]], None]] = None,
    ):
        self.labeler = labeler
        self.boxes_for_meta = boxes_for_meta
        self.on_label = on_label

    def __call__(
        self,
        emb: np.ndarray,
        tile_img: Any,
        meta: Dict[str, Any],
    ) -> Tuple[str, bool]:
        meta = meta or {}
        bbox = meta.get("bbox")
        if bbox is None and tile_img is not None and hasattr(tile_img, "bbox"):
            bbox = tile_img.bbox
        if bbox is None:
            # Fallback: no spatial info — treat as water (should not happen in runner)
            result = self.labeler.label_tile((0, 0, 0, 0), [])
        else:
            boxes = self.boxes_for_meta(meta)
            result = self.labeler.label_tile(tuple(bbox), boxes)  # type: ignore[arg-type]
        if self.on_label is not None:
            try:
                self.on_label(result, meta)
            except Exception:
                pass
        return result.label, bool(result.relevant)
