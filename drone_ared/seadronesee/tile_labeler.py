"""Map tile bboxes to labels via COCO box intersection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from .coco_index import CocoBox
from .config import TileLabelConfig


@dataclass(frozen=True)
class TileLabelResult:
    label: str
    relevant: bool
    overlap_area: float = 0.0
    winning_category_id: Optional[int] = None
    n_overlapping_boxes: int = 0


def _intersection_area(
    tile_xyxy: Tuple[float, float, float, float],
    box: CocoBox,
) -> float:
    tx0, ty0, tx1, ty1 = tile_xyxy
    bx0, by0, bx1, by1 = box.xyxy
    ix0 = max(tx0, bx0)
    iy0 = max(ty0, by0)
    ix1 = min(tx1, bx1)
    iy1 = min(ty1, by1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    return float(iw * ih)


class BBoxTileLabeler:
    """Derive (label, relevant) for a tile from COCO instance boxes."""

    def __init__(self, config: Optional[TileLabelConfig] = None):
        self.config = config or TileLabelConfig()
        self._non_object = {
            str(c).strip().casefold()
            for c in (self.config.non_object_categories or [])
            if c
        }
        if self.config.ignored_as_negative:
            self._non_object.add("ignored")
        rel = self.config.relevant_categories
        self._relevant_filter: Optional[Set[str]] = None
        if rel is not None:
            self._relevant_filter = {str(c).strip() for c in rel if c}

    def label_tile(
        self,
        tile_bbox: Tuple[int, int, int, int],
        boxes: Sequence[CocoBox],
    ) -> TileLabelResult:
        """
        tile_bbox: (x0, y0, x1, y1) in image pixels (as produced by GridTiler).
        """
        cfg = self.config
        tx0, ty0, tx1, ty1 = map(float, tile_bbox)
        tile_xyxy = (tx0, ty0, tx1, ty1)
        tile_area = max(1e-6, (tx1 - tx0) * (ty1 - ty0))

        best_key: Optional[Tuple[float, int]] = None  # (-area, category_id)
        best_box: Optional[CocoBox] = None
        n_hit = 0

        for box in boxes:
            name_cf = box.category_name.strip().casefold()
            if name_cf in self._non_object:
                continue
            area = _intersection_area(tile_xyxy, box)
            if area <= 0:
                continue
            if area < float(cfg.min_overlap_px or 0.0):
                continue
            if float(cfg.min_overlap_frac_of_tile or 0.0) > 0:
                if (area / tile_area) < float(cfg.min_overlap_frac_of_tile):
                    continue
            n_hit += 1
            # Largest intersection wins; ties → lower category_id
            key = (-area, int(box.category_id))
            if best_key is None or key < best_key:
                best_key = key
                best_box = box

        if best_box is None or best_key is None:
            return TileLabelResult(
                label=str(cfg.negative_label),
                relevant=False,
                overlap_area=0.0,
                winning_category_id=None,
                n_overlapping_boxes=0,
            )

        label = str(best_box.category_name)
        if self._relevant_filter is None:
            relevant = True
        else:
            relevant = label in self._relevant_filter

        return TileLabelResult(
            label=label,
            relevant=bool(relevant),
            overlap_area=float(-best_key[0]),
            winning_category_id=int(best_box.category_id),
            n_overlapping_boxes=n_hit,
        )
