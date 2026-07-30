"""COCO annotation index for SeaDronesSee processed export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


@dataclass(frozen=True)
class CocoBox:
    """One COCO instance box in pixel coordinates (x, y, w, h top-left)."""

    ann_id: int
    image_id: int
    category_id: int
    category_name: str
    x: float
    y: float
    w: float
    h: float

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def area(self) -> float:
        return max(0.0, float(self.w)) * max(0.0, float(self.h))


class CocoAnnotationIndex:
    """In-memory COCO instances index keyed by image_id."""

    def __init__(self, coco: Dict[str, Any]):
        self._coco = coco
        self.categories: Dict[int, str] = {
            int(c["id"]): str(c.get("name", str(c["id"])))
            for c in coco.get("categories", [])
        }
        self.images_by_id: Dict[int, Dict[str, Any]] = {
            int(im["id"]): im for im in coco.get("images", [])
        }
        self._boxes: Dict[int, List[CocoBox]] = {}
        for a in coco.get("annotations", []):
            iid = int(a["image_id"])
            cid = int(a.get("category_id", -1))
            bbox = a.get("bbox") or [0, 0, 0, 0]
            if len(bbox) < 4:
                continue
            x, y, w, h = map(float, bbox[:4])
            box = CocoBox(
                ann_id=int(a.get("id", 0)),
                image_id=iid,
                category_id=cid,
                category_name=self.categories.get(cid, str(cid)),
                x=x,
                y=y,
                w=w,
                h=h,
            )
            self._boxes.setdefault(iid, []).append(box)

    @classmethod
    def from_json_path(cls, path: str | Path) -> "CocoAnnotationIndex":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def get_boxes(self, image_id: int) -> List[CocoBox]:
        return list(self._boxes.get(int(image_id), []))

    def image_ids(self) -> List[int]:
        return sorted(self.images_by_id.keys())

    @property
    def num_images(self) -> int:
        return len(self.images_by_id)

    @property
    def num_annotations(self) -> int:
        return sum(len(v) for v in self._boxes.values())
