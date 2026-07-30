"""SeaDronesSee processed export dataset loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Any, Dict

from .coco_index import CocoAnnotationIndex, CocoBox


@dataclass(frozen=True)
class SeaDronesSeeImage:
    """One still image in the processed export."""

    split: str
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int
    meta: Dict[str, Any]

    @property
    def identity(self) -> str:
        """Stable stream identity for TileKey / metrics (basename after DB normalize)."""
        # Prefer bare filename — TileAnnotationDB stores basename only.
        # Filenames are unique across the filtered export.
        return self.file_name

    def boxes(self, index: CocoAnnotationIndex) -> List[CocoBox]:
        return index.get_boxes(self.image_id)


class SeaDronesSeeDataset:
    """Load train/val splits from SeaDroneSeeProcessedDataExport layout."""

    SPLITS = ("train", "val")

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"SeaDronesSee dataset root not found: {self.root}")
        self._indexes: Dict[str, CocoAnnotationIndex] = {}
        self._images: Dict[str, List[SeaDronesSeeImage]] = {}

    def _ann_path(self, split: str) -> Path:
        return self.root / "annotations" / f"instances_{split}.json"

    def _img_dir(self, split: str) -> Path:
        return self.root / "images" / split

    def load_split(self, split: str) -> Tuple[CocoAnnotationIndex, List[SeaDronesSeeImage]]:
        split = split.lower().strip()
        if split not in self.SPLITS:
            raise ValueError(f"Unknown split '{split}'; expected one of {self.SPLITS}")
        if split in self._indexes:
            return self._indexes[split], self._images[split]

        ann_path = self._ann_path(split)
        if not ann_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotations: {ann_path}")
        index = CocoAnnotationIndex.from_json_path(ann_path)
        img_dir = self._img_dir(split)
        images: List[SeaDronesSeeImage] = []
        missing = 0
        # Deterministic order by image_id
        for iid in index.image_ids():
            im = index.images_by_id[iid]
            fname = str(im.get("file_name") or "")
            path = img_dir / fname
            if not path.is_file():
                missing += 1
                continue
            images.append(
                SeaDronesSeeImage(
                    split=split,
                    image_id=int(iid),
                    file_name=fname,
                    path=path,
                    width=int(im.get("width") or 0),
                    height=int(im.get("height") or 0),
                    meta={
                        "coco_meta": im.get("meta") or {},
                        "source": im.get("source") or {},
                        "split": split,
                    },
                )
            )
        if missing:
            print(f"[SeaDronesSeeDataset] {split}: skipped {missing} missing image file(s)")
        self._indexes[split] = index
        self._images[split] = images
        print(
            f"[SeaDronesSeeDataset] Loaded {split}: {len(images)} images, "
            f"{index.num_annotations} annotations from {self.root}"
        )
        return index, images

    def iter_images(
        self,
        split: str = "train",
        max_images: Optional[int] = None,
    ) -> Iterator[Tuple[SeaDronesSeeImage, CocoAnnotationIndex]]:
        """Yield (image, index) for split, or train then val when split=='both'."""
        splits: Sequence[str]
        if split.lower().strip() == "both":
            splits = self.SPLITS
        else:
            splits = (split.lower().strip(),)

        count = 0
        for sp in splits:
            index, images = self.load_split(sp)
            for im in images:
                if max_images is not None and count >= max_images:
                    return
                yield im, index
                count += 1

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"root": str(self.root), "splits": {}}
        for sp in self.SPLITS:
            try:
                idx, imgs = self.load_split(sp)
                out["splits"][sp] = {
                    "images": len(imgs),
                    "annotations": idx.num_annotations,
                    "categories": dict(idx.categories),
                }
            except Exception as e:
                out["splits"][sp] = {"error": str(e)}
        return out
