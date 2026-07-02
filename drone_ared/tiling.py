"""
Tiling module.

Provides an abstract Tiler and a concrete grid-based implementation.
The design allows future strategies (saliency, object-proposal, temporal, multi-scale, etc.)
without changing the rest of the pipeline.

A "Tile" carries:
- the original RGB image crop (PIL or np) so the GUI can display it
- metadata (frame index, tile row/col or bbox, global tile id)
- convenience methods

All tiles are produced in deterministic streaming order.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator, Any
import numpy as np
from PIL import Image


@dataclass
class Tile:
    """
    One spatial tile extracted from a video frame.

    Identity fields (new for exact label DB):
    - video_path: full path to the source video. Used as stable key for annotations.
    - frame_idx: absolute frame index from the *beginning of this video file* (0-based).
                  This is independent of any processing stride. Critical for label persistence
                  across different frame_stride settings.
    - tile_row / tile_col + bbox + (width/height): uniquely locate the crop inside that frame.

    We never persist the .image pixels to disk (storage). The image is only held in RAM for the
    current processing batch + GUI display. For future review of old labels we re-extract on demand.
    """
    image: Image.Image                  # RGB PIL image (the crop that will be shown to user + fed to DINO)
    frame_idx: int
    tile_row: int
    tile_col: int
    bbox: Tuple[int, int, int, int]     # (x0, y0, x1, y1) in original frame pixels
    global_idx: int = 0                 # strictly increasing id across the whole run
    video_path: str = ""                # NEW: exact source video for identity-based labeling
    extra: dict = field(default_factory=dict)  # future: timestamp, gps, etc.

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_numpy(self) -> np.ndarray:
        """Return HWC uint8 RGB numpy view (copy)."""
        return np.array(self.image)


class Tiler(ABC):
    """Abstract base for any frame -> list[Tile] strategy."""

    @abstractmethod
    def tile_frame(self, frame: np.ndarray, frame_idx: int, global_start_idx: int = 0,
                   video_path: str = "") -> List[Tile]:
        """
        Given a single video frame (H, W, 3) uint8, return ordered list of Tile objects.
        `global_start_idx` allows the caller to maintain a monotonic global tile counter.
        `video_path` (if supplied) is attached to tiles for exact identity labeling.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class GridTiler(Tiler):
    """
    Simple non-overlapping (or overlapping) regular grid tiling.

    This matches the user's stated "Tiling" approach.
    Default tile size 224x224 matches common DINO input expectations.
    """

    def __init__(
        self,
        tile_width: int = 224,
        tile_height: int = 224,
        stride_x: Optional[int] = None,
        stride_y: Optional[int] = None,
    ):
        self.tile_w = tile_width
        self.tile_h = tile_height
        self.stride_x = stride_x or tile_width
        self.stride_y = stride_y or tile_height

    @property
    def name(self) -> str:
        return f"grid_{self.tile_w}x{self.tile_h}_s{self.stride_x}x{self.stride_y}"

    def tile_frame(self, frame: np.ndarray, frame_idx: int, global_start_idx: int = 0,
                   video_path: str = "") -> List[Tile]:
        """Produce ONLY full, exactly-sized tiles.

        This guarantees every tile is identical in dimensions (tile_w x tile_h pixels).
        Clipped rectangles at frame edges are skipped. This keeps all inputs to DINO
        uniform before the model's internal resize.

        video_path (optional but recommended): attached to every Tile so that the exact
        label database can key annotations by (video, absolute frame, tile position, resolution).
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("GridTiler expects HWC RGB uint8 frame")

        h, w = frame.shape[:2]
        tiles: List[Tile] = []
        gidx = global_start_idx

        if self.tile_w > w or self.tile_h > h:
            # Frame too small for even one tile; return empty (or caller can decide to resize whole frame)
            return tiles

        row = 0
        y = 0
        while y + self.tile_h <= h:
            col = 0
            x = 0
            while x + self.tile_w <= w:
                crop = frame[y : y + self.tile_h, x : x + self.tile_w]
                pil_img = Image.fromarray(crop).convert("RGB")

                tile = Tile(
                    image=pil_img,
                    frame_idx=frame_idx,
                    tile_row=row,
                    tile_col=col,
                    bbox=(x, y, x + self.tile_w, y + self.tile_h),
                    global_idx=gidx,
                    video_path=video_path or "",
                )
                tiles.append(tile)
                gidx += 1

                x += self.stride_x
                col += 1

            y += self.stride_y
            row += 1

        return tiles

    def __repr__(self):
        return f"GridTiler(tile={self.tile_w}x{self.tile_h}, stride={self.stride_x}x{self.stride_y})"
