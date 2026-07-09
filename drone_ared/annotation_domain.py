"""
annotation_domain.py

Domain models for the exact tile annotation system.

These provide clear, type-safe, reusable representations for:
- Tile identity (the natural key for labels)
- Filters for queries and bulk operations
- Annotated tile records

This is part of the OOP/SRP refactor to improve modularity, readability,
and expandability. The models are independent of storage (sqlite) or UI.

Retains full backward compatibility with previous dict-based and 6-tuple APIs.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


@dataclass(frozen=True)
class TileKey:
    """Immutable identity for a tile. The primary key for exact annotations.

    Replaces passing around 6+ separate parameters (video, frame, row, col, w, h).
    Used for lookups, saves, deletes, etc.

    IMPORTANT for overlapping tiles:
    - (tile_row, tile_col) are *grid indices* produced by a specific stride.
    - Different stride/overlap values produce different (row, col) for the same physical pixels.
    - Therefore identity for correct label matching must incorporate stride (or the actual
      pixel crop origin) when overlap is used.

    We now carry optional stride_x / stride_y (None = unknown/legacy non-overlapping run).
    Lookups and saves should supply the stride used by the current GridTiler when possible.
    The DB also stores crop_x/crop_y (absolute pixel top-left) which can be used for
    physical-region matching independent of grid addressing.
    """
    video_path: str
    abs_frame: int
    tile_row: int
    tile_col: int
    tile_width: int
    tile_height: int
    # New: the stride used to generate this grid position. Critical for overlap support.
    # When None we treat it as legacy (usually stride == tile size).
    stride_x: Optional[int] = None
    stride_y: Optional[int] = None

    def to_tuple(self) -> Tuple[str, int, int, int, int, int, Optional[int], Optional[int]]:
        return (self.video_path, self.abs_frame, self.tile_row, self.tile_col,
                self.tile_width, self.tile_height, self.stride_x, self.stride_y)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TileKey":
        """Construct from the dicts returned by legacy get_annotations_for_video etc."""
        return cls(
            video_path=d["video_path"],
            abs_frame=int(d["abs_frame"]),
            tile_row=int(d["tile_row"]),
            tile_col=int(d["tile_col"]),
            tile_width=int(d["tile_width"]),
            tile_height=int(d["tile_height"]),
            stride_x=d.get("stride_x"),
            stride_y=d.get("stride_y"),
        )

    def size(self) -> Tuple[int, int]:
        return (self.tile_width, self.tile_height)

    def stride(self) -> Tuple[Optional[int], Optional[int]]:
        return (self.stride_x, self.stride_y)

    def __repr__(self) -> str:
        s = ""
        if self.stride_x is not None or self.stride_y is not None:
            s = f" stride=({self.stride_x},{self.stride_y})"
        return (f"TileKey({self.video_path}, f{self.abs_frame}, "
                f"r{self.tile_row}c{self.tile_col}, {self.tile_width}x{self.tile_height}{s})")


@dataclass
class AnnotationFilter:
    """Declarative filter for queries, bulk reassigns, deletes, etc.

    All non-None fields are ANDed together.
    Supports scoping to video + exact tile size (prevents cross-size leakage).

    For overlapping tiles: pass stride_x/stride_y to isolate labels created
    under a particular grid step. Legacy records (NULL stride in DB) are
    matched when the filter does not specify stride.

    Example:
        filt = AnnotationFilter(video_path=..., tile_width=240, tile_height=240,
                                stride_x=192, stride_y=192, labels=["person"])
    """
    video_path: Optional[str] = None
    labels: Optional[List[str]] = None   # match any of these (for IN clause)
    tile_width: Optional[int] = None
    tile_height: Optional[int] = None
    stride_x: Optional[int] = None
    stride_y: Optional[int] = None
    relevant: Optional[bool] = None
    frame_min: Optional[int] = None
    frame_max: Optional[int] = None
    # Extensible: updated_after, etc.


@dataclass
class TileAnnotation:
    """Rich in-memory representation of a labeled tile.

    Used by the service layer (AnnotationManager) for cleaner code.
    Can be converted to/from the legacy dict format.
    """
    key: TileKey
    label: str
    relevant: bool
    crop_x: Optional[int] = None
    crop_y: Optional[int] = None
    updated_ts: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None  # or bytes

    @property
    def video_path(self) -> str:
        return self.key.video_path

    def to_dict(self) -> Dict[str, Any]:
        """For compatibility with existing code that expects dicts from get_annotations_for_video."""
        cx = self.crop_x if self.crop_x is not None else self.key.tile_col * self.key.tile_width
        cy = self.crop_y if self.crop_y is not None else self.key.tile_row * self.key.tile_height
        return {
            "video_path": self.key.video_path,
            "abs_frame": self.key.abs_frame,
            "tile_row": self.key.tile_row,
            "tile_col": self.key.tile_col,
            "tile_width": self.key.tile_width,
            "tile_height": self.key.tile_height,
            "crop_x": cx,
            "crop_y": cy,
            "label": self.label,
            "relevant": self.relevant,
            "updated_ts": self.updated_ts,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TileAnnotation":
        key = TileKey.from_dict(d)
        return cls(
            key=key,
            label=d["label"],
            relevant=bool(d.get("relevant", False)),
            crop_x=d.get("crop_x"),
            crop_y=d.get("crop_y"),
            updated_ts=d.get("updated_ts", time.time()),
        )
