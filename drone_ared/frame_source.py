"""
FrameSource abstraction: video files and image directories as one stream interface.

The pipeline, review UI, and multi-frame browser only need sequential / seekable
RGB frames. This module isolates how those frames are produced so callers do not
depend on cv2.VideoCapture or a particular on-disk layout.

Design (mirrors FeatureExtractor / Tiler ABCs):
  - FrameSource          abstract contract
  - VideoFileFrameSource wraps a video file via OpenCV
  - ImageSequenceFrameSource treats a directory of images as frame 0..N-1
  - open_frame_source()  factory: path → concrete source

Annotation identity still uses Path(identity_path).name (file or folder basename).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


PathLike = Union[str, Path]

# Non-recursive image sequence extensions (case-insensitive).
IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
})

_NAT_SPLIT_RE = re.compile(r"(\d+)")


def natural_sort_key(name: str):
    """Sort key so 'img2.jpg' precedes 'img10.jpg'."""
    parts = _NAT_SPLIT_RE.split(name)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def list_image_files(directory: PathLike) -> List[Path]:
    """Non-recursive list of image files in *directory*, natural-sorted by name."""
    d = Path(directory)
    if not d.is_dir():
        return []
    files = [
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: natural_sort_key(p.name))
    return files


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR (or BGRA) uint8 to RGB HWC."""
    if frame is None:
        raise ValueError("frame is None")
    if frame.ndim == 2:
        # Grayscale → RGB
        if HAS_CV2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return np.stack([frame, frame, frame], axis=-1)
    if frame.shape[2] == 4:
        if HAS_CV2:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        return frame[:, :, :3][:, :, ::-1].copy()
    if HAS_CV2:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame[:, :, ::-1].copy()


class FrameSource(ABC):
    """
    Seekable stream of RGB frames (H, W, 3) uint8.

    Subclass for new backends (RTSP, in-memory buffers, etc.) without touching
    the pipeline or GUI.
    """

    @property
    @abstractmethod
    def identity_path(self) -> str:
        """Path whose basename is the annotation DB video key."""
        ...

    @property
    @abstractmethod
    def frame_count(self) -> Optional[int]:
        """Total frames if known, else None."""
        ...

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame and advance the cursor.

        Returns (True, rgb_array) on success, (False, None) at EOF / failure.
        """
        ...

    @abstractmethod
    def seek(self, frame_idx: int) -> bool:
        """
        Position so the next read() returns absolute frame *frame_idx* (0-based).

        Returns False if the index is out of range or seek failed.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release native resources. Safe to call more than once."""
        ...

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Seek to *frame_idx* and return one RGB frame, or None."""
        if not self.seek(frame_idx):
            return None
        ok, frame = self.read()
        return frame if ok else None


class VideoFileFrameSource(FrameSource):
    """RGB frames from a video file via OpenCV VideoCapture."""

    def __init__(self, path: PathLike):
        if not HAS_CV2:
            raise RuntimeError("opencv (cv2) is required to open video files")
        self._path = str(Path(path))
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            self._cap.release()
            raise OSError(f"Failed to open video: {self._path}")
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._frame_count: Optional[int] = count if count > 0 else None
        self._pos = 0  # next frame index after successful sequential reads (best-effort)

    @property
    def identity_path(self) -> str:
        return self._path

    @property
    def frame_count(self) -> Optional[int]:
        return self._frame_count

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        self._pos += 1
        try:
            return True, _bgr_to_rgb(frame)
        except Exception:
            return False, None

    def seek(self, frame_idx: int) -> bool:
        if self._cap is None:
            return False
        idx = int(frame_idx)
        if idx < 0:
            return False
        if self._frame_count is not None and idx >= self._frame_count:
            return False
        ok = bool(self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx))
        if ok:
            self._pos = idx
        return ok

    def release(self) -> None:
        if getattr(self, "_cap", None) is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class ImageSequenceFrameSource(FrameSource):
    """
    Directory of images as a video-equivalent stream.

    Frame index i maps to the i-th path in a natural-sorted, non-recursive listing.
    identity_path is the directory path (basename becomes the DB key).
    """

    def __init__(self, directory: PathLike, image_paths: Optional[List[Path]] = None):
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        self._dir = d.resolve()
        if image_paths is not None:
            self._paths = [Path(p) for p in image_paths]
        else:
            self._paths = list_image_files(self._dir)
        if not self._paths:
            raise OSError(f"No images found in directory: {self._dir}")
        self._cursor = 0  # next index to read

    @property
    def identity_path(self) -> str:
        return str(self._dir)

    @property
    def frame_count(self) -> Optional[int]:
        return len(self._paths)

    @property
    def image_paths(self) -> List[Path]:
        return list(self._paths)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cursor < 0 or self._cursor >= len(self._paths):
            return False, None
        path = self._paths[self._cursor]
        self._cursor += 1
        frame = self._load_rgb(path)
        if frame is None:
            return False, None
        return True, frame

    def seek(self, frame_idx: int) -> bool:
        idx = int(frame_idx)
        if idx < 0 or idx >= len(self._paths):
            return False
        self._cursor = idx
        return True

    def release(self) -> None:
        # Nothing held open; clear path list to drop references if desired.
        pass

    @staticmethod
    def _load_rgb(path: Path) -> Optional[np.ndarray]:
        """Load one image file as RGB uint8 HWC."""
        try:
            if HAS_PIL:
                with Image.open(path) as im:
                    rgb = im.convert("RGB")
                    return np.asarray(rgb, dtype=np.uint8)
            if HAS_CV2:
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if bgr is None:
                    return None
                return _bgr_to_rgb(bgr)
        except Exception as e:
            print(f"[FrameSource] Failed to load image {path}: {e}")
        return None


def open_frame_source(path: PathLike) -> FrameSource:
    """
    Factory: open *path* as a FrameSource.

    - Directory → ImageSequenceFrameSource (must contain at least one image)
    - File      → VideoFileFrameSource

    Raises OSError / NotADirectoryError / RuntimeError on failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Frame source not found: {p}")
    if p.is_dir():
        return ImageSequenceFrameSource(p)
    if p.is_file():
        return VideoFileFrameSource(p)
    raise OSError(f"Unsupported frame source path: {p}")
