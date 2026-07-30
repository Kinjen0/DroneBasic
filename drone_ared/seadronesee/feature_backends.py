"""Feature extractors for SeaDronesSee: raw pixels or DINOv3."""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
from PIL import Image

from ..config import FeatureConfig, TilingConfig
from .config import RawFeatureConfig

# FeatureExtractor ABC is imported lazily where needed so raw-only callers
# do not require torch at import time. DINOFeatureExtractor is also lazy.


class RawPixelFeatureExtractor:
    """Flatten tile pixels into a fixed-length float32 vector.

    Duck-types ``FeatureExtractor`` (extract_images / output_dim / extract_single).
    """

    def __init__(
        self,
        tile_width: int,
        tile_height: int,
        grayscale: bool = False,
        scale_to_unit: bool = True,
        l2_normalize: bool = True,
    ):
        self.tile_w = int(tile_width)
        self.tile_h = int(tile_height)
        self.grayscale = bool(grayscale)
        self.scale_to_unit = bool(scale_to_unit)
        self.l2_normalize = bool(l2_normalize)
        channels = 1 if self.grayscale else 3
        self._output_dim = self.tile_w * self.tile_h * channels

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def extract_single(self, image: Image.Image) -> np.ndarray:
        return self.extract_images([image])[0]

    def extract_images(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.output_dim), dtype=np.float32)
        rows = [self._one(img) for img in images]
        return np.stack(rows, axis=0).astype(np.float32)

    def _one(self, image: Image.Image) -> np.ndarray:
        if self.grayscale:
            arr = np.asarray(image.convert("L"), dtype=np.float32)
        else:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        if arr.shape[0] != self.tile_h or arr.shape[1] != self.tile_w:
            raise ValueError(
                f"RawPixelFeatureExtractor expected {self.tile_w}x{self.tile_h} tiles, "
                f"got {arr.shape[1]}x{arr.shape[0]}"
            )
        vec = arr.reshape(-1)
        if self.scale_to_unit:
            vec = vec / 255.0
        if self.l2_normalize:
            n = float(np.linalg.norm(vec))
            if n > 1e-12:
                vec = vec / n
        return vec.astype(np.float32)


def build_feature_extractor(
    mode: Literal["raw", "dino"] | str,
    tiling: TilingConfig,
    feature_cfg: Optional[FeatureConfig] = None,
    raw_cfg: Optional[RawFeatureConfig] = None,
):
    """Factory: raw pixels or HuggingFace DINO (v2/v3)."""
    mode_l = (mode or "dino").strip().lower()
    if mode_l == "raw":
        rc = raw_cfg or RawFeatureConfig()
        return RawPixelFeatureExtractor(
            tile_width=int(tiling.tile_width),
            tile_height=int(tiling.tile_height),
            grayscale=rc.grayscale,
            scale_to_unit=rc.scale_to_unit,
            l2_normalize=rc.l2_normalize,
        )
    if mode_l in ("dino", "dinov2", "dinov3"):
        from ..feature_extractor import DINOFeatureExtractor  # lazy: needs torch

        fc = feature_cfg or FeatureConfig()
        return DINOFeatureExtractor(
            model_name=fc.model_name,
            device=fc.device,
            normalize=fc.normalize,
            pooling=fc.pooling,
            batch_size=fc.batch_size,
        )
    raise ValueError(f"Unknown feature_mode '{mode}' (expected 'raw' or 'dino')")
