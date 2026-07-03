"""
Data Augmentation for DINO-based A/RED runs.

When a tile receives a label (via real A/RED query or DB hit on a queried point),
we can rotate the original tile image, extract fresh DINO embeddings for the
rotated views, and insert those embeddings as additional labeled points
with the *exact same* label and relevance.

This augments the cluster(s) with multiple views of the same semantic content
(helpful for rotation variance in drone footage).

This is our custom implementation and is separate from (and does not use)
the internal pixel-based DATA_AUG_VAR in A_RED / FiniteBuffer, which expects
raw image pixels and does np.rot90 on them.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
from PIL import Image

# We import the feature extractor type for hints only
try:
    from .feature_extractor import FeatureExtractor
except Exception:
    FeatureExtractor = object  # type: ignore


def rotate_image(pil_img: Image.Image, angle: int, resample: int = Image.BICUBIC) -> Image.Image:
    """Rotate a square tile image by the given degrees.

    expand=False keeps the output the same size (important for square tiles).
    """
    if pil_img is None:
        raise ValueError("Cannot rotate None image")
    # Ensure RGB
    img = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
    return img.rotate(angle, resample=resample, expand=False)


class DINOAugmenter:
    """Helper that generates rotated versions of a tile and their DINO embeddings."""

    def __init__(self, feature_extractor: FeatureExtractor):
        self.fe = feature_extractor

    def get_rotated_embeddings(
        self,
        pil_image: Image.Image,
        angles: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """Return a list of DINO embeddings for the rotated versions of the image.

        The original image is NOT included here — the caller is responsible for
        the main embedding that went through the normal A/RED process.
        """
        if angles is None:
            angles = [90, 180, 270]

        if not angles or pil_image is None:
            return []

        rotated_imgs: List[Image.Image] = []
        for angle in angles:
            try:
                rot = rotate_image(pil_image, angle)
                rotated_imgs.append(rot)
            except Exception as e:
                print(f"[Augmenter] Failed to rotate by {angle}°: {e}")
                continue

        if not rotated_imgs:
            return []

        # Batch extract for efficiency
        try:
            embs = self.fe.extract_images(rotated_imgs)
            return [embs[i] for i in range(len(rotated_imgs))]
        except Exception as e:
            print(f"[Augmenter] DINO extraction on rotations failed: {e}")
            return []


def generate_augmented_embeddings(
    feature_extractor: FeatureExtractor,
    pil_image: Image.Image,
    angles: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """Convenience function (stateless)."""
    augmenter = DINOAugmenter(feature_extractor)
    return augmenter.get_rotated_embeddings(pil_image, angles)
