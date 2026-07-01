"""
FeatureExtractor abstraction + concrete DINO implementation.

The design deliberately separates "how we turn a PIL image / numpy tile into a vector"
so that in the future we can:
  - Swap DINOv2 <-> DINOv3 <-> other foundation models (CLIP, SigLIP, etc.)
  - Add a learned projector / autoencoder head on top
  - Use different pooling or multi-scale features
  - Support ONNX / TensorRT runtimes for speed on drone edge

All extractors return float32 numpy arrays of shape (N, D).
They are expected to be usable from worker threads (extraction is read-only after init).
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
import warnings

# Suppress noisy transformer warnings during normal runs
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class FeatureExtractor:
    """
    Abstract base for any image -> embedding function.

    Subclass and implement .extract_images() for new backends.
    The rest of the pipeline only calls this interface.
    """

    def extract_images(self, images: List[Image.Image]) -> np.ndarray:
        """Return (N, D) float32 embeddings for a batch of PIL images."""
        raise NotImplementedError

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def extract_single(self, image: Image.Image) -> np.ndarray:
        """Convenience for one image -> (D,) vector."""
        return self.extract_images([image])[0]


class DINOFeatureExtractor(FeatureExtractor):
    """
    DINOv2 / DINOv3 (and compatible) extractor using Hugging Face transformers.

    This closely follows the pattern used in the original A_RED DINOv*_*.py modules,
    but packaged cleanly for reuse and extension.

    Typical models:
      - "facebook/dinov2-base"   -> 768 dim
      - "facebook/dinov2-large"  -> 1024 dim
      - Future DINOv3 names (user request) will work the same way once on HF.

    We do NOT apply the extra autoencoder reduction used in some parking-lot
    experiments; raw DINO features work well with A/RED.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: Optional[str] = None,
        normalize: bool = True,
        pooling: str = "mean",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.pooling = pooling.lower()
        self.batch_size = max(1, batch_size)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[DINOFeatureExtractor] Loading {model_name} on {self.device} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Probe native dimension using the chosen pooling
        with torch.no_grad():
            dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
            inputs = self.processor(images=[dummy], return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            feats = self._pool(outputs)
            self._output_dim = int(feats.shape[-1])

        print(f"[DINOFeatureExtractor] Ready. Output dim = {self._output_dim}")

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _pool(self, outputs) -> torch.Tensor:
        """
        Pool ViT hidden states into a single vector per image.

        - "cls"  : first token
        - "mean" : average over patch tokens (recommended for DINO)
        - "max"  : max over patch tokens
        """
        hidden = outputs.last_hidden_state  # (B, 1+patches, D)

        if self.pooling == "cls":
            return hidden[:, 0, :]
        elif self.pooling == "mean":
            return hidden[:, 1:, :].mean(dim=1)
        elif self.pooling == "max":
            return hidden[:, 1:, :].amax(dim=1)
        else:
            return hidden.mean(dim=1)

    def extract_images(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.output_dim), dtype=np.float32)

        all_feats: List[np.ndarray] = []

        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                feats = self._pool(outputs)

            if self.normalize:
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)

            all_feats.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_feats, axis=0)
