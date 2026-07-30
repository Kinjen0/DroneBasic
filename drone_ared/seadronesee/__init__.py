"""
SeaDronesSee auto-labeled A/RED side path.

Runs A/RED on SeaDroneSeeProcessedDataExport with ground-truth labels derived
from COCO bounding boxes (no human multi-frame labeling). Supports raw-pixel
or DINOv3 features. Isolated from the interactive drone GUI path.
"""

from .config import SeaDronesSeeConfig
from .runner import SeaDronesSeeRunner

__all__ = ["SeaDronesSeeConfig", "SeaDronesSeeRunner"]
