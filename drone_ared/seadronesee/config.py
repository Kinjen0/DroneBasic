"""Configuration for the SeaDronesSee auto-labeled A/RED pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set
import json
from pathlib import Path

from ..config import (
    TilingConfig,
    FeatureConfig,
    AREDConfig,
    MetricsLoggingConfig,
    ModelSaveConfig,
)


@dataclass
class RawFeatureConfig:
    """Flattened pixel features (no foundation model)."""

    grayscale: bool = False
    scale_to_unit: bool = True  # divide by 255
    l2_normalize: bool = True
    # Expected tile size is taken from TilingConfig at build time.


@dataclass
class TileLabelConfig:
    """How COCO boxes map to per-tile (label, relevant)."""

    negative_label: str = "water"
    ignored_as_negative: bool = True  # category "ignored" → negative_label
    # Any pixel overlap counts when both thresholds are 0.
    min_overlap_px: float = 0.0
    # Fraction of *tile* area that must overlap a box (0 = any pixel).
    min_overlap_frac_of_tile: float = 0.0
    # If set, only these category names are marked relevant=True.
    # None → all non-negative object categories are relevant.
    relevant_categories: Optional[List[str]] = None
    # Category names treated as non-objects (in addition to ignored when flagged).
    non_object_categories: List[str] = field(default_factory=lambda: ["ignored"])


@dataclass
class SeaDronesSeeConfig:
    """Top-level config for the SDS auto pipeline."""

    dataset_root: str = "SeaDroneSeeProcessedDataExport"
    split: str = "train"  # train | val | both

    tiling: TilingConfig = field(
        default_factory=lambda: TilingConfig(
            tile_width=32,
            tile_height=32,
            stride_x=32,
            stride_y=32,
            overlap_x=0,
            overlap_y=0,
            frame_stride=1,
        )
    )
    feature_mode: str = "dino"  # "raw" | "dino"
    features: FeatureConfig = field(default_factory=FeatureConfig)
    raw_features: RawFeatureConfig = field(default_factory=RawFeatureConfig)
    labeling: TileLabelConfig = field(default_factory=TileLabelConfig)
    ared: AREDConfig = field(default_factory=AREDConfig)
    metrics_logging: MetricsLoggingConfig = field(default_factory=MetricsLoggingConfig)
    model_save: ModelSaveConfig = field(default_factory=ModelSaveConfig)

    tile_annotations_db: str = "seadronesee_tile_annotations.db"
    # Smoke / subset controls (None = unlimited)
    max_images: Optional[int] = None
    max_tiles: Optional[int] = None
    # Feature extract batch size override (falls back to features.batch_size)
    extract_batch_size: Optional[int] = None
    # How often to flush GT rows to sqlite
    gt_commit_every: int = 2000
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "SeaDronesSeeConfig":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeaDronesSeeConfig":
        def _filter(dc_cls, raw):
            if not isinstance(raw, dict):
                return dc_cls()
            fields = {f.name for f in dc_cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            return dc_cls(**{k: v for k, v in raw.items() if k in fields})

        tdata = data.get("tiling", {}) or {}
        return cls(
            dataset_root=data.get("dataset_root", "SeaDroneSeeProcessedDataExport"),
            split=data.get("split", "train"),
            tiling=_filter(TilingConfig, tdata),
            feature_mode=str(data.get("feature_mode", "dino")),
            features=_filter(FeatureConfig, data.get("features", {})),
            raw_features=_filter(RawFeatureConfig, data.get("raw_features", {})),
            labeling=_filter(TileLabelConfig, data.get("labeling", {})),
            ared=_filter(AREDConfig, data.get("ared", {})),
            metrics_logging=_filter(MetricsLoggingConfig, data.get("metrics_logging", {})),
            model_save=_filter(ModelSaveConfig, data.get("model_save", {})),
            tile_annotations_db=data.get("tile_annotations_db", "seadronesee_tile_annotations.db"),
            max_images=data.get("max_images"),
            max_tiles=data.get("max_tiles"),
            extract_batch_size=data.get("extract_batch_size"),
            gt_commit_every=int(data.get("gt_commit_every", 2000) or 2000),
            random_seed=int(data.get("random_seed", 42) or 42),
        )

    @classmethod
    def default(cls) -> "SeaDronesSeeConfig":
        return cls()

    def relevant_category_set(self) -> Optional[Set[str]]:
        cats = self.labeling.relevant_categories
        if cats is None:
            return None
        return {str(c).strip() for c in cats if c}


@dataclass
class PipelineConfigShim:
    """Minimal duck-type of PipelineConfig fields that RunMetricsLogger / metrics touch.

    SeaDronesSeeRunner exposes ``.config`` as this shim so existing metrics code
    can read ``.tiling``, ``.ared``, ``.features``, ``.metrics_logging``, etc.
    without depending on the interactive PipelineConfig tree.
    """

    tiling: TilingConfig
    features: FeatureConfig
    ared: AREDConfig
    metrics_logging: MetricsLoggingConfig
    tile_annotations_db: str = "seadronesee_tile_annotations.db"
    # Unused by SDS but referenced defensively elsewhere
    video_paths: list = field(default_factory=list)
    label_cache_enabled: bool = False

    @property
    def tile_annotations(self):
        # Duck attribute used by _collect_run_params style code
        class _TA:
            def __init__(self, path):
                self.db_path = path
                self.enabled = True

        return _TA(self.tile_annotations_db)

    @property
    def label_cache(self):
        class _LC:
            enabled = False
            auto_label_threshold = 0.0
            db_path = ""

        return _LC()
