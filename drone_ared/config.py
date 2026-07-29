"""
Configuration for the Drone A/RED pipeline.

Central place for all tunable parameters. Designed to be easily extended
(e.g. add new feature extractor types, tiling modes, GUI themes, etc.).

All classes are simple data containers (dataclasses) so they are easy to
serialize to JSON/YAML for experiments or GUI persistence.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class TilingConfig:
    """How to split video frames into tiles.

    IMPORTANT: All produced tiles will be exactly (tile_width, tile_height).
    Partial / edge rectangles are deliberately skipped so every tile is uniform
    and compatible with DINO models (which expect consistent input after their
    internal resize, typically 224/518 etc.). 

    Recommendation: Use 256 or 320+ for more context per tile on drone footage.
    The DINO preprocessor will still resize the crop to the model's preferred
    resolution, but a larger original crop preserves more detail/context.

    Overlap support (new):
    - stride_x / stride_y control the step between consecutive tiles.
    - stride < tile size → overlapping tiles (recommended for 240x240 to avoid cutting objects).
    - You can set stride directly, or use the GUI overlap controls (overlap_px = tile - stride).
    - Default (stride == tile size) = classic non-overlapping grid.
    """
    tile_width: int = 240
    tile_height: int = 240
    # Stride: if None, non-overlapping (stride == tile size). Overlap if smaller.
    # stride_x/y are the authoritative runtime values used by GridTiler.
    stride_x: Optional[int] = None
    stride_y: Optional[int] = None
    # Optional convenience fields for overlap in pixels (tile_size - stride).
    # These are primarily for UI / documentation. When > 0 the GUI may use them
    # to compute stride = max(1, tile - overlap). The pipeline always uses stride_*.
    overlap_x: int = 120
    overlap_y: int = 120
    # Optional: process only every Nth frame (1 = every frame). Higher = faster, fewer tiles.
    frame_stride: int = 15
    # Future: support "adaptive" tiling, saliency-based, multi-scale, etc.
    tiling_mode: str = "grid"   # "grid", "sliding", "pyramid" (future)

    def __post_init__(self):
        if self.stride_x is None:
            self.stride_x = self.tile_width
        if self.stride_y is None:
            self.stride_y = self.tile_height
        # Keep overlap fields non-negative (they are sugar; stride is truth)
        self.overlap_x = max(0, int(self.overlap_x or 0))
        self.overlap_y = max(0, int(self.overlap_y or 0))

    def effective_stride(self) -> Tuple[int, int]:
        return (self.stride_x, self.stride_y)

    def effective_overlap(self) -> Tuple[int, int]:
        """Return (overlap_x, overlap_y) implied by current tile size and stride."""
        ox = max(0, self.tile_width - (self.stride_x or self.tile_width))
        oy = max(0, self.tile_height - (self.stride_y or self.tile_height))
        return (ox, oy)


@dataclass
class FeatureConfig:
    """DINO feature extractor settings."""
    # User asked for DINOV3. Current HF models: try "facebook/dinov2-base" or
    # newer dinov3 variants when released (e.g. "facebook/dinov3-vitb16").
    # The extractor is model-agnostic via transformers AutoModel.
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m" #"facebook/dinov2-base"
    device: Optional[str] = None  # "cuda", "cpu", or None -> auto
    # Whether to L2-normalize the output embeddings (often helpful for DINO)
    normalize: bool = True
    # Pooling strategy for ViT: "cls" (first token), "mean", "max"
    pooling: str = "mean"
    # Optional: target output dimension (None = native, e.g. 768 for base)
    # Future: plug a small trainable or PCA head here.
    output_dim: Optional[int] = None
    # Batch size for feature extraction (higher = faster on GPU)
    batch_size: int = 16

    # Example for future expansion:
    # use_registered_model: bool = False
    # cache_dir: Optional[str] = None


@dataclass
class AREDConfig:
    """Parameters passed to the core A/RED algorithm (A_REDIN.ARED).

    kappa: "paranoia" parameter (higher value = more paranoid = MORE queries).
           - Higher kappa → distance * kappa exceeds cluster size more easily → A/RED
             asks the user (or cache) for labels more often.
           - Lower kappa → more tolerant of variation → fewer queries.
    """
    kappa: float = 1.0          # higher = more queries (more paranoid)
    l_buf_size: int = 10000      # memory bound for labeled points (circular)
    k_comp_pts: int = 2         # how many nearest to consider (enables neighborhood merge)
    qs_var: int = 1             # 0=diameter, 1=average NN distance (single link style)
    data_aug_var: Tuple[int, Tuple[int, int]] = (0, (0, 0))
    nghbhood_merge: bool = True
    singleton_merge: bool = True
    small_cluster_threshold: int = 3
    smart_forgetting_var: Tuple[int, float] = (3, 0.01)
    # A_REDIN VERBOSE_FLAGS: only certain ints enable internal prints (1=add_l_pt, 5=merge, 6=forget).
    # Default empty = quiet. GUI "Terminal logging" can enable [1,5,6] when desired.
    verbose_flags: list = field(default_factory=list)

    # Data augmentation for DINO-based runs (our custom implementation).
    # When a tile is labeled (via query or DB), we rotate the original image,
    # extract fresh DINO embeddings, and insert them as labeled variants
    # with the SAME label/relevance. This is separate from the internal
    # pixel-based DATA_AUG_VAR (which doesn't work on embeddings).
    data_augmentation_enabled: bool = False
    augmentation_rotations: list = field(default_factory=lambda: [90, 180, 270])

    # These are exposed for GUI control / future tuning
    # Higher buffer + smart forgetting = can handle longer streams


@dataclass
class LabelCacheConfig:
    """Embedding-similarity auto-label store (NOT the annotation SQLite DB).

    When enabled, DINO embeddings close to past human labels (stored in the .pkl)
    can answer A/RED queries without the GUI — even if the tile annotation .db is
    empty or brand new. Disable for cold-start / clean-metrics experiments.
    """
    # Default OFF so a new annotation DB is not silently "contaminated" by old .pkl labels.
    # Re-enable in the GUI when you want cross-run similarity auto-labeling.
    enabled: bool = False
    db_path: str = "drone_ared_labels.pkl"   # embedding similarity cache (pickle)
    auto_label_threshold: float = 0.15
    rebuild_interval: int = 32
    distance_metric: str = "l2"


@dataclass
class TileAnnotationConfig:
    """Settings for the exact identity-based tile label database.

    This is the new primary way to remember labels by (video, absolute frame, tile position, resolution).
    Enables editing past labels and perfect recall across different frame strides.
    """
    enabled: bool = True
    db_path: str = "drone_tile_annotations.db"   # sqlite file
    # When True in the GUI, even tiles that have previous exact labels will pop the labeling dialog
    # so the user can correct mistakes. Normal runs auto-apply saved exact labels.
    edit_mode_default: bool = False

    # New: support for pure labeling sessions (no A/RED, no DINO)
    label_only_default: bool = False


@dataclass
class ModelSaveConfig:
    """Optional A/RED model checkpointing (state of clusters + labeled buffer)."""
    enabled: bool = False
    save_path: str = "ared_model_state.pkl"
    # Save automatically after this many queries (0 = manual only via GUI)
    autosave_every_n_queries: int = 0
    # When loading, we replay the saved labeled points using cached decisions.
    # This lets you continue "as if" previous labeling session happened.


@dataclass
class GUIConfig:
    """User interface behavior."""
    # Labeling dialog starts at this size; fully resizable by user.
    label_dialog_width: int = 900
    label_dialog_height: int = 700
    # Main control window
    main_window_width: int = 1100
    main_window_height: int = 700
    # How often (ms) the GUI polls for status / pending label requests
    poll_interval_ms: int = 80
    # Show a live mosaic / last frame preview? (costs some CPU)
    enable_preview: bool = True
    # Theme / colors can be extended later (e.g. ttk styles)
    use_ttk: bool = True

    # UI scale for high-resolution / 4K displays and readability.
    # 1.0 = default, 1.5-2.0 common for large screens or to enlarge buttons/fonts.
    # Affects Tk scaling factor, font point sizes, paddings, etc.
    ui_scale: float = 1.6

    # When True, emit high-volume repeating terminal lines (per-tile progress, cache hits, etc.).
    # Errors / start-stop / finalize always print regardless.
    terminal_logging: bool = True


@dataclass
class MetricsLoggingConfig:
    """Periodic running metrics + per-run save files (paper-style evaluation logs)."""
    enabled: bool = True
    # Snapshot QP/RR/F1 etc. every N tiles processed (also always on stop/finish).
    checkpoint_every: int = 500
    # Directory for runs/<run_id>/run.json + checkpoints.csv
    output_dir: str = "runs"
    # Also write a checkpoint when each video ends (in addition to every N tiles).
    checkpoint_on_video_end: bool = True
    # Secondary track: QP/RR/F1 for each checkpoint window only (batch metrics).
    # Cumulative running metrics are always written when enabled=True; this flag
    # only toggles the per-window batch_* fields and batches.csv rows.
    batch_metrics_enabled: bool = True
    # How "first appearance of a class" counts as a should-query positive:
    #   "paper"      — paper definition: first sample of *any* class is a positive
    #                  (fair for cold-start; unfair if a warm-started A_RED already knows the class).
    #   "skip_known" — first-of-class is a positive only if that class was *not* already
    #                  known to A_RED at run start (recommended when loading a prior model).
    #                  Relevant-class samples still always count as positives.
    #   "auto"       — use skip_known when the adapter has known labels at Start, else paper.
    first_occurrence_mode: str = "auto"


@dataclass
class PipelineConfig:
    """Top level configuration object."""
    tiling: TilingConfig = field(default_factory=TilingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ared: AREDConfig = field(default_factory=AREDConfig)
    label_cache: LabelCacheConfig = field(default_factory=LabelCacheConfig)
    tile_annotations: TileAnnotationConfig = field(default_factory=TileAnnotationConfig)  # NEW
    model_save: ModelSaveConfig = field(default_factory=ModelSaveConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    metrics_logging: MetricsLoggingConfig = field(default_factory=MetricsLoggingConfig)

    # Misc
    video_paths: list[str] = field(default_factory=list)
    output_dir: str = "outputs"
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            data = json.load(f)
        # Reconstruct nested dataclasses manually for simplicity
        tdata = data.get("tiling", {})
        ml_raw = data.get("metrics_logging", {}) or {}
        # Only pass known fields so older configs without metrics_logging still load
        ml_fields = {f.name for f in MetricsLoggingConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        ml_kwargs = {k: v for k, v in ml_raw.items() if k in ml_fields}
        return cls(
            tiling=TilingConfig(**tdata),
            features=FeatureConfig(**data.get("features", {})),
            ared=AREDConfig(**data.get("ared", {})),
            label_cache=LabelCacheConfig(**data.get("label_cache", {})),
            tile_annotations=TileAnnotationConfig(**data.get("tile_annotations", {})),
            model_save=ModelSaveConfig(**data.get("model_save", {})),
            gui=GUIConfig(**data.get("gui", {})),
            metrics_logging=MetricsLoggingConfig(**ml_kwargs),
            video_paths=data.get("video_paths", []),
            output_dir=data.get("output_dir", "outputs"),
            random_seed=data.get("random_seed", 42),
        )

    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls()
