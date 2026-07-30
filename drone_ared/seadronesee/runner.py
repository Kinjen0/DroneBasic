"""
SeaDronesSeeRunner — orchestrates auto-labeled A/RED on the processed export.

Duck-types the attributes RunMetricsLogger expects from DroneAREDController so
metrics packages are identical in layout. No human labeling UI.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..ared_adapter import AREDAdapter
from ..annotation_domain import TileKey
from ..tiling import GridTiler, Tile
from ..tile_database import TileAnnotationDB
from .. import metrics as ared_metrics
from ..run_metrics_logger import RunMetricsLogger
from ..label_sentinels import LabelCancelled

from .config import SeaDronesSeeConfig, PipelineConfigShim
from .dataset import SeaDronesSeeDataset, SeaDronesSeeImage
from .coco_index import CocoAnnotationIndex, CocoBox
from .tile_labeler import BBoxTileLabeler
from .feature_backends import build_feature_extractor
from .oracle_provider import AutoAnnotationLabelProvider


def _load_rgb(path: Path) -> Optional[np.ndarray]:
    try:
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception as e:
        print(f"[SDSRunner] Failed to load image {path}: {e}")
        return None


class SeaDronesSeeRunner:
    """Background worker that streams SDS tiles into A_RED with COCO GT labels."""

    def __init__(self, config: Optional[SeaDronesSeeConfig] = None):
        self.sds_config = config or SeaDronesSeeConfig.default()
        # Duck-typed .config for RunMetricsLogger / metrics helpers
        self.config = self._make_shim(self.sds_config)

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._worker_thread: Optional[threading.Thread] = None

        self.tiler: Optional[GridTiler] = None
        self.feature_extractor = None
        self.ared_adapter: Optional[AREDAdapter] = None
        self.tile_db: Optional[TileAnnotationDB] = None
        self.annotation_manager = None  # metrics logger checks this first
        self.label_store = None

        self.dataset: Optional[SeaDronesSeeDataset] = None
        self.labeler: Optional[BBoxTileLabeler] = None

        # image identity -> boxes (for provider)
        self._boxes_by_identity: Dict[str, List[CocoBox]] = {}
        # image identity -> SeaDronesSeeImage (debug)
        self._images_by_identity: Dict[str, SeaDronesSeeImage] = {}

        self.stats: Dict[str, Any] = {
            "frames_read": 0,
            "tiles_processed": 0,
            "ared_queries": 0,
            "user_queries": 0,  # always 0 — no human dialogs
            "cache_hits": 0,
            "ared_clusters": 0,
            "ared_known_labels": 0,
            "current_video": "",
            "status": "idle",
            "images_done": 0,
            "gt_positives": 0,
            "gt_negatives": 0,
            "feature_mode": self.sds_config.feature_mode,
            "feature_dim": None,
        }

        self.queried_identities: List[Tuple] = []
        self.processed_identities: List[Tuple] = []
        self.ared_known_labels_at_run_start: set = set()
        self.ared_label_inventory_at_run_start: Dict[str, Any] = {}
        self.ared_model_provenance: Dict[str, Any] = {
            "used_existing_model": False,
            "source": "none",
            "path": None,
            "name": None,
            "strategy": None,
            "path_a": None,
            "path_b": None,
            "name_a": None,
            "name_b": None,
            "saved_path": None,
        }

        self.run_metrics_logger: Optional[RunMetricsLogger] = None
        self.on_stats: Optional[Callable[[Dict], None]] = None
        self._last_query_info: Optional[Dict[str, Any]] = None
        self._gt_buffer: List[Tuple[TileKey, str, bool]] = []
        self._global_tile_counter = 0
        self._preserve_ared_on_start = False  # set True after load_state

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_shim(cfg: SeaDronesSeeConfig) -> PipelineConfigShim:
        return PipelineConfigShim(
            tiling=cfg.tiling,
            features=cfg.features,
            ared=cfg.ared,
            metrics_logging=cfg.metrics_logging,
            tile_annotations_db=cfg.tile_annotations_db,
            video_paths=[],
        )

    def update_config(self, cfg: SeaDronesSeeConfig) -> None:
        self.sds_config = cfg
        self.config = self._make_shim(cfg)
        self.stats["feature_mode"] = cfg.feature_mode

    # ------------------------------------------------------------------
    # Public control surface
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            print("[SDSRunner] Already running")
            return

        self._stop_event.clear()
        self._pause_event.set()
        self._global_tile_counter = 0
        self.stats["tiles_processed"] = 0
        self.stats["frames_read"] = 0
        self.stats["ared_queries"] = 0
        self.stats["user_queries"] = 0
        self.stats["cache_hits"] = 0
        self.stats["images_done"] = 0
        self.stats["gt_positives"] = 0
        self.stats["gt_negatives"] = 0
        self.queried_identities = []
        self.processed_identities = []
        self._gt_buffer = []
        self._boxes_by_identity = {}
        self._images_by_identity = {}

        create_ared = (self.ared_adapter is None) or (not self._preserve_ared_on_start)
        self._init_components(create_ared=create_ared)
        if create_ared:
            self.clear_ared_model_provenance()
        self._preserve_ared_on_start = False

        if self.ared_adapter is not None:
            try:
                self.ared_adapter.apply_runtime_hyperparams(self.sds_config.ared)
            except Exception as e:
                print(f"[SDSRunner] apply_runtime_hyperparams: {e}")

        self.ared_known_labels_at_run_start = set()
        self.ared_label_inventory_at_run_start = {}
        if self.ared_adapter is not None:
            try:
                inv = self.ared_adapter.get_model_label_inventory()
                self.ared_label_inventory_at_run_start = inv
                self.ared_known_labels_at_run_start = set(inv.get("labels") or [])
            except Exception:
                try:
                    self.ared_known_labels_at_run_start = set(
                        self.ared_adapter.get_known_labels() or []
                    )
                except Exception:
                    pass

        self._start_run_metrics_logger()

        self._worker_thread = threading.Thread(
            target=self._run_loop, daemon=True, name="sds-ared-worker"
        )
        self._worker_thread.start()
        self.stats["status"] = "running"
        print("[SDSRunner] Started")

    def pause(self) -> None:
        if not self._pause_event.is_set():
            return
        self._pause_event.clear()
        self.stats["status"] = "paused"
        print("[SDSRunner] Paused")

    def resume(self) -> None:
        if self._pause_event.is_set():
            return
        self._pause_event.set()
        self.stats["status"] = "running"
        print("[SDSRunner] Resumed")

    def stop(self, join_timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=join_timeout)
            self._worker_thread = None
        self._flush_gt_buffer()
        self.stats["status"] = "stopped"
        self._finalize_run_metrics(status="stopped")
        print("[SDSRunner] Stopped")

    def is_running(self) -> bool:
        return bool(self._worker_thread and self._worker_thread.is_alive())

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    # ------------------------------------------------------------------
    # Model I/O
    # ------------------------------------------------------------------
    def save_ared_state(self, path: str | Path) -> None:
        if self.ared_adapter is None:
            raise RuntimeError("No A_RED adapter to save")
        self.ared_adapter.save_state(path)
        try:
            self.ared_model_provenance["saved_path"] = str(path)
        except Exception:
            pass

    def load_ared_state(self, path: str | Path) -> None:
        """Load a saved A_RED model; keep it across the next start()."""
        path = Path(path)
        if self.ared_adapter is None:
            self.ared_adapter = AREDAdapter(self.sds_config.ared)
        self.ared_adapter.load_state(path, prefer_current_kappa=True)
        self._preserve_ared_on_start = True
        self.ared_model_provenance = {
            "used_existing_model": True,
            "source": "loaded",
            "path": str(path),
            "name": path.name,
            "strategy": None,
            "path_a": None,
            "path_b": None,
            "name_a": None,
            "name_b": None,
            "saved_path": None,
        }
        print(f"[SDSRunner] Loaded A_RED state from {path}")

    def clear_ared_model_provenance(self) -> None:
        self.ared_model_provenance = {
            "used_existing_model": False,
            "source": "none",
            "path": None,
            "name": None,
            "strategy": None,
            "path_a": None,
            "path_b": None,
            "name_a": None,
            "name_b": None,
            "saved_path": None,
        }

    def reset_ared(self) -> None:
        """Drop warm-start model so next Start is cold."""
        self.ared_adapter = None
        self._preserve_ared_on_start = False
        self.clear_ared_model_provenance()

    # ------------------------------------------------------------------
    # Metrics host API
    # ------------------------------------------------------------------
    def get_queried_identities(self) -> List[Tuple]:
        return list(self.queried_identities)

    def get_processed_identities(self) -> List[Tuple]:
        return list(self.processed_identities)

    def _collect_run_params(self) -> Dict[str, Any]:
        cfg = self.sds_config
        t = cfg.tiling
        a = cfg.ared
        p: Dict[str, Any] = {
            "pipeline": "seadronesee_auto",
            "dataset_root": str(cfg.dataset_root),
            "split": cfg.split,
            "feature_mode": cfg.feature_mode,
            "kappa": float(a.kappa),
            "tile_size": (int(t.tile_width), int(t.tile_height)),
            "stride_x": int(t.stride_x or t.tile_width),
            "stride_y": int(t.stride_y or t.tile_height),
            "annotation_db": str(
                getattr(self.tile_db, "db_path", None) or cfg.tile_annotations_db
            ),
            "dino_model": cfg.features.model_name if cfg.feature_mode == "dino" else None,
            "raw_grayscale": bool(cfg.raw_features.grayscale) if cfg.feature_mode == "raw" else None,
            "raw_l2_normalize": bool(cfg.raw_features.l2_normalize) if cfg.feature_mode == "raw" else None,
            "l_buf_size": int(a.l_buf_size),
            "k_comp_pts": int(a.k_comp_pts),
            "qs_var": int(a.qs_var),
            "nghbhood_merge": bool(a.nghbhood_merge),
            "singleton_merge": bool(a.singleton_merge),
            "data_augmentation_enabled": False,
            "label_cache_enabled": False,
            "label_only_mode": False,
            "edit_mode": False,
            "max_images": cfg.max_images,
            "max_tiles": cfg.max_tiles,
            "negative_label": cfg.labeling.negative_label,
            "first_occurrence_mode": getattr(
                cfg.metrics_logging, "first_occurrence_mode", "auto"
            ),
            "ared_known_labels_at_run_start": sorted(
                self.ared_known_labels_at_run_start, key=lambda s: str(s).casefold()
            ),
            "ared_model_used": bool(self.ared_model_provenance.get("used_existing_model")),
            "ared_model_source": self.ared_model_provenance.get("source") or "none",
            "ared_model_path": self.ared_model_provenance.get("path"),
            "ared_model_name": self.ared_model_provenance.get("name"),
            "ared_model_provenance": dict(self.ared_model_provenance),
            "video_paths": [],
            "feature_dim": self.stats.get("feature_dim"),
        }
        try:
            if self.ared_adapter is not None:
                p["kappa_effective"] = float(self.ared_adapter.ared.kappa)
            else:
                p["kappa_effective"] = float(a.kappa)
        except Exception:
            p["kappa_effective"] = float(a.kappa)
        if p["ared_model_used"]:
            if p["ared_model_source"] == "loaded":
                p["ared_model_summary"] = f"loaded:{p.get('ared_model_name') or '?'}"
            else:
                p["ared_model_summary"] = f"session:{p.get('ared_model_name') or 'warm-start'}"
        else:
            p["ared_model_summary"] = "cold-start (no preloaded model)"
        return p

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _init_components(self, create_ared: bool = True) -> None:
        cfg = self.sds_config
        tcfg = cfg.tiling
        self.tiler = GridTiler(
            tile_width=tcfg.tile_width,
            tile_height=tcfg.tile_height,
            stride_x=tcfg.stride_x,
            stride_y=tcfg.stride_y,
        )

        print(f"[SDSRunner] Building feature extractor mode={cfg.feature_mode}")
        self.feature_extractor = build_feature_extractor(
            cfg.feature_mode,
            tiling=tcfg,
            feature_cfg=cfg.features,
            raw_cfg=cfg.raw_features,
        )
        try:
            self.stats["feature_dim"] = int(self.feature_extractor.output_dim)
        except Exception:
            self.stats["feature_dim"] = None

        self.dataset = SeaDronesSeeDataset(cfg.dataset_root)
        self.labeler = BBoxTileLabeler(cfg.labeling)

        db_path = cfg.tile_annotations_db
        self.tile_db = TileAnnotationDB(db_path)
        print(f"[SDSRunner] Annotation DB: {db_path}")

        if create_ared or self.ared_adapter is None:
            self.ared_adapter = AREDAdapter(cfg.ared)

        # Wire auto label provider (A_RED queries only)
        provider = AutoAnnotationLabelProvider(
            labeler=self.labeler,
            boxes_for_meta=self._boxes_for_meta,
            on_label=self._on_provider_label,
        )
        self.ared_adapter.set_label_provider(provider)
        self.ared_adapter.set_label_store(None)
        # No DINO augmentation on SDS path by default
        try:
            self.ared_adapter.set_feature_extractor(None)
        except Exception:
            pass

    def _boxes_for_meta(self, meta: Dict[str, Any]) -> List[CocoBox]:
        v = meta.get("video_path") or meta.get("video") or ""
        # Try exact, then basename
        if v in self._boxes_by_identity:
            return self._boxes_by_identity[v]
        base = Path(str(v)).name
        if base in self._boxes_by_identity:
            return self._boxes_by_identity[base]
        return []

    def _on_provider_label(self, result, meta: Dict[str, Any]) -> None:
        self._last_query_info = {
            "label": result.label,
            "relevant": result.relevant,
            "meta": dict(meta or {}),
        }

    def _start_run_metrics_logger(self) -> None:
        self.run_metrics_logger = None
        try:
            ml = self.sds_config.metrics_logging
            if ml is None or not getattr(ml, "enabled", True):
                return
            params = self._collect_run_params()
            self.run_metrics_logger = RunMetricsLogger(
                output_dir=getattr(ml, "output_dir", "runs") or "runs",
                checkpoint_every=int(getattr(ml, "checkpoint_every", 500) or 500),
                run_params=params,
                enabled=True,
                checkpoint_on_video_end=bool(getattr(ml, "checkpoint_on_video_end", True)),
                batch_metrics_enabled=bool(getattr(ml, "batch_metrics_enabled", True)),
            )
            self.stats["metrics_run_dir"] = str(self.run_metrics_logger.run_dir)
            self.stats["metrics_last_line"] = "Metrics logging started."
        except Exception as e:
            print(f"[SDSRunner] metrics logger failed: {e}")
            self.run_metrics_logger = None

    def _maybe_metrics_checkpoint(self, reason: str = "interval") -> None:
        logger = self.run_metrics_logger
        if not logger:
            return
        try:
            # Metrics read GT from the annotation DB. Flush only when a checkpoint
            # will actually fire (not on every tile).
            if reason == "interval":
                n = int(self.stats.get("tiles_processed", 0) or 0)
                every = int(getattr(logger, "checkpoint_every", 500) or 500)
                if n <= 0 or every <= 0 or (n % every) != 0:
                    return
                if n == getattr(logger, "_last_checkpoint_tiles", -1):
                    return
                self._flush_gt_buffer()
                snap = logger.maybe_checkpoint(self, reason="interval")
            else:
                self._flush_gt_buffer()
                snap = logger.checkpoint(self, reason=reason)
            if snap:
                self.stats["metrics_last_line"] = logger.one_line_status()
                self.stats["metrics_run_dir"] = str(logger.run_dir)
                self._emit_stats()
        except Exception as e:
            print(f"[SDSRunner] metrics checkpoint error: {e}")

    def _finalize_run_metrics(self, status: str = "finished") -> None:
        logger = self.run_metrics_logger
        if not logger:
            return
        try:
            self._flush_gt_buffer()
            logger.finalize(self, status=status)
            self.stats["metrics_last_line"] = logger.one_line_status()
            self.stats["metrics_run_dir"] = str(logger.run_dir)
        except Exception as e:
            print(f"[SDSRunner] metrics finalize error: {e}")

    def _emit_stats(self) -> None:
        if self.on_stats:
            try:
                self.on_stats(self.stats.copy())
            except Exception:
                pass

    def _flush_gt_buffer(self) -> None:
        if not self._gt_buffer or self.tile_db is None:
            self._gt_buffer = []
            return
        try:
            self.tile_db.bulk_set_annotations(
                self._gt_buffer,
                quiet=True,
                commit_every=max(1, int(self.sds_config.gt_commit_every)),
            )
        except Exception as e:
            print(f"[SDSRunner] GT flush failed: {e}")
        self._gt_buffer = []

    def _tile_key(self, tile: Tile, identity: str) -> TileKey:
        tcfg = self.sds_config.tiling
        return TileKey(
            video_path=identity,
            abs_frame=int(tile.frame_idx),
            tile_row=int(tile.tile_row),
            tile_col=int(tile.tile_col),
            tile_width=int(tile.width),
            tile_height=int(tile.height),
            stride_x=int(tcfg.stride_x or tcfg.tile_width),
            stride_y=int(tcfg.stride_y or tcfg.tile_height),
        )

    def _run_loop(self) -> None:
        try:
            assert self.dataset is not None
            assert self.tiler is not None
            assert self.feature_extractor is not None
            assert self.ared_adapter is not None
            assert self.labeler is not None

            cfg = self.sds_config
            batch_size = int(
                cfg.extract_batch_size
                or getattr(cfg.features, "batch_size", 16)
                or 16
            )
            max_tiles = cfg.max_tiles

            for image, index in self.dataset.iter_images(
                split=cfg.split, max_images=cfg.max_images
            ):
                if self._stop_event.is_set():
                    break
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                self._process_image(image, index, batch_size=batch_size, max_tiles=max_tiles)
                self.stats["images_done"] = int(self.stats.get("images_done", 0)) + 1
                self._maybe_metrics_checkpoint(reason="video_end")

                if max_tiles is not None and self.stats["tiles_processed"] >= max_tiles:
                    print(f"[SDSRunner] Reached max_tiles={max_tiles}")
                    break

            self._flush_gt_buffer()

            if self._stop_event.is_set():
                if self.stats.get("status") != "stopped":
                    self.stats["status"] = "stopped"
                    self._finalize_run_metrics(status="stopped")
            else:
                self.stats["status"] = "finished"
                self._finalize_run_metrics(status="finished")
            print(f"[SDSRunner] Done. stats={self.stats}")
            self._emit_stats()
        except Exception as e:
            print(f"[SDSRunner] FATAL: {e}")
            import traceback
            traceback.print_exc()
            self.stats["status"] = "error"
            self._flush_gt_buffer()
            self._finalize_run_metrics(status="error")
            self._emit_stats()

    def _process_image(
        self,
        image: SeaDronesSeeImage,
        index: CocoAnnotationIndex,
        *,
        batch_size: int,
        max_tiles: Optional[int],
    ) -> None:
        identity = image.identity
        self.stats["current_video"] = identity
        boxes = index.get_boxes(image.image_id)
        self._boxes_by_identity[identity] = boxes
        self._boxes_by_identity[Path(identity).name] = boxes
        self._images_by_identity[identity] = image

        frame = _load_rgb(image.path)
        if frame is None:
            return
        self.stats["frames_read"] = int(self.stats.get("frames_read", 0)) + 1

        tiles = self.tiler.tile_frame(
            frame, frame_idx=0, global_start_idx=self._global_tile_counter, video_path=identity
        )
        if not tiles:
            return

        # Process in batches for feature extraction
        for start in range(0, len(tiles), batch_size):
            if self._stop_event.is_set():
                break
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            batch = tiles[start : start + batch_size]
            if max_tiles is not None:
                remaining = max_tiles - int(self.stats["tiles_processed"])
                if remaining <= 0:
                    break
                batch = batch[:remaining]

            try:
                embs = self.feature_extractor.extract_images([t.image for t in batch])
            except Exception as e:
                print(f"[SDSRunner] feature extract failed on {identity}: {e}")
                import traceback
                traceback.print_exc()
                break

            for tile, emb in zip(batch, embs):
                if self._stop_event.is_set():
                    break
                self._process_one_tile(tile, emb, identity, boxes)
                self._global_tile_counter += 1

                if max_tiles is not None and self.stats["tiles_processed"] >= max_tiles:
                    break

            if max_tiles is not None and self.stats["tiles_processed"] >= max_tiles:
                break

        # Advance global counter even if we broke early mid-image
        if tiles:
            self._global_tile_counter = max(
                self._global_tile_counter,
                tiles[-1].global_idx + 1,
            )

    def _process_one_tile(
        self,
        tile: Tile,
        emb: np.ndarray,
        identity: str,
        boxes: List[CocoBox],
    ) -> None:
        assert self.labeler is not None
        assert self.ared_adapter is not None

        gt = self.labeler.label_tile(tile.bbox, boxes)
        key = self._tile_key(tile, identity)
        self._gt_buffer.append((key, gt.label, bool(gt.relevant)))
        if gt.relevant:
            self.stats["gt_positives"] = int(self.stats.get("gt_positives", 0)) + 1
        else:
            self.stats["gt_negatives"] = int(self.stats.get("gt_negatives", 0)) + 1

        if len(self._gt_buffer) >= max(1, int(self.sds_config.gt_commit_every)):
            self._flush_gt_buffer()

        tcfg = self.sds_config.tiling
        meta = {
            "video_path": identity,
            "frame": tile.frame_idx,
            "abs_frame": tile.frame_idx,
            "row": tile.tile_row,
            "col": tile.tile_col,
            "tile_row": tile.tile_row,
            "tile_col": tile.tile_col,
            "tile_width": tile.width,
            "tile_height": tile.height,
            "bbox": tile.bbox,
            "stride_x": int(tcfg.stride_x or tcfg.tile_width),
            "stride_y": int(tcfg.stride_y or tcfg.tile_height),
            "gt_label": gt.label,
            "gt_relevant": bool(gt.relevant),
        }

        try:
            info = self.ared_adapter.process(emb, tile_image=tile, meta=meta)
        except LabelCancelled:
            return
        except Exception as e:
            print(f"[SDSRunner] A_RED process error: {e}")
            import traceback
            traceback.print_exc()
            return

        self.stats["tiles_processed"] = int(self.stats.get("tiles_processed", 0)) + 1

        pkey = ared_metrics.tile_identity_from_meta(meta, tile)
        if pkey:
            self.processed_identities.append(pkey)
            if info.get("queried", False):
                self.queried_identities.append(pkey)

        if info.get("queried", False):
            self.stats["ared_queries"] = int(self.stats.get("ared_queries", 0)) + 1

        self.stats["ared_clusters"] = getattr(self.ared_adapter, "num_clusters", 0)
        self.stats["ared_known_labels"] = getattr(self.ared_adapter, "num_known_labels", 0)

        self._maybe_metrics_checkpoint(reason="interval")

        if self.stats["tiles_processed"] % 32 == 0:
            self._emit_stats()
