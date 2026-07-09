"""
Pipeline / Controller.

Owns the background threads that:
  1. Read video frames (cv2 preferred)
  2. Tile them (GridTiler)
  3. Extract features (DINO)
  4. Feed embeddings into AREDAdapter (which may request labels via queue)

Communication with the GUI thread uses two simple queues + events:
- label_request_queue : (tile, emb, meta) -> GUI pops and shows dialog
- label_response_queue : (label, relevant) sent back by GUI
- A threading.Event is used for blocking the ARED worker until response arrives.

The controller also manages pause / stop / config updates in a thread-safe way.
All heavy work is commented for future extension (e.g. multiple video sources, RTSP, etc.).
"""

from __future__ import annotations
import threading
import queue
import time
from pathlib import Path
from typing import Optional, List, Callable, Any, Dict, Tuple  # List still used for other tile lists / annotations
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[pipeline] WARNING: opencv (cv2) not found. Video reading will be limited.")

from .config import PipelineConfig
from .tiling import GridTiler, Tile
from .feature_extractor import DINOFeatureExtractor, FeatureExtractor
from .ared_adapter import AREDAdapter
from .label_store import PersistentLabelStore
from .tile_database import TileAnnotationDB
from .annotation_domain import TileKey  # domain model (wiring refactor)
from .annotation_manager import AnnotationManager
from . import metrics as ared_metrics   # Query Precision / Relevant Recall (see papers)


class LabelRequest:
    """Message sent to GUI when a label is needed (either from A/RED query or Label Only mode).

    Supports:
    - Normal labeling for A/RED.
    - "Label Only" / sparse labeling sessions.
    - Skip / Move On without labeling (for efficient relevant-focused labeling to support metrics).
    - Edit / resume: pre-filling current label when tile already exists in DB.
    """
    __slots__ = ("tile", "embedding", "meta", "response_event", "result", "skipped")

    def __init__(self, tile: Tile, embedding: np.ndarray, meta: Dict):
        self.tile = tile
        self.embedding = embedding
        self.meta = meta or {}
        self.response_event = threading.Event()
        self.result: Optional[tuple[str, bool]] = None   # (label, relevant) or None
        self.skipped: bool = False   # True if user chose "Skip / Move On" without assigning

    def set_result(self, label: Optional[str], relevant: bool = False):
        """Set a normal label result. label may be None only for internal cases."""
        self.result = (label, relevant) if label is not None else None
        self.response_event.set()

    def set_skip(self):
        """User chose to move on without labeling this tile (saves effort for large sets).
        The tile remains unlabeled in the DB (unless it already had a label).
        """
        self.skipped = True
        self.result = None
        self.response_event.set()

    def wait(self, timeout: Optional[float] = None) -> Optional[tuple[str, bool]]:
        self.response_event.wait(timeout)
        return self.result


class DroneAREDController:
    """
    Main orchestrator. Lives in the main thread's view but spawns workers.

    Typical usage (from GUI):
        ctrl = DroneAREDController(config)
        ctrl.set_label_store(store)
        ctrl.start()
        ...
        ctrl.pause()
        ctrl.stop()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()   # when set -> paused
        self._pause_event.set()  # start unpaused

        self.label_request_queue: queue.Queue[LabelRequest] = queue.Queue(maxsize=4)
        self.stats: Dict[str, Any] = {
            "frames_read": 0,
            "tiles_processed": 0,
            "ared_queries": 0,   # A/RED decided a label was needed for this point (the "queries" for QP/RR and user labels needed). Cache may satisfy without GUI.
            "user_queries": 0,   # times we actually popped the GUI (real human dialog this session)
            "cache_hits": 0,     # auto-labeled from previous sessions / earlier in run (embedding cache)
            "ared_clusters": 0,
            "ared_known_labels": 0,
            "current_video": "",
            "status": "idle",
        }

        # Components (created on start)
        self.tiler: Optional[GridTiler] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.ared_adapter: Optional[AREDAdapter] = None
        self.label_store: Optional[PersistentLabelStore] = None
        self.tile_db: Optional["TileAnnotationDB"] = None   # NEW: exact identity annotations
        self.annotation_manager: Optional["AnnotationManager"] = None  # service layer (full wiring)
        self._last_feature_extractor = None  # for restoring after adapter re-init / load

        self._worker_thread: Optional[threading.Thread] = None
        self._global_tile_counter = 0
        self._current_label_req: Optional[LabelRequest] = None

        # When True, even tiles with previous exact labels will be shown to the user for correction
        self.edit_mode: bool = False

        # "Label Only" mode: pure human labeling (no DINO, no A/RED).
        # Perfect for building reference labeled datasets needed for performance metrics
        # (Query Precision / Relevant Recall as defined in the A/RED papers).
        self.label_only_mode: bool = False

        # Live collection of tiles that A/RED actually decided to query (stable identities).
        # Populated during normal runs. Used by metrics.py for QP / RR calculation.
        self.queried_identities: List[Tuple] = []

        # All tiles actually sent to / processed by A/RED in this run (for accurate "should" computation
        # over only the tiles that were actually presented to the algorithm).
        self.processed_identities: List[Tuple] = []

        # ------------------------------------------------------------------
        # Label Only navigation state (back/forward/jump)
        # These allow the user to move around the video without being stuck
        # in a strict forward-only stream. This is especially useful in the
        # sparse/resume labeling workflow.
        # ------------------------------------------------------------------
        self._label_only_current_video: Optional[str] = None
        self._label_only_current_frame: int = 0
        self._label_only_current_tile_idx: int = 0   # index within the tiles of the current frame
        self._label_only_navigation_event = threading.Event()  # used to unblock when GUI requests nav
        self._label_only_nav_command: Optional[Dict] = None    # {'action': 'next'|'prev'|'jump', 'frame': int, ...}

        # Callback the GUI can register to receive periodic status updates
        self.on_stats: Optional[Callable[[Dict], None]] = None

    # ------------------------------------------------------------------
    # Public control surface (called from GUI)
    # ------------------------------------------------------------------
    def start(self, video_paths: Optional[List[str]] = None):
        if self._worker_thread and self._worker_thread.is_alive():
            print("[Controller] Already running")
            return

        if video_paths:
            self.config.video_paths = video_paths

        self._stop_event.clear()
        self._pause_event.set()

        # Reset per-run counters for a fresh processing pass (ARED state is kept if present)
        self._global_tile_counter = 0
        self.stats["tiles_processed"] = 0
        self.stats["frames_read"] = 0
        self.stats["ared_queries"] = 0
        self.stats["user_queries"] = 0
        self.stats["cache_hits"] = 0
        self.queried_identities = []
        self.processed_identities = []

        # Create components. We pass create_ared=False if we already have one (from Load ARED).
        create_ared = (self.ared_adapter is None)
        self._init_components(create_ared=create_ared)

        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="ared-worker")
        self._worker_thread.start()
        self.stats["status"] = "running"
        print("[Controller] Started processing thread")

    def pause(self):
        if not self._pause_event.is_set():
            return
        self._pause_event.clear()
        self.stats["status"] = "paused"
        print("[Controller] Paused")

    def resume(self):
        if self._pause_event.is_set():
            return
        self._pause_event.set()
        self.stats["status"] = "running"
        print("[Controller] Resumed")

    def stop(self, join_timeout: float = 3.0):
        self._stop_event.set()
        self._pause_event.set()  # unblock if paused

        # Unblock any pending label request (worker may be blocked in req.wait() for human label)
        if getattr(self, "_current_label_req", None):
            try:
                self._current_label_req.set_result("__STOPPED__", False)
            except Exception:
                pass
            self._current_label_req = None

        # Drain any still-queued label requests so they don't block future runs
        while True:
            try:
                req = self.label_request_queue.get_nowait()
                if req:
                    try:
                        req.set_result("__STOPPED__", False)
                    except Exception:
                        pass
            except queue.Empty:
                break

        if self._worker_thread:
            self._worker_thread.join(timeout=join_timeout)
            self._worker_thread = None
        self.stats["status"] = "stopped"
        # Keep queried_identities for post-run metrics; clear only on next start
        print("[Controller] Stopped")

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def set_label_store(self, store: PersistentLabelStore):
        self.label_store = store
        if self.ared_adapter:
            self.ared_adapter.set_label_store(store)

    def set_tile_database(self, db: Optional[TileAnnotationDB]):
        """NEW: exact-match annotation store (video + abs_frame + tile pos + resolution)."""
        self.tile_db = db

    def set_annotation_manager(self, manager: Optional["AnnotationManager"]):
        """Wired service layer for scoped queries, bulk ops, etc."""
        self.annotation_manager = manager
        if manager and not self.tile_db:
            self.tile_db = manager.db  # compat

    def set_edit_mode(self, enabled: bool):
        """When enabled, the label provider will show the GUI even for previously labeled exact tiles."""
        self.edit_mode = bool(enabled)
        print(f"[Controller] Edit mode = {self.edit_mode}")

    def set_label_only_mode(self, enabled: bool):
        """Pure labeling mode: every tile is shown for labeling, no A/RED or DINO involved."""
        self.label_only_mode = bool(enabled)
        print(f"[Controller] Label Only mode = {self.label_only_mode}")

    def get_queried_identities(self) -> List[Tuple]:
        """Return list of stable tile identities that A/RED actually queried in this run.
        Used for computing Query Precision and Relevant Recall (see A/RED papers).
        Note: cache-satisfied decisions are included (treated as user queries per requirements)."""
        return list(self.queried_identities)

    def get_processed_identities(self) -> List[Tuple]:
        """Return list of *all* tile identities that were sent to A/RED in this run.
        Used so that metric 'should_query' positives (firsts + relevant) only count
        tiles/classes that were actually presented to A/RED (not extra labels from other sessions)."""
        return list(self.processed_identities)

    def get_labels_from_annotation_db(self) -> List[str]:
        """Return unique labels from the exact TileAnnotationDB.
        This populates the main GUI "Discovered Classes" list with classes
        that were labeled during sparse/resume Label Only sessions (even if
        we only labeled the relevant ones and skipped most tiles).
        """
        if self.tile_db:
            return self.tile_db.get_all_labels()
        return []

    # ------------------------------------------------------------------
    # Label Only navigation API (back / forward / jump to frame)
    # Called from the GUI. These update the cursor and signal the worker
    # loop so the user can move around the video freely.
    # This is essential when using the sparse "label relevant + skip the rest"
    # workflow so the user can go back if they miss a tile.
    # ------------------------------------------------------------------
    def label_only_next(self):
        """Request move to the next tile in the labeling sequence."""
        if not self.label_only_mode:
            return
        self._label_only_nav_command = {"action": "next"}
        self._label_only_navigation_event.set()

    def label_only_prev(self):
        """Request move to the previous tile."""
        if not self.label_only_mode:
            return
        self._label_only_nav_command = {"action": "prev"}
        self._label_only_navigation_event.set()

    def label_only_jump_to_frame(self, frame: int):
        """Jump directly to a specific absolute frame (0-based from start of video).
        The worker will seek and present the first valid tile on that frame.
        """
        if not self.label_only_mode:
            return
        self._label_only_nav_command = {"action": "jump", "frame": max(0, int(frame))}
        self._label_only_navigation_event.set()

    def compute_metrics_for_video(self, video_path: str) -> Dict[str, Any]:
        """Compute Query Precision / Relevant Recall using current DB labels + logged queries.

        This implements the evaluation methodology from:
          - IJSC_2026-1.pdf : "Real-Time Memory-Bounded A/RED"
          - SPIE_IVSP_2026.pdf : "Shallow vs. Deep Features for A/RED"
        """
        if self.annotation_manager:
            anns = self.annotation_manager.get_annotations(video=video_path, use_scope=False)
        elif self.tile_db is not None:
            anns = self.tile_db.get_annotations_for_video(video_path)
        else:
            return {"error": "No tile database loaded"}

        if not anns and self.tile_db is not None:
            # Try matching by basename
            base = Path(video_path).name
            for v in self.tile_db.list_videos():
                if Path(v).name == base:
                    if self.annotation_manager:
                        anns = self.annotation_manager.get_annotations(video=v, use_scope=False)
                    else:
                        anns = self.tile_db.get_annotations_for_video(v)
                    video_path = v
                    break

        if not anns:
            return {"error": f"No annotations for {video_path}"}

        queried = self.get_queried_identities()
        processed = self.get_processed_identities()
        labeled_total = len(anns)

        # Use the number of tiles actually sent to A/RED for this run.
        # This ensures "should_query" (firsts + relevant) only counts tiles that were
        # actually presented ("sent") to A/RED, not extraneous labels from other runs/sessions.
        stream_total = len(processed) if processed else self.stats.get("tiles_processed", labeled_total)
        ared_query_count = self.stats.get("ared_queries", len(queried))

        # Collect run parameters so metrics reports are reproducible (kappa, tile size, stride, DB, model, etc.)
        run_params = self._collect_run_params()

        result = ared_metrics.evaluate_from_annotations_and_queries(
            anns, queried, 
            total_points=stream_total,
            ared_query_count_override=ared_query_count,
            processed_keys=processed,
            run_params=run_params,
        )
        result["video"] = video_path
        result["n_labeled"] = labeled_total
        result["total_stream_tiles"] = stream_total
        result["ared_queries_made"] = ared_query_count
        result["n_processed_in_run"] = len(processed)
        # Ensure top-level visibility of key params even if caller inspects the flat dict
        if run_params:
            result["run_params"] = run_params
            # Convenience top-level aliases for the most requested items
            result["kappa"] = run_params.get("kappa")
            result["tile_size"] = run_params.get("tile_size")
            result["frame_stride"] = run_params.get("frame_stride")
            result["annotation_db"] = run_params.get("annotation_db")
            result["dino_model"] = run_params.get("dino_model")
        return result

    def _collect_run_params(self) -> Dict[str, Any]:
        """Snapshot key experiment settings at metrics computation time."""
        p: Dict[str, Any] = {}
        try:
            t = self.config.tiling
            a = self.config.ared
            f = self.config.features
            ta = self.config.tile_annotations

            p["kappa"] = float(a.kappa)
            p["tile_size"] = (int(t.tile_width), int(t.tile_height))
            p["frame_stride"] = int(t.frame_stride)
            p["stride_x"] = int(t.stride_x) if t.stride_x is not None else int(t.tile_width)
            p["stride_y"] = int(t.stride_y) if t.stride_y is not None else int(t.tile_height)

            # Which annotation DB was active
            if self.tile_db is not None:
                p["annotation_db"] = str(getattr(self.tile_db, "db_path", "?"))
            else:
                p["annotation_db"] = getattr(ta, "db_path", None) or "?"

            # DINO / feature config
            p["dino_model"] = getattr(f, "model_name", None)
            p["dino_normalize"] = getattr(f, "normalize", None)
            p["dino_pooling"] = getattr(f, "pooling", None)

            # A/RED core knobs
            p["l_buf_size"] = int(a.l_buf_size)
            p["k_comp_pts"] = int(a.k_comp_pts)
            p["qs_var"] = int(a.qs_var)
            p["nghbhood_merge"] = bool(a.nghbhood_merge)
            p["singleton_merge"] = bool(a.singleton_merge)
            p["data_augmentation_enabled"] = bool(getattr(a, "data_augmentation_enabled", False))

            # Label cache
            lc = self.config.label_cache
            p["label_cache_enabled"] = bool(lc.enabled)
            p["label_cache_threshold"] = float(getattr(lc, "auto_label_threshold", 0.0))

            # Misc
            p["edit_mode"] = bool(getattr(self, "edit_mode", False))
            p["label_only_mode"] = bool(getattr(self, "label_only_mode", False))
        except Exception as e:
            p["collect_error"] = str(e)
        return p

    def update_config(self, new_config: PipelineConfig):
        """Apply new parameters for the *next* start()."""
        self.config = new_config

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _init_components(self, create_ared: bool = True):
        """Initialize (or re-initialize) pipeline components.
        If create_ared=False, we preserve an existing ared_adapter (e.g. from Load ARED Model).
        """
        tcfg = self.config.tiling
        self.tiler = GridTiler(
            tile_width=tcfg.tile_width,
            tile_height=tcfg.tile_height,
            stride_x=tcfg.stride_x,
            stride_y=tcfg.stride_y,
        )

        if not self.label_only_mode:
            fcfg = self.config.features
            self.feature_extractor = DINOFeatureExtractor(
                model_name=fcfg.model_name,
                device=fcfg.device,
                normalize=fcfg.normalize,
                pooling=fcfg.pooling,
                batch_size=fcfg.batch_size,
            )
            self._last_feature_extractor = self.feature_extractor
        else:
            self.feature_extractor = None
            self._last_feature_extractor = None

        if (create_ared or self.ared_adapter is None) and not self.label_only_mode:
            self.ared_adapter = AREDAdapter(self.config.ared)
            if self.label_store:
                self.ared_adapter.set_label_store(self.label_store)
            if self.feature_extractor:
                self.ared_adapter.set_feature_extractor(self.feature_extractor)
        elif self.label_only_mode:
            self.ared_adapter = None

        # (Re)wire provider every time (the closure must see current stores etc.)
        def _gui_label_provider(emb: np.ndarray, tile_img: Any, meta: Dict):
            """
            Called (via the adapter shim) only when A/RED has decided this point needs a label.

            Priority order for answering (so we honor previous human work):
            1. Exact identity match from TileAnnotationDB (video + abs_frame + row/col + resolution).
               - Unless self.edit_mode is True, in which case we still show GUI for correction.
            2. Old embedding similarity cache (PersistentLabelStore).
            3. Human via GUI (persistent LabelingDialog).

            After a human decision we save BOTH to the exact DB (for perfect future recall)
            and to the embedding store (for "similar appearance" on new content).
            """
            meta = meta or {}
            print(f"[Pipeline] _gui_label_provider called for tile: {meta} (emb shape: {emb.shape if hasattr(emb,'shape') else 'N/A'})")

            # --- 1. EXACT IDENTITY LOOKUP (new primary mechanism) ---
            vpath = meta.get("video_path") or meta.get("video") or ""
            abs_f = meta.get("abs_frame", meta.get("frame", -1))
            row = meta.get("row", meta.get("tile_row", 0))
            col = meta.get("col", meta.get("tile_col", 0))
            tw = meta.get("tile_width", 0) or getattr(tile_img, 'width', 0) if tile_img else 0
            th = meta.get("tile_height", 0) or getattr(tile_img, 'height', 0) if tile_img else 0

            if self.tile_db is not None and vpath and abs_f >= 0:
                key = TileKey(vpath, int(abs_f), int(row), int(col), int(tw or 0), int(th or 0))
                exact = self.tile_db.lookup_key(key)
                if exact:
                    label, rel = exact
                    if not self.edit_mode:
                        print(f"[Pipeline]   -> EXACT DB HIT for {Path(vpath).name} f{abs_f} [{row},{col}]: '{label}' (relevant={rel}). Auto (no GUI).")
                        if self.label_store is not None:
                            try:
                                self.label_store.add(emb, label, rel)
                            except Exception:
                                pass
                        return label, rel
                    else:
                        print(f"[Pipeline]   -> EXACT DB HIT but EDIT MODE active -> forcing GUI for correction.")

            # --- 2. FALLBACK: embedding similarity cache (existing behavior) ---
            if self.label_store is not None:
                hit = self.label_store.lookup(emb)
                if hit:
                    self.stats["cache_hits"] = self.stats.get("cache_hits", 0) + 1
                    print(f"[Pipeline]   -> Label store (embedding) CACHE HIT: '{hit[0]}' (relevant={hit[1]}). Returning without GUI.")
                    return hit
                else:
                    print(f"[Pipeline]   -> Label store (embedding) CACHE MISS.")

            # --- 3. Real human labeling via GUI (only when A/RED asked or edit mode) ---
            self.stats["user_queries"] = self.stats.get("user_queries", 0) + 1
            print(f"[Pipeline]   -> Requesting HUMAN label via GUI (actual human this session now {self.stats['user_queries']}; ared_queries already counted the decision)  meta={meta}")

            # Ensure the tile object carries identity for the dialog / later saving
            safe_tile = tile_img
            if safe_tile is None or not hasattr(safe_tile, 'image'):
                safe_tile = Tile(image=Image.new("RGB", (64, 64)), frame_idx=int(abs_f) if abs_f >= 0 else 0,
                                 tile_row=row, tile_col=col, bbox=(0, 0, tw or 64, th or 64),
                                 video_path=vpath)

            req = LabelRequest(tile=safe_tile, embedding=emb, meta=meta)
            self._current_label_req = req
            try:
                self.label_request_queue.put(req, timeout=5)
            except queue.Full:
                self._current_label_req = None
                print("[Pipeline]   -> WARNING: label queue full, falling back to __BACKGROUND__")
                return "__BACKGROUND__", False

            print("[Pipeline]   -> Waiting for GUI response (blocking worker thread)...")
            result = req.wait(timeout=300)
            self._current_label_req = None
            if result is None:
                print("[Pipeline]   -> TIMEOUT waiting for label, using __TIMEOUT__")
                return "__TIMEOUT__", False

            label, rel = result
            print(f"[Pipeline]   -> Received label from GUI: '{label}' (relevant={rel})")

            # Save the human decision to BOTH stores
            # a) Exact DB (for perfect recall by identity, stride-independent, editable)
            if self.tile_db is not None and vpath and abs_f >= 0:
                try:
                    bx = meta.get("bbox", (col * tw, row * th, col * tw + tw, row * th + th))
                    cx, cy = bx[0], bx[1]
                    key = TileKey(vpath, int(abs_f), int(row), int(col), int(tw or 0), int(th or 0))
                    self.tile_db.set_annotation_for_key(key, label, rel, embedding=emb, crop_x=cx, crop_y=cy)
                except Exception as e:
                    print(f"[Pipeline]   WARNING: failed to save to TileAnnotationDB: {e}")

            # b) Embedding similarity (for "looks like" on future novel tiles)
            if self.label_store is not None:
                try:
                    self.label_store.add(emb, label, rel)
                    print(f"[Pipeline]   -> Also added to embedding similarity cache.")
                except Exception as e:
                    print(f"[Pipeline]   -> WARNING: failed to add to label_store: {e}")

            return label, rel

        if self.ared_adapter:
            self.ared_adapter.set_label_provider(_gui_label_provider)

            # Forward feature extractor for DINO data augmentation (rotations)
            if self.feature_extractor:
                self.ared_adapter.set_feature_extractor(self.feature_extractor)

        # Make sure the provider closure can see the current tile_db and edit_mode
        # (already closed over self)

    def _run_loop(self):
        """Main worker loop: for each video, for selected frames, tile -> embed -> ARED."""
        print("[Pipeline] Worker thread started. Processing will log each tile and any ARED queries.")
        self.stats["status"] = "running"
        videos = self.config.video_paths or []

        try:
            for vpath in videos:
                if self._stop_event.is_set():
                    break
                self.stats["current_video"] = Path(vpath).name
                self._process_one_video(vpath)

            self.stats["status"] = "finished"
            print(f"[Pipeline] All videos finished. Final stats: {self.stats}")
            if self.on_stats:
                self.on_stats(self.stats.copy())
        except Exception as e:
            print("[Pipeline] FATAL ERROR in worker thread (this would cause silent freeze):", e)
            import traceback
            traceback.print_exc()
            self.stats["status"] = "error"
            if self.on_stats:
                try:
                    self.on_stats(self.stats.copy())
                except Exception:
                    pass

    def _process_one_video(self, video_path: str):
        if not HAS_CV2:
            print("[Controller] cv2 not available - cannot read video.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Controller] Failed to open {video_path}")
            return

        print(f"[Pipeline] Starting video: {video_path}")

        if self.label_only_mode:
            # Delegate entirely to the dedicated label-only processor.
            # It handles its own video reading, stride, navigation (back/forward/jump),
            # resume skipping, and labeling. This avoids interference with the
            # normal A/RED frame-batching loop.
            self._process_label_only_tiles(video_path)
            print(f"[Pipeline] Finished label-only for video: {video_path}")
            return

        frame_idx = -1
        frame_stride = max(1, self.config.tiling.frame_stride)
        batch_imgs: List[Image.Image] = []
        batch_tiles: List[Tile] = []

        try:
            try:
                while not self._stop_event.is_set():
                    # Pause handling
                    self._pause_event.wait()  # blocks while paused
                    if self._stop_event.is_set():
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    self.stats["frames_read"] = frame_idx + 1

                    if frame_idx % frame_stride != 0:
                        continue

                    print(f"[Pipeline] Processing frame {frame_idx} (stride={frame_stride})")

                    # Convert BGR (cv2) -> RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Tile - attach full identity (video + absolute frame). This is crucial
                    # so that the exact label DB can recall previous human decisions even when
                    # frame_stride changes between runs.
                    tiles = self.tiler.tile_frame(frame_rgb, frame_idx, self._global_tile_counter,
                                                  video_path=video_path)
                    if not tiles:
                        continue

                    # Note: label_only_mode is handled via early delegation above; this branch
                    # is only for the normal A/RED + DINO path.

                    # Normal A/RED path: Collect for batched feature extraction
                    for t in tiles:
                        batch_tiles.append(t)
                        batch_imgs.append(t.image)
                        self._global_tile_counter += 1

                        # Flush batch when full or at end of logical group
                        if len(batch_imgs) >= self.config.features.batch_size:
                            self._process_batch(batch_tiles, batch_imgs)
                            batch_tiles = []
                            batch_imgs = []

                    # Also flush small remainder periodically
                    if len(batch_imgs) >= 4:
                        self._process_batch(batch_tiles, batch_imgs)
                        batch_tiles = []
                        batch_imgs = []

                    # Throttle a little so GUI can keep up
                    time.sleep(0.001)

                # Final flush
                if batch_imgs:
                    self._process_batch(batch_tiles, batch_imgs)

                print(f"[Pipeline] Finished video: {video_path}. Total tiles processed so far: {self.stats['tiles_processed']}")

            finally:
                cap.release()
        except Exception as e:
            print(f"[Pipeline] ERROR while processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()

    def _process_batch(self, tiles: List[Tile], pil_images: List[Image.Image]):
        if not tiles or not self.feature_extractor or not self.ared_adapter:
            return

        try:
            embs = self.feature_extractor.extract_images(pil_images)

            for tile, emb in zip(tiles, embs):
                print(f"[Pipeline->ARED] passing new tile global={tile.global_idx} (frame={tile.frame_idx}, r={tile.tile_row}, c={tile.tile_col}) to A_RED")
                # Rich identity for the exact TileAnnotationDB (primary) + backward compat for embedding cache
                meta = {
                    "video_path": getattr(tile, 'video_path', '') or video_path,
                    "frame": tile.frame_idx,
                    "abs_frame": tile.frame_idx,   # always the true video frame number
                    "row": tile.tile_row,
                    "col": tile.tile_col,
                    "tile_width": tile.width,
                    "tile_height": tile.height,
                    "bbox": tile.bbox,
                }
                info = self.ared_adapter.process(emb, tile_image=tile, meta=meta)
                self.stats["tiles_processed"] += 1

                ared_queried = info.get("queried", False)
                label = info.get("label")

                # Always record every tile sent to A/RED (for accurate filtering of "should" positives
                # to only tiles/classes actually presented to A/RED this run).
                pkey = ared_metrics.tile_identity_from_meta(meta, tile)
                if pkey:
                    self.processed_identities.append(pkey)
                    if ared_queried:
                        self.queried_identities.append(pkey)

                if ared_queried:
                    self.stats["ared_queries"] = self.stats.get("ared_queries", 0) + 1
                    # Record was already done above for queried list.
                    # "ared_queried" means A/RED's internal decision (Query_Cdn met).
                    # Cache hits still count fully (as user queries per spec).
                    # See drone_ared/metrics.py and the A/RED papers (IJSC_2026-1, SPIE_IVSP_2026).

                # Always log finish of tile processing so we can see forward progress even when A/RED is not querying.
                # This is key to confirm that "Waiting for next query" in the GUI just means A/RED chose not to ask for a label.
                print(f"[Pipeline] Finished tile global={tile.global_idx} (frame={tile.frame_idx}, r={tile.tile_row}, c={tile.tile_col}). "
                      f"ARED queried? {ared_queried}  label='{label}'  clusters={info.get('num_clusters', '?')} known_labels={info.get('num_known_labels', '?')}")

                # "queried" here means A/RED decided it needed a label (cache or human).
                # We track user_queries separately when we actually show the dialog.

                # Update ARED internal state for GUI display
                if self.ared_adapter:
                    self.stats["ared_clusters"] = self.ared_adapter.num_clusters
                    self.stats["ared_known_labels"] = self.ared_adapter.num_known_labels

                # Push stats to GUI occasionally
                if self.on_stats and (self.stats["tiles_processed"] % 8 == 0):
                    self.on_stats(self.stats.copy())

                # Progress heartbeat (uses reliable stats counter so it fires even for long non-query stretches)
                if (self.stats["tiles_processed"] % 10 == 0) or ared_queried:
                    print(f"[Pipeline] Progress: tiles_processed={self.stats['tiles_processed']}, "
                          f"ared_clusters={self.stats.get('ared_clusters', 0)}, "
                          f"ared_queries (labels needed)={self.stats.get('ared_queries', 0)}, "
                          f"human_dialogs={self.stats.get('user_queries', 0)}, "
                          f"cache_hits={self.stats.get('cache_hits', 0)}")
        except Exception as e:
            print(f"[Pipeline] ERROR in _process_batch: {e}")
            import traceback
            traceback.print_exc()

    def _process_label_only_tiles(self, video_path: str):
        """Pure labeling path (no DINO, no A/RED) with full back/forward/jump navigation.

        Supports the user request for non-linear navigation in Label Only mode
        so they can go back if they miss a tile or jump to a specific frame.
        Combined with Skip + resume logic.

        IMPORTANT FIXES APPLIED:
        - No longer receives/depends on a caller-provided tiles list (was always [] from delegation).
        - Cursor advancement now uses the *actual* number of tiles on each frame instead of
          hardcoded sentinels ( >500 / =999 ). This was the direct cause of:
            * "frame 1, tile 300" (internal tile_idx grew unbounded while clamping selection)
            * re-processing the same frame/tile repeatedly (clamped always to last tile)
            * never advancing frames visibly until hundreds of steps
        - Auto-skip (resume) of already-labeled tiles is throttled lightly to avoid CPU spin
          on video re-decode + re-tile for long pre-labeled stretches.
        - Frame/tile progress is now correct so the LabelingDialog receives sequential distinct
          tiles and the info bar shows advancing "Frame X | Tile rY cZ".

        Key behaviors (designed for large numbers of tiles and Relevant Recall measurement):
        - Checks the exact TileAnnotationDB first.
        - If the tile already has a label:
            - If edit_mode is False: automatically skip (resume support). We do not re-show.
            - If edit_mode is True: show the dialog pre-filled so user can correct or explicitly skip.
        - For tiles without a label (or in edit): show dialog.
        - Dialog provides:
            - Normal label assignment (including "relevant" flag).
            - "Skip / Move On" : do not assign anything now. Tile stays unlabeled in DB.
              This lets you quickly label only the relevant class instances (e.g. every "person")
              and defer background / normal tiles.
        - Unlabeled tiles (never assigned or previously skipped) will be shown again on resume
          unless you labeled them.
        - All decisions are saved with full identity so metrics (QP/RR) and future A/RED runs
          can use them perfectly, even across different frame strides.
        - Integrates with global edit_mode for corrections.

        This directly supports the metric definitions in the papers:
        Relevant Recall requires knowing (and not missing) instances of relevant classes.
        You can do a fast pass labeling only the relevant ones, then compute metrics
        against A/RED's query decisions.
        """
        if self.tile_db is None:
            print("[Pipeline] Label Only mode: No tile_db configured. Labels will not be saved.")

        # Initialize navigation cursor state for this video
        self._label_only_current_video = video_path
        self._label_only_current_frame = 0
        self._label_only_current_tile_idx = 0
        self._label_only_nav_command = None
        self._label_only_navigation_event.clear()
        self._label_only_force_present = False  # when True (after explicit nav), present even labeled tiles (for review)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Pipeline] Label Only: could not open {video_path} for seeking/navigation")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 999999)
        frame_stride = max(1, self.config.tiling.frame_stride)

        # Probe once to learn how many tiles the GridTiler produces for frames of this video.
        # This is constant for uniform grid + fixed video resolution (typical case).
        # We use it for proper modular wrap-around instead of magic numbers.
        tiles_per_frame = 1
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, probe = cap.read()
            if ret and probe is not None:
                prgb = cv2.cvtColor(probe, cv2.COLOR_BGR2RGB)
                probe_tiles = self.tiler.tile_frame(prgb, 0, 0, video_path=video_path)
                if probe_tiles:
                    tiles_per_frame = len(probe_tiles)
        except Exception:
            pass
        if tiles_per_frame < 1:
            tiles_per_frame = 1

        # Reset seek head
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        auto_skip_count = 0

        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                # --- Handle navigation commands from the GUI (Prev/Next/Jump) ---
                # These are ONLY active for label-only mode (see controller guards + GUI only
                # wires nav buttons/binds when allow_skip=True which label-only sets in meta).
                # A/RED runs never set allow_skip and never create these controls.
                if self._label_only_nav_command is not None:
                    cmd = self._label_only_nav_command
                    self._label_only_nav_command = None
                    self._label_only_navigation_event.clear()
                    print(f"[Pipeline] Label Only: applying nav command {cmd} (cursor before: f{self._label_only_current_frame} t{self._label_only_current_tile_idx})")

                    if cmd.get("action") == "next":
                        self._advance_label_only_cursor(1, frame_stride, total_frames, tiles_per_frame)
                    elif cmd.get("action") == "prev":
                        self._advance_label_only_cursor(-1, frame_stride, total_frames, tiles_per_frame)
                    elif cmd.get("action") == "jump":
                        self._label_only_current_frame = max(0, min(cmd.get("frame", 0), total_frames-1))
                        self._label_only_current_tile_idx = 0

                    print(f"[Pipeline] Label Only: nav applied, now at f{self._label_only_current_frame} t{self._label_only_current_tile_idx}")

                    # Mark that the *next* tile we land on should be presented even if already
                    # labeled (so user can review/edit previous tiles via explicit nav).
                    self._label_only_force_present = True

                    # Unblock any pending label wait so we can reposition immediately
                    if self._current_label_req is not None:
                        try:
                            self._current_label_req.set_skip()
                        except Exception:
                            pass
                        self._current_label_req = None

                    # Small yield so GUI can breathe after a nav jump
                    time.sleep(0.01)

                # Seek + decode the frame indicated by current cursor
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._label_only_current_frame)
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Reached end or unreadable frame; stop for this video
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tiles_on_frame = self.tiler.tile_frame(frame_rgb, self._label_only_current_frame,
                                                       0, video_path=video_path)
                if not tiles_on_frame:
                    # No tiles possible on this frame (very small resolution?); move forward
                    self._label_only_force_present = False
                    self._advance_label_only_cursor(1, frame_stride, total_frames, tiles_per_frame)
                    continue

                N = len(tiles_on_frame)

                # CRITICAL: normalize cursor tile index to the *actual* count for this frame.
                # This prevents the previous unbounded growth and "always last tile" clamping bug.
                if self._label_only_current_tile_idx >= N or self._label_only_current_tile_idx < 0:
                    self._label_only_current_tile_idx = max(0, min(self._label_only_current_tile_idx, N-1))

                t_idx = self._label_only_current_tile_idx
                tile = tiles_on_frame[t_idx]

                # Resume / already-labeled skip logic (exact DB identity)
                existing = None
                if self.tile_db is not None:
                    try:
                        key = TileKey(video_path, tile.frame_idx, tile.tile_row, tile.tile_col,
                                      tile.width, tile.height)
                        existing = self.tile_db.lookup_key(key)
                    except Exception:
                        existing = None

                # If an explicit navigation (prev/next/jump) just landed us here, force-present
                # the tile (with prefill if it had a prior label) so the user can review it.
                # Normal forward auto-resume still skips known tiles (unless edit_mode).
                force_present = getattr(self, '_label_only_force_present', False)
                self._label_only_force_present = False

                if existing and not self.edit_mode and not force_present:
                    # Auto-skip: do not bother user; just advance.
                    # Throttle to keep CPU/decode reasonable during long resume passes over
                    # already-labeled content. Without this, tight loop of seek+retile+lookup
                    # made "labeling mode" unacceptably slow even when skipping.
                    self.stats["tiles_processed"] = self.stats.get("tiles_processed", 0) + 1
                    auto_skip_count += 1
                    self._advance_label_only_cursor(1, frame_stride, total_frames, tiles_per_frame)

                    if auto_skip_count % 20 == 0 and self.on_stats:
                        self.on_stats(self.stats.copy())
                    time.sleep(0.002)  # yield; prevents 100% CPU spin on video decode for skips
                    continue

                # Reset skip counter when we actually present something
                auto_skip_count = 0

                # Present this tile to the (persistent) labeling dialog
                req = LabelRequest(tile=tile, embedding=np.zeros(1, dtype=np.float32), meta={
                    "video_path": video_path,
                    "frame": tile.frame_idx,
                    "abs_frame": tile.frame_idx,
                    "row": tile.tile_row,
                    "col": tile.tile_col,
                    "tile_width": tile.width,
                    "tile_height": tile.height,
                    "bbox": tile.bbox,
                    "label_only": True,
                    "allow_skip": True,
                    "current_label": existing[0] if existing else None,
                    "current_relevant": existing[1] if existing else False,
                })
                self._current_label_req = req

                try:
                    self.label_request_queue.put(req, timeout=5)
                except queue.Full:
                    self._current_label_req = None
                    if self._label_only_nav_command is None:
                        self._advance_label_only_cursor(1, frame_stride, total_frames, tiles_per_frame)
                    continue

                result = req.wait(timeout=300)
                self._current_label_req = None

                # If a navigation command arrived while we were waiting for a decision on this tile,
                # the GUI already did set_skip() + signaled the desired move. We suppress the normal
                # "skip means +1 forward" here so the nav delta (next/prev/jump) is the only one applied.
                # This prevents double-advancing on Next or fighting Prev.
                # The top-of-loop nav handler (or the one below) will have applied the correct cursor change.
                nav_pending_on_unblock = (self._label_only_nav_command is not None)

                if getattr(req, "skipped", False) or result is None:
                    self.stats["tiles_processed"] = self.stats.get("tiles_processed", 0) + 1
                    if not nav_pending_on_unblock:
                        self._advance_label_only_cursor(1, frame_stride, total_frames, tiles_per_frame)
                    continue

                label, rel = result
                if not label or label == "__SKIP__":
                    self.stats["tiles_processed"] = self.stats.get("tiles_processed", 0) + 1
                    if not nav_pending_on_unblock:
                        self._advance_label_only_cursor(1, frame_stride, total_frames, tiles_per_frame)
                    continue

                # Save the (new or corrected) label to the exact DB
                if self.tile_db is not None:
                    try:
                        cx, cy = tile.bbox[0], tile.bbox[1]
                        key = TileKey(video_path, tile.frame_idx, tile.tile_row, tile.tile_col,
                                      tile.width, tile.height)
                        self.tile_db.set_annotation_for_key(key, label, rel, embedding=None, crop_x=cx, crop_y=cy)
                    except Exception as e:
                        print(f"[Pipeline] Label Only save error: {e}")

                self.stats["tiles_processed"] = self.stats.get("tiles_processed", 0) + 1
                if self.on_stats and (self.stats["tiles_processed"] % 3 == 0):
                    self.on_stats(self.stats.copy())

                # Normal forward advance after a user *label assignment* decision.
                # (Skips go through earlier paths that may suppress when nav is active.)
                if self._label_only_nav_command is None:
                    self._advance_label_only_cursor(1, frame_stride, total_frames, tiles_per_frame)

        finally:
            cap.release()

    def _advance_label_only_cursor(self, delta: int, stride: int, total_frames: int, tiles_per_frame: int = 30):
        """Internal helper to move the labeling cursor (used by both normal flow and nav commands).

        FIXED: Now performs proper carry/borrow using the real tiles_per_frame for the video
        instead of the previous hardcoded >500 / 999 sentinels. This was the root cause of
        the "re-running same frame", "tile 300", and failure to report/advance frames.
        """
        if tiles_per_frame < 1:
            tiles_per_frame = 1

        self._label_only_current_tile_idx += delta

        # Carry/borrow across frames using the actual tile count (wraps tile_idx correctly)
        while self._label_only_current_tile_idx < 0:
            self._label_only_current_frame -= stride
            self._label_only_current_tile_idx += tiles_per_frame

        while self._label_only_current_tile_idx >= tiles_per_frame:
            self._label_only_current_frame += stride
            self._label_only_current_tile_idx -= tiles_per_frame

        # Final safety clamps
        if self._label_only_current_frame < 0:
            self._label_only_current_frame = 0
            self._label_only_current_tile_idx = 0
        if self._label_only_current_frame >= total_frames:
            self._label_only_current_frame = max(0, total_frames - 1)
            self._label_only_current_tile_idx = max(0, tiles_per_frame - 1)

        self._label_only_current_frame = max(0, min(self._label_only_current_frame, total_frames - 1))
        self._label_only_current_tile_idx = max(0, self._label_only_current_tile_idx)
