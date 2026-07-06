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
from typing import Optional, List, Callable, Any, Dict, Tuple
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
from .tile_database import TileAnnotationDB  # NEW exact identity DB
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
            "user_queries": 0,   # times we actually popped the GUI (real human work)
            "cache_hits": 0,     # auto-labeled from previous sessions / earlier in run
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
        self.stats["user_queries"] = 0
        self.stats["cache_hits"] = 0
        self.queried_identities = []

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
        Used for computing Query Precision and Relevant Recall (see A/RED papers)."""
        return list(self.queried_identities)

    def get_labels_from_annotation_db(self) -> List[str]:
        """Return unique labels from the exact TileAnnotationDB.
        This populates the main GUI "Discovered Classes" list with classes
        that were labeled during sparse/resume Label Only sessions (even if
        we only labeled the relevant ones and skipped most tiles).
        """
        if self.tile_db:
            return self.tile_db.get_all_labels()
        return []

    def compute_metrics_for_video(self, video_path: str) -> Dict[str, Any]:
        """Compute Query Precision / Relevant Recall using current DB labels + logged queries.

        This implements the evaluation methodology from:
          - IJSC_2026-1.pdf : "Real-Time Memory-Bounded A/RED"
          - SPIE_IVSP_2026.pdf : "Shallow vs. Deep Features for A/RED"
        """
        if self.tile_db is None:
            return {"error": "No tile database loaded"}

        anns = self.tile_db.get_annotations_for_video(video_path)
        if not anns:
            # Try matching by basename (DB stores resolved paths)
            base = Path(video_path).name
            for v in self.tile_db.list_videos():
                if Path(v).name == base:
                    anns = self.tile_db.get_annotations_for_video(v)
                    video_path = v
                    break

        if not anns:
            return {"error": f"No annotations for {video_path}"}

        queried = self.get_queried_identities()
        total = len(anns)

        result = ared_metrics.evaluate_from_annotations_and_queries(anns, queried, total_points=total)
        result["video"] = video_path
        result["n_labeled"] = total
        return result

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
                exact = self.tile_db.lookup(vpath, abs_f, row, col, tw, th)
                if exact:
                    label, rel = exact
                    if not self.edit_mode:
                        print(f"[Pipeline]   -> EXACT DB HIT for {Path(vpath).name} f{abs_f} [{row},{col}]: '{label}' (relevant={rel}). Auto (no GUI).")
                        # Still feed embedding store for future similarity if present
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
            print(f"[Pipeline]   -> Requesting HUMAN label via GUI (user_queries now {self.stats['user_queries']})  meta={meta}")

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
                    self.tile_db.set_annotation(vpath, abs_f, row, col, tw, th, label, rel,
                                                embedding=emb, crop_x=cx, crop_y=cy)
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

                    if self.label_only_mode:
                        # Pure labeling mode: no DINO, no A/RED.
                        # Every tile (at stride) is presented for human labeling and saved.
                        self._process_label_only_tiles(tiles, video_path)
                        continue

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

                if ared_queried:
                    # Record stable identity for metrics (Query Precision / Relevant Recall).
                    # See drone_ared/metrics.py and the A/RED papers (IJSC_2026-1, SPIE_IVSP_2026).
                    qkey = ared_metrics.tile_identity_from_meta(meta, tile)
                    if qkey:
                        self.queried_identities.append(qkey)

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
                          f"user_queries={self.stats.get('user_queries', 0)}, "
                          f"cache_hits={self.stats.get('cache_hits', 0)}")
        except Exception as e:
            print(f"[Pipeline] ERROR in _process_batch: {e}")
            import traceback
            traceback.print_exc()

    def _process_label_only_tiles(self, tiles: List[Tile], video_path: str):
        """Pure labeling path (no DINO, no A/RED) with resume / sparse support.

        This is the "alternative" efficient labeling mode for metrics-focused work.

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
        if not tiles:
            return
        if self.tile_db is None:
            print("[Pipeline] Label Only mode: No tile_db configured. Labels will not be saved. "
                  "All tiles will be treated as new.")

        for tile in tiles:
            if self._stop_event.is_set():
                break

            # Pause support
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            vpath = video_path
            f = tile.frame_idx
            r = tile.tile_row
            c = tile.tile_col
            tw = tile.width
            th = tile.height

            # === Resume / sparse logic ===
            existing = None
            if self.tile_db is not None:
                try:
                    existing = self.tile_db.lookup(vpath, f, r, c, tw, th)
                except Exception:
                    existing = None

            if existing and not self.edit_mode:
                # Already labeled in a previous session (or earlier in this run).
                # Skip automatically to save effort. This is the core of "resume".
                print(f"[Pipeline] Label Only: already labeled as '{existing[0]}' (relevant={existing[1]}). Skipping.")
                self.stats["tiles_processed"] = self.stats.get("tiles_processed", 0) + 1
                continue

            if existing and self.edit_mode:
                print(f"[Pipeline] Label Only (EDIT): re-showing already labeled tile '{existing[0]}' for correction.")

            print(f"[Pipeline] Label Only: requesting label/skip for frame={f} r={r} c={c} "
                  f"({'edit' if self.edit_mode else 'new/resume'})")

            # Build request. We pass "label_only" and "allow_skip" so the GUI knows
            # to show the Skip button and pre-fill if we have an existing label.
            req = LabelRequest(tile=tile, embedding=np.zeros(1, dtype=np.float32), meta={
                "video_path": vpath,
                "frame": f,
                "abs_frame": f,
                "row": r,
                "col": c,
                "tile_width": tw,
                "tile_height": th,
                "bbox": tile.bbox,
                "label_only": True,
                "allow_skip": True,                    # enables Skip button in dialog
                "current_label": existing[0] if existing else None,
                "current_relevant": existing[1] if existing else False,
            })
            self._current_label_req = req
            try:
                self.label_request_queue.put(req, timeout=5)
            except queue.Full:
                self._current_label_req = None
                print("[Pipeline] Label Only: queue full, skipping tile")
                continue

            result = req.wait(timeout=300)
            self._current_label_req = None

            if getattr(req, 'skipped', False) or result is None:
                # User explicitly chose "Skip / Move On" or timeout.
                # Do NOT write anything to DB. Tile remains unlabeled for future passes.
                print(f"[Pipeline] Label Only: skipped (no label written).")
                self.stats["tiles_processed"] = self.stats.get("tiles_processed", 0) + 1
                if self.on_stats and (self.stats["tiles_processed"] % 5 == 0):
                    self.on_stats(self.stats.copy())
                continue

            label, rel = result
            if not label or label == "__SKIP__":
                # Defensive: treat as skip
                print(f"[Pipeline] Label Only: no label returned, treated as skip.")
                self.stats["tiles_processed"] = self.stats.get("tiles_processed", 0) + 1
                continue

            print(f"[Pipeline] Label Only: labeled as '{label}' (relevant={rel})")

            # Save (or overwrite in edit mode) to the exact DB.
            # This is what enables both resume and accurate metrics.
            if self.tile_db is not None:
                try:
                    cx, cy = tile.bbox[0], tile.bbox[1]
                    self.tile_db.set_annotation(
                        vpath, f, r, c, tw, th,
                        label, rel,
                        embedding=None,   # pure label-only never stores embeddings here
                        crop_x=cx,
                        crop_y=cy,
                    )
                except Exception as e:
                    print(f"[Pipeline] Label Only: failed to save annotation: {e}")

            self.stats["tiles_processed"] = self.stats.get("tiles_processed", 0) + 1
            if self.on_stats and (self.stats["tiles_processed"] % 5 == 0):
                self.on_stats(self.stats.copy())
