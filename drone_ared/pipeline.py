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
from typing import Optional, List, Callable, Any, Dict
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


class LabelRequest:
    """Message sent to GUI when ARED needs a human decision."""
    __slots__ = ("tile", "embedding", "meta", "response_event", "result")

    def __init__(self, tile: Tile, embedding: np.ndarray, meta: Dict):
        self.tile = tile
        self.embedding = embedding
        self.meta = meta
        self.response_event = threading.Event()
        self.result: Optional[tuple[str, bool]] = None   # filled by GUI

    def set_result(self, label: str, relevant: bool):
        self.result = (label, relevant)
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

        self._worker_thread: Optional[threading.Thread] = None
        self._global_tile_counter = 0
        self._current_label_req: Optional[LabelRequest] = None

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
        print("[Controller] Stopped")

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def set_label_store(self, store: PersistentLabelStore):
        self.label_store = store
        if self.ared_adapter:
            self.ared_adapter.set_label_store(store)

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

        fcfg = self.config.features
        self.feature_extractor = DINOFeatureExtractor(
            model_name=fcfg.model_name,
            device=fcfg.device,
            normalize=fcfg.normalize,
            pooling=fcfg.pooling,
            batch_size=fcfg.batch_size,
        )

        if create_ared or self.ared_adapter is None:
            self.ared_adapter = AREDAdapter(self.config.ared)
            if self.label_store:
                self.ared_adapter.set_label_store(self.label_store)

        # (Re)wire provider every time (the closure must see current stores etc.)
        def _gui_label_provider(emb: np.ndarray, tile_img: Any, meta: Dict):
            # This runs in the ARED worker thread.
            # A/RED decides whether a "query" is needed (anomalous or near-relevant cluster).
            # We only pop the GUI when we cannot satisfy from the persistent label cache.
            print(f"[Pipeline] _gui_label_provider called for tile: {meta} (emb shape: {emb.shape if hasattr(emb,'shape') else 'N/A'})")

            if self.label_store is not None:
                hit = self.label_store.lookup(emb)
                if hit:
                    self.stats["cache_hits"] = self.stats.get("cache_hits", 0) + 1
                    print(f"[Pipeline]   -> Label store CACHE HIT: '{hit[0]}' (relevant={hit[1]}). Returning without GUI.")
                    # Important: still return the label so A/RED can use it to grow clusters
                    # without forcing a human every time.
                    return hit
                else:
                    print(f"[Pipeline]   -> Label store CACHE MISS.")

            # Real human labeling needed
            self.stats["user_queries"] = self.stats.get("user_queries", 0) + 1
            print(f"[Pipeline]   -> Requesting HUMAN label via GUI (user_queries now {self.stats['user_queries']})  meta={meta}")

            req = LabelRequest(tile=tile_img or Tile(image=Image.new("RGB", (64, 64)), frame_idx=0, tile_row=0, tile_col=0, bbox=(0,0,0,0)),
                               embedding=emb, meta=meta)
            self._current_label_req = req
            try:
                self.label_request_queue.put(req, timeout=5)
            except queue.Full:
                self._current_label_req = None
                # Fallback: treat as irrelevant background if GUI is overwhelmed
                print("[Pipeline]   -> WARNING: label queue full, falling back to __BACKGROUND__")
                return "__BACKGROUND__", False

            print("[Pipeline]   -> Waiting for GUI response (blocking worker thread)...")
            result = req.wait(timeout=300)  # 5 minutes max for a single label - generous
            self._current_label_req = None
            if result is None:
                print("[Pipeline]   -> TIMEOUT waiting for label, using __TIMEOUT__")
                return "__TIMEOUT__", False
            print(f"[Pipeline]   -> Received label from GUI: '{result[0]}' (relevant={result[1]})")
            return result

        if self.ared_adapter:
            self.ared_adapter.set_label_provider(_gui_label_provider)

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

                    # Tile
                    tiles = self.tiler.tile_frame(frame_rgb, frame_idx, self._global_tile_counter)
                    if not tiles:
                        continue

                    # Collect for batched feature extraction
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
                info = self.ared_adapter.process(emb, tile_image=tile, meta={
                    "frame": tile.frame_idx,
                    "row": tile.tile_row,
                    "col": tile.tile_col,
                })
                self.stats["tiles_processed"] += 1

                ared_queried = info.get("queried", False)
                label = info.get("label")
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
