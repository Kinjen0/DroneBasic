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
            "queries": 0,
            "cache_hits": 0,
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

        # (Re)create heavy objects here so GUI can change config between runs
        self._init_components()

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
        if self._worker_thread:
            self._worker_thread.join(timeout=join_timeout)
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
    def _init_components(self):
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

        self.ared_adapter = AREDAdapter(self.config.ared)
        if self.label_store:
            self.ared_adapter.set_label_store(self.label_store)

        # Wire the label provider that the adapter will call on queries
        def _gui_label_provider(emb: np.ndarray, tile_img: Any, meta: Dict):
            # This runs in the ARED worker thread
            if self.label_store is not None:
                hit = self.label_store.lookup(emb)
                if hit:
                    self.stats["cache_hits"] = self.stats.get("cache_hits", 0) + 1
                    return hit

            # Ask the GUI
            req = LabelRequest(tile=tile_img or Tile(image=Image.new("RGB", (64, 64)), frame_idx=0, tile_row=0, tile_col=0, bbox=(0,0,0,0)),
                               embedding=emb, meta=meta)
            try:
                self.label_request_queue.put(req, timeout=5)
            except queue.Full:
                # Fallback: treat as irrelevant background if GUI is overwhelmed
                return "__BACKGROUND__", False

            result = req.wait(timeout=300)  # 5 minutes max for a single label - generous
            if result is None:
                return "__TIMEOUT__", False
            return result

        self.ared_adapter.set_label_provider(_gui_label_provider)

    def _run_loop(self):
        """Main worker loop: for each video, for selected frames, tile -> embed -> ARED."""
        self.stats["status"] = "running"
        videos = self.config.video_paths or []

        for vpath in videos:
            if self._stop_event.is_set():
                break
            self.stats["current_video"] = Path(vpath).name
            self._process_one_video(vpath)

        self.stats["status"] = "finished"
        if self.on_stats:
            self.on_stats(self.stats.copy())

    def _process_one_video(self, video_path: str):
        if not HAS_CV2:
            print("[Controller] cv2 not available - cannot read video.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Controller] Failed to open {video_path}")
            return

        frame_idx = -1
        frame_stride = max(1, self.config.tiling.frame_stride)
        batch_imgs: List[Image.Image] = []
        batch_tiles: List[Tile] = []

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

        finally:
            cap.release()

    def _process_batch(self, tiles: List[Tile], pil_images: List[Image.Image]):
        if not tiles or not self.feature_extractor or not self.ared_adapter:
            return

        embs = self.feature_extractor.extract_images(pil_images)

        for tile, emb in zip(tiles, embs):
            info = self.ared_adapter.process(emb, tile_image=tile, meta={
                "frame": tile.frame_idx,
                "row": tile.tile_row,
                "col": tile.tile_col,
            })
            self.stats["tiles_processed"] += 1
            if info.get("queried"):
                self.stats["queries"] = self.ared_adapter.num_queries

            # Push stats to GUI occasionally
            if self.on_stats and (self.stats["tiles_processed"] % 8 == 0):
                self.on_stats(self.stats.copy())
