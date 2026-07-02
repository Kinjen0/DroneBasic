"""
TileAnnotationDB

Exact-identity persistent store for human-provided tile labels.

Primary key is the combination of:
- video_path (stable identifier for the source video)
- abs_frame (absolute 0-based frame index from start of the *video file*, independent of stride)
- tile_row, tile_col (position in the grid for that frame)
- tile_width, tile_height (exact resolution of the crop)

This allows:
- Perfect recall of previous labels when re-processing the same video (even with different frame_stride).
- Editing / correcting past labels.
- Growing a high-quality database of labeled examples over time (multiple videos, multiple passes).
- "Edit mode" vs normal: in normal runs, known identities are auto-supplied without bothering the user.

We deliberately do *not* store the pixel data of tiles (storage cost). When review/editing is needed,
we re-decode the specific frame from the original video file and re-crop the tile on demand.

Complements (does not replace) the embedding-based PersistentLabelStore for "looks similar" auto-labeling
on completely new content.

Storage: sqlite3 (stdlib) + optional embedding bytes. Very lightweight per entry.
"""

from __future__ import annotations
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


class TileAnnotationDB:
    """
    Persistent database of exact tile annotations.

    Usage:
        db = TileAnnotationDB("my_annotations.db")
        label, rel = db.lookup(video, frame, r, c, tw, th)
        db.set_annotation(video, frame, r, c, tw, th, "Person", True, embedding=emb)

        for ann in db.get_annotations_for_video(video):
            ...
    """

    def __init__(self, db_path: str | Path = "drone_tile_annotations.db"):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # better concurrency / durability for edits
        self._create_tables()

        print(f"[TileDB] Opened annotation database at {self.db_path}")

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                video_path TEXT NOT NULL,
                abs_frame INTEGER NOT NULL,
                tile_row INTEGER NOT NULL,
                tile_col INTEGER NOT NULL,
                tile_width INTEGER NOT NULL,
                tile_height INTEGER NOT NULL,
                crop_x INTEGER,                      -- pixel origin of the crop (for exact re-extract)
                crop_y INTEGER,
                label TEXT NOT NULL,
                relevant INTEGER NOT NULL,           -- 0 / 1
                embedding BLOB,                      -- float32 bytes or NULL
                embedding_dim INTEGER,
                updated_ts REAL NOT NULL,
                PRIMARY KEY (video_path, abs_frame, tile_row, tile_col, tile_width, tile_height)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_video_frame ON annotations (video_path, abs_frame)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_label ON annotations (label)
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def lookup(self, video_path: str, abs_frame: int, tile_row: int, tile_col: int,
               tile_width: int, tile_height: int) -> Optional[Tuple[str, bool]]:
        """
        Return (label, relevant) for an exact tile identity, or None.
        """
        if not video_path:
            return None
        vpath = self._normalize_video_path(video_path)

        cur = self.conn.cursor()
        cur.execute("""
            SELECT label, relevant FROM annotations
            WHERE video_path=? AND abs_frame=? AND tile_row=? AND tile_col=?
                  AND tile_width=? AND tile_height=?
            LIMIT 1
        """, (vpath, int(abs_frame), int(tile_row), int(tile_col), int(tile_width), int(tile_height)))
        row = cur.fetchone()
        if row:
            label, rel = row
            return str(label), bool(rel)
        return None

    def set_annotation(self, video_path: str, abs_frame: int, tile_row: int, tile_col: int,
                       tile_width: int, tile_height: int, label: str, relevant: bool,
                       embedding: Optional[np.ndarray] = None,
                       crop_x: Optional[int] = None, crop_y: Optional[int] = None) -> None:
        """
        Insert or update (upsert) a label for this exact tile.
        Pass crop_x/crop_y (pixel top-left of the tile inside the frame) when available
        so that review can re-extract the exact same pixels without guessing stride.
        """
        if not video_path or not label:
            return
        vpath = self._normalize_video_path(video_path)
        ts = time.time()

        emb_blob = None
        emb_dim = None
        if embedding is not None:
            emb_arr = np.asarray(embedding, dtype=np.float32).ravel()
            emb_blob = emb_arr.tobytes()
            emb_dim = int(emb_arr.shape[0])

        cx = crop_x if crop_x is not None else (tile_col * tile_width)
        cy = crop_y if crop_y is not None else (tile_row * tile_height)

        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO annotations
                (video_path, abs_frame, tile_row, tile_col, tile_width, tile_height,
                 crop_x, crop_y, label, relevant, embedding, embedding_dim, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_path, abs_frame, tile_row, tile_col, tile_width, tile_height)
            DO UPDATE SET
                label=excluded.label,
                relevant=excluded.relevant,
                crop_x=COALESCE(excluded.crop_x, crop_x),
                crop_y=COALESCE(excluded.crop_y, crop_y),
                embedding=COALESCE(excluded.embedding, embedding),
                embedding_dim=COALESCE(excluded.embedding_dim, embedding_dim),
                updated_ts=excluded.updated_ts
        """, (
            vpath, int(abs_frame), int(tile_row), int(tile_col),
            int(tile_width), int(tile_height),
            int(cx), int(cy),
            str(label), 1 if relevant else 0,
            emb_blob, emb_dim, ts
        ))
        self.conn.commit()
        print(f"[TileDB] Saved/updated annotation: {Path(vpath).name} f{abs_frame} [{tile_row},{tile_col}] -> '{label}' (rel={relevant})")

    def get_annotations_for_video(self, video_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return list of annotations for a video (with crop info for re-extraction)."""
        vpath = self._normalize_video_path(video_path)
        cur = self.conn.cursor()
        q = """
            SELECT video_path, abs_frame, tile_row, tile_col, tile_width, tile_height,
                   crop_x, crop_y, label, relevant, updated_ts
            FROM annotations
            WHERE video_path=?
            ORDER BY abs_frame, tile_row, tile_col
        """
        if limit:
            q += f" LIMIT {int(limit)}"
        cur.execute(q, (vpath,))
        rows = cur.fetchall()
        results = []
        for r in rows:
            results.append({
                "video_path": r[0],
                "abs_frame": r[1],
                "tile_row": r[2],
                "tile_col": r[3],
                "tile_width": r[4],
                "tile_height": r[5],
                "crop_x": r[6] if r[6] is not None else r[3] * r[4],
                "crop_y": r[7] if r[7] is not None else r[2] * r[5],
                "label": r[8],
                "relevant": bool(r[9]),
                "updated_ts": r[10],
            })
        return results

    def list_videos(self) -> List[str]:
        """Return distinct video paths that have at least one annotation."""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT video_path FROM annotations ORDER BY video_path")
        return [row[0] for row in cur.fetchall()]

    def get_annotation_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM annotations")
        return cur.fetchone()[0]

    def delete_annotation(self, video_path: str, abs_frame: int, tile_row: int, tile_col: int,
                          tile_width: int, tile_height: int) -> bool:
        """Remove a specific annotation (rarely needed)."""
        vpath = self._normalize_video_path(video_path)
        cur = self.conn.cursor()
        cur.execute("""
            DELETE FROM annotations
            WHERE video_path=? AND abs_frame=? AND tile_row=? AND tile_col=?
                  AND tile_width=? AND tile_height=?
        """, (vpath, int(abs_frame), int(tile_row), int(tile_col), int(tile_width), int(tile_height)))
        deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _normalize_video_path(self, path: str) -> str:
        """Use absolute path for stability across runs (user must keep videos in consistent locations)."""
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(path)

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.get_annotation_count()

    # Optional: allow using as context manager
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ------------------------------------------------------------------
# Small helper to re-materialize a tile image from source video (no stored pixels)
# ------------------------------------------------------------------

def extract_tile_from_video(video_path: str, abs_frame: int, bbox: Tuple[int, int, int, int]) -> Optional["Image.Image"]:
    """
    Re-decode a specific frame from the video file and return the exact tile crop as PIL Image.
    Used for review/edit UI so we never need to store the actual image data.

    Returns None if seek/read fails (common on some video containers or very large seeks).
    Note: frame-accurate seeking is not guaranteed for all compressed videos, but is usually
    good enough for review/correction workflows.
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        # Try exact frame seek
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(abs_frame))
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback: try to read sequentially a bit (rare)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(abs_frame) - 2))
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    break
        if not ret or frame is None:
            return None

        x0, y0, x1, y1 = bbox
        h, w = frame.shape[:2]
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            return None

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb).convert("RGB")
    except Exception as e:
        print(f"[TileDB] Warning: failed to re-extract tile from {video_path} frame {abs_frame}: {e}")
        return None
    finally:
        cap.release()
