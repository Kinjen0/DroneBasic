"""
TileAnnotationDB

Exact-identity persistent store for human-provided tile labels.

Primary key is the combination of:
- video_path (stored as the **video filename only**, e.g. ``DJI_0018.MP4``, so DBs
  can be shared across machines without depending on absolute directory paths)
- abs_frame (absolute 0-based frame index from start of the *video file*, independent of stride)
- tile_row, tile_col (position in the grid for that frame)
- tile_width, tile_height (exact resolution of the crop)

This allows:
- Perfect recall of previous labels when re-processing the same video (even with different frame_stride).
- Editing / correcting past labels.
- Growing a high-quality database of labeled examples over time (multiple videos, multiple passes).
- "Edit mode" vs normal: in normal runs, known identities are auto-supplied without bothering the user.
- Sharing annotation DBs between users/systems that keep the same video filenames.

We deliberately do *not* store the pixel data of tiles (storage cost). When review/editing is needed,
we re-decode the specific frame from the original video file and re-crop the tile on demand.

Complements (does not replace) the embedding-based PersistentLabelStore for "looks similar" auto-labeling
on completely new content.

Storage: sqlite3 (stdlib) + optional embedding bytes. Very lightweight per entry.
"""

from __future__ import annotations
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

# Re-export domain models for backward compatibility and clean imports.
# New code should prefer: from drone_ared.annotation_domain import TileKey, ...
from .annotation_domain import TileKey, AnnotationFilter, TileAnnotation  # noqa: F401
from .label_sentinels import CONTROL_LABEL_SENTINELS, is_control_label, is_persistable_label


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

        # Older DBs stored absolute paths; rewrite keys to filename-only once.
        try:
            self._migrate_video_paths_to_basename()
        except Exception as e:
            print(f"[TileDB] Note: video-path basename migration skipped: {e}")

        # Remove any historically poisoned control-sentinel rows (e.g. __STOPPED__)
        # so they cannot auto-answer future A/RED queries via exact DB hit.
        try:
            n = self.purge_control_sentinel_labels()
            if n:
                print(f"[TileDB] Purged {n} control-sentinel label row(s) from {self.db_path.name}")
        except Exception as e:
            print(f"[TileDB] Note: control-sentinel purge skipped: {e}")

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
                stride_x INTEGER,                    -- NEW for overlap support (NULL = legacy/non-overlap)
                stride_y INTEGER,
                crop_x INTEGER,                      -- pixel origin of the crop (for exact re-extract)
                crop_y INTEGER,
                label TEXT NOT NULL,
                relevant INTEGER NOT NULL,           -- 0 / 1
                embedding BLOB,                      -- float32 bytes or NULL
                embedding_dim INTEGER,
                updated_ts REAL NOT NULL,
                -- stride_x/stride_y are stored (NULL for legacy runs).
                -- The application query logic (lookup_key + get_annotations_for_video) now
                -- scopes results so that different strides for the same (w,h) do not bleed labels.
                -- We keep the original PK for backward compatibility with existing DBs.
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
        # Lightweight migration for overlap support (add stride columns if missing)
        self._ensure_stride_columns()

    def _ensure_stride_columns(self):
        """Add stride_x / stride_y columns if the DB is from before overlap support."""
        try:
            cur = self.conn.cursor()
            cur.execute("PRAGMA table_info(annotations)")
            cols = {row[1] for row in cur.fetchall()}
            if "stride_x" not in cols:
                cur.execute("ALTER TABLE annotations ADD COLUMN stride_x INTEGER")
            if "stride_y" not in cols:
                cur.execute("ALTER TABLE annotations ADD COLUMN stride_y INTEGER")
            self.conn.commit()
        except Exception as e:
            print(f"[TileDB] Note: stride column migration check: {e}")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def lookup(self, video_path: str, abs_frame: int, tile_row: int, tile_col: int,
               tile_width: int, tile_height: int) -> Optional[Tuple[str, bool]]:
        """
        Return (label, relevant) for an exact tile identity, or None.
        (Legacy 6-arg form kept for compatibility during refactor.)
        """
        key = TileKey(video_path, int(abs_frame), int(tile_row), int(tile_col),
                      int(tile_width), int(tile_height))
        return self.lookup_key(key)

    def lookup_key(self, key: TileKey) -> Optional[Tuple[str, bool]]:
        """Preferred: lookup using a TileKey domain object.

        Overlap-aware behavior:
        - If the key carries explicit stride_x/stride_y, we prefer rows that match that stride.
        - We also accept legacy rows where stride_x/stride_y IS NULL (pre-overlap data).
        - This prevents non-overlap labels from "bleeding" onto the wrong physical tiles
          when the current grid uses a different stride.
        - If no stride is supplied in the key we do a legacy lookup (size + grid pos only).
        """
        if not key.video_path:
            return None
        vpath = self._normalize_video_path(key.video_path)

        cur = self.conn.cursor()

        base = """
            SELECT label, relevant FROM annotations
            WHERE video_path=? AND abs_frame=? AND tile_row=? AND tile_col=?
                  AND tile_width=? AND tile_height=?
        """
        # IMPORTANT: use the *normalized* vpath as first param (to_tuple may contain the raw path)
        t = key.to_tuple()
        params = [vpath, t[1], t[2], t[3], t[4], t[5]]

        if key.stride_x is not None or key.stride_y is not None:
            # Stride-aware: match exact stride OR legacy NULL stride rows.
            # This lets old data still be found when the user hasn't labeled under the new stride yet,
            # while protecting against wrong-grid bleed for explicitly different strides.
            base += " AND ( (stride_x IS NULL AND stride_y IS NULL) OR (stride_x=? AND stride_y=?) )"
            params.extend([key.stride_x, key.stride_y])

        base += " LIMIT 1"
        cur.execute(base, params)
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
        Legacy form; new code should prefer set_annotation_for_key.
        """
        key = TileKey(video_path, int(abs_frame), int(tile_row), int(tile_col), int(tile_width), int(tile_height))
        self.set_annotation_for_key(key, label, relevant, embedding=embedding, crop_x=crop_x, crop_y=crop_y)

    def set_annotation_for_key(self, key: TileKey, label: str, relevant: bool,
                               embedding: Optional[np.ndarray] = None,
                               crop_x: Optional[int] = None, crop_y: Optional[int] = None) -> None:
        """Preferred form using domain TileKey (improves readability & type safety)."""
        if not key.video_path or not label:
            return
        # Never persist control-plane sentinels (__STOPPED__, __TIMEOUT__, etc.)
        if not is_persistable_label(label):
            print(f"[TileDB] Refusing to store control-sentinel label '{label}' (not a real class).")
            return
        vpath = self._normalize_video_path(key.video_path)
        ts = time.time()

        emb_blob = None
        emb_dim = None
        if embedding is not None:
            emb_arr = np.asarray(embedding, dtype=np.float32).ravel()
            emb_blob = emb_arr.tobytes()
            emb_dim = int(emb_arr.shape[0])

        cx = crop_x if crop_x is not None else (key.tile_col * key.tile_width)
        cy = crop_y if crop_y is not None else (key.tile_row * key.tile_height)

        cur = self.conn.cursor()
        # stride columns (may be NULL for legacy rows)
        sx = key.stride_x
        sy = key.stride_y

        cur.execute("""
            INSERT INTO annotations
                (video_path, abs_frame, tile_row, tile_col, tile_width, tile_height,
                 stride_x, stride_y, crop_x, crop_y, label, relevant, embedding, embedding_dim, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_path, abs_frame, tile_row, tile_col, tile_width, tile_height)
            DO UPDATE SET
                label=excluded.label,
                relevant=excluded.relevant,
                stride_x=COALESCE(excluded.stride_x, stride_x),
                stride_y=COALESCE(excluded.stride_y, stride_y),
                crop_x=COALESCE(excluded.crop_x, crop_x),
                crop_y=COALESCE(excluded.crop_y, crop_y),
                embedding=COALESCE(excluded.embedding, embedding),
                embedding_dim=COALESCE(excluded.embedding_dim, embedding_dim),
                updated_ts=excluded.updated_ts
        """, (
            vpath, int(key.abs_frame), int(key.tile_row), int(key.tile_col),
            int(key.tile_width), int(key.tile_height),
            sx, sy,
            int(cx), int(cy),
            str(label), 1 if relevant else 0,
            emb_blob, emb_dim, ts
        ))
        self.conn.commit()
        print(f"[TileDB] Saved/updated annotation: {Path(vpath).name} f{key.abs_frame} [{key.tile_row},{key.tile_col}] -> '{label}' (rel={relevant})")

    def get_annotations_for_video(self, video_path: str, limit: Optional[int] = None,
                                  tile_width: Optional[int] = None,
                                  tile_height: Optional[int] = None,
                                  stride_x: Optional[int] = None,
                                  stride_y: Optional[int] = None,
                                  filt: Optional[AnnotationFilter] = None) -> List[Dict[str, Any]]:
        """Return list of annotations for a video (with crop info for re-extraction).

        Pass tile_width and/or tile_height (or a full AnnotationFilter) to scope results.
        For overlapping tiles, also pass stride_x/stride_y (or via filt) to avoid mixing
        labels created under different grid steps. Legacy rows (NULL stride) are included
        when no explicit stride filter is given.
        """
        if filt is not None:
            video_path = video_path or filt.video_path or video_path
            tile_width = tile_width or filt.tile_width
            tile_height = tile_height or filt.tile_height
            stride_x = stride_x if stride_x is not None else filt.stride_x
            stride_y = stride_y if stride_y is not None else filt.stride_y
            # labels filter not applied here for simplicity (list is per video)

        vpath = self._normalize_video_path(video_path)
        cur = self.conn.cursor()
        q = """
            SELECT video_path, abs_frame, tile_row, tile_col, tile_width, tile_height,
                   stride_x, stride_y, crop_x, crop_y, label, relevant, updated_ts
            FROM annotations
            WHERE video_path=?
        """
        params = [vpath]
        if tile_width is not None:
            q += " AND tile_width=?"
            params.append(int(tile_width))
        if tile_height is not None:
            q += " AND tile_height=?"
            params.append(int(tile_height))
        if stride_x is not None or stride_y is not None:
            # Match exact stride OR legacy NULL-stride records (best-effort compatibility)
            q += " AND ( (stride_x IS NULL AND stride_y IS NULL) OR (stride_x=? AND stride_y=?) )"
            params.extend([stride_x, stride_y])
        q += " ORDER BY abs_frame, tile_row, tile_col"
        if limit:
            q += f" LIMIT {int(limit)}"
        cur.execute(q, params)
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
                "stride_x": r[6],
                "stride_y": r[7],
                "crop_x": r[8] if r[8] is not None else r[3] * r[4],
                "crop_y": r[9] if r[9] is not None else r[2] * r[5],
                "label": r[10],
                "relevant": bool(r[11]),
                "updated_ts": r[12],
            })
        return results

    def list_videos(self) -> List[str]:
        """Return distinct video paths that have at least one annotation."""
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT video_path FROM annotations ORDER BY video_path")
        return [row[0] for row in cur.fetchall()]

    def get_tile_sizes_for_video(self, video_path: str) -> List[Tuple[int, int]]:
        """Return the distinct (width, height) tile resolutions that have annotations for this video.
        Useful for UI warnings when the current GUI tile size has no data in the loaded DB.
        """
        vpath = self._normalize_video_path(video_path)
        cur = self.conn.cursor()
        cur.execute("""
            SELECT DISTINCT tile_width, tile_height
            FROM annotations
            WHERE video_path=?
            ORDER BY tile_width, tile_height
        """, (vpath,))
        return [(int(r[0]), int(r[1])) for r in cur.fetchall()]

    def get_annotation_count(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM annotations")
        return cur.fetchone()[0]

    def get_label_counts(self, filt: Optional[AnnotationFilter] = None,
                         video_path: Optional[str] = None,
                         tile_width: Optional[int] = None,
                         tile_height: Optional[int] = None) -> Dict[str, int]:
        """Return {label: count} for annotations matching filter.
        Useful for previewing mass edit/remove operations.
        """
        if filt is not None:
            video_path = video_path or filt.video_path
            tile_width = tile_width or filt.tile_width
            tile_height = tile_height or filt.tile_height
            labels = filt.labels
            relevant = filt.relevant
            frame_min = filt.frame_min
            frame_max = filt.frame_max
        else:
            labels = relevant = frame_min = frame_max = None

        cur = self.conn.cursor()
        where = []
        params: List[Any] = []
        if video_path:
            where.append("video_path=?")
            params.append(self._normalize_video_path(video_path))
        if labels:
            if len(labels) == 1:
                where.append("label=?")
                params.append(str(labels[0]))
            else:
                placeholders = ",".join("?" for _ in labels)
                where.append(f"label IN ({placeholders})")
                params.extend(str(l) for l in labels)
        if tile_width is not None:
            where.append("tile_width=?")
            params.append(int(tile_width))
        if tile_height is not None:
            where.append("tile_height=?")
            params.append(int(tile_height))
        if relevant is not None:
            where.append("relevant=?")
            params.append(1 if relevant else 0)
        if frame_min is not None:
            where.append("abs_frame >= ?")
            params.append(int(frame_min))
        if frame_max is not None:
            where.append("abs_frame <= ?")
            params.append(int(frame_max))

        sql = "SELECT label, COUNT(*) FROM annotations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY label ORDER BY label"
        cur.execute(sql, params)
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_db_summary(self) -> Dict[str, Any]:
        """Return a dict with useful info for UI: path, total entries, num videos, sizes present, labels, last update."""
        summary = {
            "path": str(self.db_path),
            "total_entries": self.get_annotation_count(),
            "num_videos": len(self.list_videos()),
            "tile_sizes": [],  # list of (w,h) across all
            "labels": self.get_all_labels(),
            "last_updated": None,
        }
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT DISTINCT tile_width, tile_height FROM annotations ORDER BY tile_width, tile_height")
            summary["tile_sizes"] = [(int(r[0]), int(r[1])) for r in cur.fetchall()]

            cur.execute("SELECT MAX(updated_ts) FROM annotations")
            row = cur.fetchone()
            if row and row[0]:
                summary["last_updated"] = row[0]
        except Exception:
            pass
        return summary

    def vacuum(self) -> None:
        """Compact the DB (removes free space after deletes/bulk ops)."""
        try:
            self.conn.execute("VACUUM")
            self.conn.commit()
            print("[TileDB] Vacuum completed.")
        except Exception as e:
            print(f"[TileDB] Vacuum failed: {e}")

    def get_all_labels(self) -> List[str]:
        """Return all unique labels that have been assigned in this DB.

        Critical for sparse/resume Label Only mode:
        When the user quickly labels only the relevant tiles (e.g. every "person")
        and skips the rest, those class names must still appear in the main GUI's
        "Discovered Classes" list so they can be clicked instead of re-typed on
        subsequent tiles.

        Control-plane sentinels (__STOPPED__, etc.) are filtered out.
        """
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT DISTINCT label FROM annotations ORDER BY label")
            return [row[0] for row in cur.fetchall() if is_persistable_label(row[0])]
        except Exception:
            return []

    def purge_control_sentinel_labels(self) -> int:
        """Delete rows whose label is a control-plane sentinel (not a real class).

        Returns the number of rows deleted. Safe to call on every open.
        """
        if not CONTROL_LABEL_SENTINELS:
            return 0
        try:
            cur = self.conn.cursor()
            placeholders = ",".join("?" for _ in CONTROL_LABEL_SENTINELS)
            cur.execute(
                f"DELETE FROM annotations WHERE label IN ({placeholders})",
                tuple(CONTROL_LABEL_SENTINELS),
            )
            n = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
            self.conn.commit()
            return int(n)
        except Exception as e:
            print(f"[TileDB] purge_control_sentinel_labels failed: {e}")
            return 0

    def get_class_relevance(self, label: str) -> Optional[bool]:
        """Return the relevant flag for a class if it was previously assigned
        via this exact annotation DB.
        Used to show the [relevant] tag in class lists during resume/sparse labeling
        even when the embedding label_store was not used.
        """
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT relevant FROM annotations
                WHERE label = ?
                LIMIT 1
            """, (label,))
            row = cur.fetchone()
            if row is not None:
                return bool(row[0])
        except Exception:
            pass
        return None

    def delete_annotation(self, video_path: str, abs_frame: int, tile_row: int, tile_col: int,
                          tile_width: int, tile_height: int, stride_x: Optional[int] = None,
                          stride_y: Optional[int] = None) -> bool:
        """Remove a specific annotation (rarely needed)."""
        key = TileKey(video_path, int(abs_frame), int(tile_row), int(tile_col), int(tile_width), int(tile_height),
                      stride_x=stride_x, stride_y=stride_y)
        return self.delete_key(key)

    def delete_key(self, key: TileKey) -> bool:
        """Delete using domain key (preferred during refactor).

        When stride is present on the key we scope the delete to that stride (or legacy NULL).
        """
        vpath = self._normalize_video_path(key.video_path)
        cur = self.conn.cursor()
        sql = """
            DELETE FROM annotations
            WHERE video_path=? AND abs_frame=? AND tile_row=? AND tile_col=?
                  AND tile_width=? AND tile_height=?
        """
        # Use normalized filename key (not the raw path from to_tuple).
        t = key.to_tuple()
        params = [vpath, t[1], t[2], t[3], t[4], t[5]]
        if key.stride_x is not None or key.stride_y is not None:
            sql += " AND ( (stride_x IS NULL AND stride_y IS NULL) OR (stride_x=? AND stride_y=?) )"
            params.extend([key.stride_x, key.stride_y])
        cur.execute(sql, params)
        deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    # ------------------------------------------------------------------
    # Bulk operations (for mass curation / cleanup of labeling experiments)
    # ------------------------------------------------------------------

    def delete_by_filter(self, video_path: Optional[str] = None,
                         label: Optional[str] = None,
                         tile_width: Optional[int] = None,
                         tile_height: Optional[int] = None,
                         relevant: Optional[bool] = None,
                         filt: Optional[AnnotationFilter] = None) -> int:
        """Delete annotations matching the supplied filters.
        Supports single label or list of labels (via AnnotationFilter.labels or legacy).
        Supports frame ranges via filt.
        Returns number of rows deleted. Safety: refuses if no criteria.
        """
        if filt is not None:
            video_path = video_path or filt.video_path
            tile_width = tile_width or filt.tile_width
            tile_height = tile_height or filt.tile_height
            relevant = relevant if relevant is not None else filt.relevant
            if filt.labels and not label:
                labels = filt.labels
            else:
                labels = [label] if label is not None else None
            frame_min = filt.frame_min
            frame_max = filt.frame_max
        else:
            labels = [label] if label is not None else None
            frame_min = frame_max = None

        cur = self.conn.cursor()
        where = []
        params: List[Any] = []
        if video_path:
            where.append("video_path=?")
            params.append(self._normalize_video_path(video_path))
        if labels:
            if len(labels) == 1:
                where.append("label=?")
                params.append(str(labels[0]))
            else:
                placeholders = ",".join("?" for _ in labels)
                where.append(f"label IN ({placeholders})")
                params.extend(str(l) for l in labels)
        if tile_width is not None:
            where.append("tile_width=?")
            params.append(int(tile_width))
        if tile_height is not None:
            where.append("tile_height=?")
            params.append(int(tile_height))
        if relevant is not None:
            where.append("relevant=?")
            params.append(1 if relevant else 0)
        if frame_min is not None:
            where.append("abs_frame >= ?")
            params.append(int(frame_min))
        if frame_max is not None:
            where.append("abs_frame <= ?")
            params.append(int(frame_max))

        if not where:
            return 0

        sql = "DELETE FROM annotations WHERE " + " AND ".join(where)
        cur.execute(sql, params)
        deleted = cur.rowcount
        self.conn.commit()
        return deleted

    def reassign_label(self, old_label: str, new_label: str,
                       video_path: Optional[str] = None,
                       tile_width: Optional[int] = None,
                       tile_height: Optional[int] = None,
                       filt: Optional[AnnotationFilter] = None) -> int:
        """Mass reassign labels matching criteria to new_label.
        Supports multiple old labels via filt.labels or single old_label.
        Supports frame ranges. Returns count updated.
        """
        if filt is not None:
            video_path = video_path or filt.video_path
            tile_width = tile_width or filt.tile_width
            tile_height = tile_height or filt.tile_height
            if filt.labels and not old_label:
                old_labels = filt.labels
            else:
                old_labels = [old_label] if old_label else []
            frame_min = filt.frame_min
            frame_max = filt.frame_max
        else:
            old_labels = [old_label] if old_label else []
            frame_min = frame_max = None

        if not old_labels or not new_label or new_label in old_labels:
            return 0

        cur = self.conn.cursor()
        set_clause = "label = ?, updated_ts = ?"
        params: List[Any] = [str(new_label), time.time()]

        where = []
        if len(old_labels) == 1:
            where.append("label = ?")
            params.append(str(old_labels[0]))
        else:
            placeholders = ",".join("?" for _ in old_labels)
            where.append(f"label IN ({placeholders})")
            params.extend(str(l) for l in old_labels)

        if video_path:
            where.append("video_path = ?")
            params.append(self._normalize_video_path(video_path))
        if tile_width is not None:
            where.append("tile_width = ?")
            params.append(int(tile_width))
        if tile_height is not None:
            where.append("tile_height = ?")
            params.append(int(tile_height))
        if frame_min is not None:
            where.append("abs_frame >= ?")
            params.append(int(frame_min))
        if frame_max is not None:
            where.append("abs_frame <= ?")
            params.append(int(frame_max))

        if not where:
            return 0

        sql = f"UPDATE annotations SET {set_clause} WHERE {' AND '.join(where)}"
        cur.execute(sql, params)
        updated = cur.rowcount
        self.conn.commit()
        return updated

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _normalize_video_path(self, path: str) -> str:
        """
        Stable video identity for DB keys: **filename only** (not full path).

        Using the basename (e.g. ``DJI_0018.MP4``) lets annotation DBs be shared
        across machines/users as long as video *filenames* match. Callers may still
        pass absolute paths when opening files; only the DB key is normalized here.
        """
        if path is None:
            return ""
        try:
            # Normalize mixed Windows/Unix separators, then take the final component.
            s = str(path).strip().replace("\\", "/")
            name = os.path.basename(s)
            return name if name else s
        except Exception:
            return str(path)

    def _migrate_video_paths_to_basename(self) -> None:
        """One-time rewrite of legacy absolute/relative path keys → filename only.

        Safe to call every open: rows already stored as bare filenames are skipped.
        If two different full paths collapse to the same basename and would violate
        the primary key, those rows are left unchanged and a warning is printed.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT video_path FROM annotations")
        distinct = [r[0] for r in cur.fetchall() if r and r[0]]
        migrated_rows = 0
        for vp in distinct:
            base = self._normalize_video_path(vp)
            if not base or base == vp:
                continue
            try:
                cur.execute(
                    "UPDATE annotations SET video_path=? WHERE video_path=?",
                    (base, vp),
                )
                migrated_rows += int(cur.rowcount or 0)
            except sqlite3.IntegrityError:
                print(
                    f"[TileDB] Basename migration skipped for conflict: "
                    f"{vp!r} -> {base!r}"
                )
        if migrated_rows:
            self.conn.commit()
            print(
                f"[TileDB] Migrated {migrated_rows} annotation row(s) "
                f"to filename-only video keys (portable across systems)."
            )

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

    @classmethod
    def clone(cls, source_path: str | Path, dest_path: str | Path, open_after: bool = True) -> Optional["TileAnnotationDB"]:
        """Safely clone a DB file to a new location.
        Closes any implicit, copies the main .db (and attempts sidecar WAL files if present).
        Returns a new TileAnnotationDB instance open on the clone if open_after=True.
        """
        import shutil
        src = Path(source_path).resolve()
        dst = Path(dest_path).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            raise FileNotFoundError(f"Source DB not found: {src}")

        # Copy main file
        shutil.copy2(src, dst)

        # Try to copy WAL/SHM sidecars (for WAL mode safety; harmless if missing)
        for suffix in (".db-wal", ".db-shm"):
            side = src.with_suffix(suffix) if src.suffix == ".db" else Path(str(src) + suffix)
            if side.exists():
                try:
                    shutil.copy2(side, dst.with_suffix(suffix) if dst.suffix == ".db" else Path(str(dst) + suffix))
                except Exception:
                    pass

        if open_after:
            return cls(db_path=dst)
        return None


# Note: AnnotationManager lives in annotation_manager.py for clean separation.
# Import directly: from drone_ared.annotation_manager import AnnotationManager
# tile_database re-exports domain models only for compat.


# ------------------------------------------------------------------
# Resolve filename-only DB keys → openable video paths (for review UIs)
# ------------------------------------------------------------------

def resolve_video_file(
    name_or_path: str,
    *,
    search_paths: Optional[List[str | Path]] = None,
    search_dirs: Optional[List[str | Path]] = None,
) -> Optional[str]:
    """
    Map a DB video key (often just a filename like ``DJI_0018.MP4``) to a real
    filesystem path that ``cv2.VideoCapture`` can open.

    Search order:
      1. ``name_or_path`` itself if it already exists on disk
      2. Any path in ``search_paths`` whose basename matches
      3. ``basename`` joined under each directory in ``search_dirs``

    Returns an absolute path string, or None if nothing is found.
    """
    if not name_or_path:
        return None

    raw = str(name_or_path).strip()
    if not raw:
        return None

    # 1) Already a usable path?
    try:
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
    except Exception:
        pass

    # Basename used as the portable DB key
    try:
        base = os.path.basename(raw.replace("\\", "/"))
    except Exception:
        base = raw
    if not base:
        return None

    # 2) Known full paths (e.g. videos loaded in the main GUI this session)
    for cand in search_paths or []:
        try:
            cp = Path(cand)
            if cp.is_file() and cp.name == base:
                return str(cp.resolve())
            # Also accept exact string match after normalize
            if os.path.basename(str(cand).replace("\\", "/")) == base and cp.is_file():
                return str(cp.resolve())
        except Exception:
            continue

    # 3) Look under candidate directories
    seen_dirs = set()
    for d in search_dirs or []:
        try:
            dp = Path(d)
            if not dp.is_dir():
                continue
            key = str(dp.resolve())
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            hit = dp / base
            if hit.is_file():
                return str(hit.resolve())
        except Exception:
            continue

    return None


# ------------------------------------------------------------------
# Small helper to re-materialize a tile image from source video (no stored pixels)
# ------------------------------------------------------------------

def extract_tile_from_video(
    video_path: str,
    abs_frame: int,
    bbox: Tuple[int, int, int, int],
    *,
    search_paths: Optional[List[str | Path]] = None,
    search_dirs: Optional[List[str | Path]] = None,
) -> Optional["Image.Image"]:
    """
    Re-decode a specific frame from the video file and return the exact tile crop as PIL Image.
    Used for review/edit UI so we never need to store the actual image data.

    ``video_path`` may be a full path or a filename-only DB key. When it is only a
    name, pass ``search_paths`` / ``search_dirs`` (or rely on callers to resolve first).

    Returns None if seek/read fails (common on some video containers or very large seeks).
    Note: frame-accurate seeking is not guaranteed for all compressed videos, but is usually
    good enough for review/correction workflows.
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return None

    openable = resolve_video_file(
        video_path, search_paths=search_paths, search_dirs=search_dirs
    )
    if openable is None:
        # Last resort: try the raw string (legacy full-path DBs / cwd-relative names)
        openable = str(video_path) if video_path else None
    if not openable:
        return None

    cap = cv2.VideoCapture(openable)
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
