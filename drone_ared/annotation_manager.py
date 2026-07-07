"""
annotation_manager.py

Reusable service layer for the tile annotation system.

Responsibilities (SRP):
- Scoped access (current video + tile size)
- Convenience wrappers around queries and bulk operations
- Hides raw DB details from higher layers (GUI, pipeline)
- Can be passed around instead of raw TileAnnotationDB

This, together with annotation_domain.py, forms the core reusable annotation module.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict, Any
from .annotation_domain import AnnotationFilter

if TYPE_CHECKING:
    from .tile_database import TileAnnotationDB


class AnnotationManager:
    """Coordinates access to a TileAnnotationDB (or future repo impl).

    Use set_scope() to establish the current (video, tile_size) context.
    get_annotations() and bulk_*() will automatically apply it unless overridden.
    """

    def __init__(self, db: "TileAnnotationDB"):
        self.db = db
        self._current_video: Optional[str] = None
        self._current_tile_size: Optional[Tuple[int, int]] = None

    def set_scope(self, video_path: Optional[str] = None, tile_size: Optional[Tuple[int, int]] = None):
        if video_path:
            # normalize via db if available
            norm = getattr(self.db, '_normalize_video_path', lambda p: p)
            self._current_video = norm(video_path)
        if tile_size:
            self._current_tile_size = tile_size

    def get_annotations(self, video: Optional[str] = None, use_scope: bool = True,
                        limit: Optional[int] = None, **filter_kwargs) -> List[Dict]:
        v = video or (self._current_video if use_scope else None)
        tw = th = None
        if use_scope and self._current_tile_size:
            tw, th = self._current_tile_size
        if v is None:
            return []
        # Pop injected keys to avoid duplicate kwarg if caller also supplied tile sizes via **filter_kwargs
        fk = dict(filter_kwargs)
        fk.pop('tile_width', None)
        fk.pop('tile_height', None)
        return self.db.get_annotations_for_video(v, limit=limit, tile_width=tw, tile_height=th, **fk)

    def bulk_delete(self, label: Optional[str] = None, labels: Optional[List[str]] = None,
                    use_scope: bool = True, **kw) -> int:
        """Delete matching. Pass label or labels list. Respects scope unless overridden."""
        v = self._current_video if use_scope else None
        tw, th = self._current_tile_size if (use_scope and self._current_tile_size) else (None, None)
        filt = kw.pop('filt', None)
        # Defensively remove keys we will pass explicitly to avoid "multiple values for keyword arg"
        kw.pop('video_path', None)
        kw.pop('tile_width', None)
        kw.pop('tile_height', None)
        if labels and not filt:
            filt = AnnotationFilter(labels=labels)
        return self.db.delete_by_filter(video_path=v, label=label, tile_width=tw, tile_height=th, filt=filt, **kw)

    def bulk_reassign(self, old_label: str, new_label: str, old_labels: Optional[List[str]] = None,
                      use_scope: bool = True, **kw) -> int:
        """Reassign matching old labels to new_label. Supports list of old labels."""
        v = self._current_video if use_scope else None
        tw, th = self._current_tile_size if (use_scope and self._current_tile_size) else (None, None)
        filt = kw.pop('filt', None)
        kw.pop('video_path', None)
        kw.pop('tile_width', None)
        kw.pop('tile_height', None)
        if old_labels and not filt:
            filt = AnnotationFilter(labels=old_labels)
        return self.db.reassign_label(old_label or (old_labels[0] if old_labels else ''), new_label,
                                      video_path=v, tile_width=tw, tile_height=th, filt=filt, **kw)

    def get_label_counts(self, use_scope: bool = True, **kw) -> Dict[str, int]:
        """Preview counts for mass ops. Respects current scope + any extra filters."""
        v = self._current_video if use_scope else None
        tw, th = self._current_tile_size if (use_scope and self._current_tile_size) else (None, None)

        passed_filt = kw.pop('filt', None)
        # Defensively remove scope keys we explicitly forward to db to prevent
        # "got multiple values for keyword argument 'xxx'" when callers pass them too.
        kw.pop('video_path', None)
        kw.pop('tile_width', None)
        kw.pop('tile_height', None)

        if use_scope and (v or tw or th):
            if passed_filt is None:
                effective_filt = AnnotationFilter(video_path=v, tile_width=tw, tile_height=th)
            else:
                # Merge: passed_filt takes precedence, scope fills in missing fields (e.g. labels + current video/size)
                effective_filt = AnnotationFilter(
                    video_path=passed_filt.video_path or v,
                    labels=passed_filt.labels,
                    tile_width=passed_filt.tile_width or tw,
                    tile_height=passed_filt.tile_height or th,
                    relevant=passed_filt.relevant,
                    frame_min=passed_filt.frame_min,
                    frame_max=passed_filt.frame_max,
                )
        else:
            effective_filt = passed_filt

        if effective_filt is not None:
            return self.db.get_label_counts(filt=effective_filt, video_path=v, tile_width=tw, tile_height=th, **kw)
        else:
            return self.db.get_label_counts(video_path=v, tile_width=tw, tile_height=th, **kw)

    def preview_bulk_change(self, old_labels: Optional[List[str]] = None, new_label: Optional[str] = None,
                            use_scope: bool = True, filt: Optional[AnnotationFilter] = None) -> Dict[str, Any]:
        """Return preview of what a mass change/delete would affect.
        Useful for confirmation dialogs. Pass full filt for video/size/frame specificity, or just old_labels.
        """
        if filt is None:
            filt = AnnotationFilter(labels=(old_labels or []))
        counts = self.get_label_counts(use_scope=use_scope, filt=filt)
        total = sum(counts.values())
        return {
            "affected_labels": counts,
            "total_affected": total,
            "action": "delete" if new_label is None else f"reassign to '{new_label}'",
        }

    def summary(self) -> Dict[str, Any]:
        return self.db.get_db_summary()

    def close(self):
        self.db.close()
