"""
RunMetricsLogger — periodic running metrics + per-run disk packages.

Every N tiles (default 5000), computes QP / RR / F1 via the same paper formulas
as metrics.evaluate_from_annotations_and_queries, and appends a checkpoint.

Each Start creates:
  runs/<run_id>/
    run.json           full document (params + checkpoints + final)
    checkpoints.csv    flat table for plotting
    final_audit.txt    human-readable final summary (on finalize)
"""

from __future__ import annotations

import csv
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from . import metrics as ared_metrics
from .metrics import summarize_for_checkpoint, make_tile_key
from .label_sentinels import is_persistable_label
from .logutil import vprint

if TYPE_CHECKING:
    from .pipeline import DroneAREDController


CHECKPOINT_CSV_FIELDS = [
    "checkpoint_index",
    "reason",
    "tiles_processed",
    "frames_read",
    "ared_queries",
    "user_queries",
    "cache_hits",
    # Cumulative rates (stream so far)
    "query_rate",
    "relevant_rate",
    # Section rates for the tiles since the previous checkpoint (size ≈ checkpoint_every)
    "section_tiles",
    "section_ared_queries",
    "section_query_rate",
    "section_relevant_rate",
    "query_precision",
    "relevant_recall",
    "f1_score",
    "n_should_query",
    "tp",
    "fp",
    "fn",
    "total_relevant_tiles",
    "total_relevant_tiles_queried",
    "classes_discovered_x_of_y",
    "n_classes_queried",
    "n_unique_classes",
    "baseline_random_qp",
    "baseline_random_rr",
    "qp_improvement_ratio_vs_random",
    "rr_improvement_ratio_vs_random",
    "relevant_recall_strict",
    "ared_clusters",
    "ared_known_labels",
    "elapsed_sec",
    "current_video",
    "metrics_available",
    "note",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(obj: Any) -> Any:
    """Make objects JSON-serializable (tuples → lists, paths → str)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, set):
        return [_json_safe(x) for x in sorted(obj, key=str)]
    try:
        return float(obj)
    except Exception:
        return str(obj)


def make_run_id(run_params: Dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    kappa = run_params.get("kappa", "?")
    ts_sz = run_params.get("tile_size") or (0, 0)
    if isinstance(ts_sz, (list, tuple)) and len(ts_sz) >= 2:
        tw, th = ts_sz[0], ts_sz[1]
    else:
        tw = th = ts_sz
    sx = run_params.get("stride_x", tw)
    sy = run_params.get("stride_y", th)
    fs = run_params.get("frame_stride", "?")
    return f"{ts}__kappa{kappa}__tile{tw}x{th}__s{sx}x{sy}__fs{fs}"


class RunMetricsLogger:
    """Owns one run directory and appends checkpoints during streaming."""

    def __init__(
        self,
        output_dir: str | Path = "runs",
        checkpoint_every: int = 5000,
        run_params: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        checkpoint_on_video_end: bool = True,
    ):
        self.enabled = bool(enabled)
        self.checkpoint_every = max(1, int(checkpoint_every or 5000))
        self.checkpoint_on_video_end = bool(checkpoint_on_video_end)
        self.run_params: Dict[str, Any] = dict(run_params or {})
        self.run_params["metrics_checkpoint_every"] = self.checkpoint_every

        self.run_id = make_run_id(self.run_params)
        self.base_dir = Path(output_dir).resolve()
        self.run_dir = self.base_dir / self.run_id
        # Avoid collisions if two starts land in the same second
        if self.run_dir.exists():
            suffix = 1
            while (self.base_dir / f"{self.run_id}_{suffix}").exists():
                suffix += 1
            self.run_id = f"{self.run_id}_{suffix}"
            self.run_dir = self.base_dir / self.run_id

        self.started_at = _utc_now_iso()
        self._t0 = time.time()
        self.ended_at: Optional[str] = None
        self.status = "running"
        self.checkpoints: List[Dict[str, Any]] = []
        self.final_metrics: Optional[Dict[str, Any]] = None
        self.last_checkpoint: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._last_checkpoint_tiles = 0
        self._finalized = False
        self._csv_path = self.run_dir / "checkpoints.csv"
        self._json_path = self.run_dir / "run.json"

        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._init_csv()
            self._write_json()
            print(f"[RunMetrics] Logging to {self.run_dir}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def maybe_checkpoint(self, controller: "DroneAREDController", reason: str = "interval") -> Optional[Dict[str, Any]]:
        """If tiles_processed hit a multiple of N (or forced reason), record a checkpoint."""
        if not self.enabled:
            return None
        n = int(controller.stats.get("tiles_processed", 0) or 0)
        if n <= 0:
            return None
        if reason == "interval":
            if n % self.checkpoint_every != 0:
                return None
            if n == self._last_checkpoint_tiles:
                return None
        return self.checkpoint(controller, reason=reason)

    def checkpoint(self, controller: "DroneAREDController", reason: str = "interval") -> Dict[str, Any]:
        """Compute and persist one running snapshot."""
        if not self.enabled:
            return {}

        with self._lock:
            snap = self._build_snapshot(controller, reason=reason)
            self.checkpoints.append(snap)
            self.last_checkpoint = snap
            self._last_checkpoint_tiles = int(snap.get("tiles_processed") or 0)
            self._append_csv(snap)
            self._write_json()
            line = (
                f"[RunMetrics] checkpoint #{snap.get('checkpoint_index')} "
                f"@ {snap.get('tiles_processed')} tiles  "
                f"QP={snap.get('query_precision')} RR={snap.get('relevant_recall')} "
                f"F1={snap.get('f1_score')}  "
                f"QR={snap.get('query_rate')} RelRate={snap.get('relevant_rate')}  "
                f"secQR={snap.get('section_query_rate')} secRel={snap.get('section_relevant_rate')}  "
                f"queries={snap.get('ared_queries')}  ({reason})"
            )
            vprint(line)
            return snap

    def finalize(
        self,
        controller: "DroneAREDController",
        status: str = "finished",
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Final checkpoint + full final_metrics package.

        Idempotent: stop() and the worker loop may both call this; only the first
        call writes the end package.

        Always writes:
          - end checkpoint in checkpoints.csv / run.json
          - final_metrics inside run.json
          - final_metrics.json (standalone full package for tools)
          - final_audit.txt (human-readable)
        """
        if not self.enabled:
            return None

        with self._lock:
            if self._finalized:
                return self.last_checkpoint
            self._finalized = True

            # Always take an end snapshot if we have processed anything
            n = int(controller.stats.get("tiles_processed", 0) or 0)
            if n > 0 and (not self.checkpoints or self.checkpoints[-1].get("tiles_processed") != n):
                snap = self._build_snapshot(controller, reason="end")
                self.checkpoints.append(snap)
                self.last_checkpoint = snap
                self._append_csv(snap)

            if final_metrics is None:
                try:
                    final_metrics = self._compute_full_metrics(controller)
                except Exception as e:
                    print(f"[RunMetrics] Final metrics compute FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    final_metrics = {
                        "error": f"final_metrics_compute_failed: {e}",
                        "tiles_processed": n,
                        "ared_queries": int(controller.stats.get("ared_queries", 0) or 0),
                        "frames_read": int(controller.stats.get("frames_read", 0) or 0),
                    }

            # If evaluation returned only an error (e.g. no annotations yet), still
            # package stream counters so the run dir is never "empty" of finals.
            if not final_metrics:
                final_metrics = {"error": "no final_metrics produced"}
            if isinstance(final_metrics, dict) and "error" in final_metrics:
                final_metrics.setdefault("tiles_processed", n)
                final_metrics.setdefault(
                    "ared_queries", int(controller.stats.get("ared_queries", 0) or 0)
                )
                final_metrics.setdefault(
                    "frames_read", int(controller.stats.get("frames_read", 0) or 0)
                )
                final_metrics.setdefault(
                    "n_processed_identities",
                    len(getattr(controller, "processed_identities", None) or []),
                )
                final_metrics.setdefault(
                    "n_queried_identities",
                    len(getattr(controller, "queried_identities", None) or []),
                )
                final_metrics.setdefault("run_params", _json_safe(self.run_params))
                final_metrics.setdefault(
                    "summary",
                    f"Metrics incomplete: {final_metrics.get('error')} "
                    f"(tiles={n}, queries={final_metrics.get('ared_queries')})",
                )

            # Compact for run.json (drop huge audit dict; keep summary + key fields)
            compact = None
            if final_metrics:
                compact = {
                    k: v
                    for k, v in final_metrics.items()
                    if k not in ("detailed_breakdown", "audit")
                }
                audit = final_metrics.get("detailed_breakdown") or final_metrics.get("audit")
                if audit and isinstance(audit, dict):
                    compact["audit_summary"] = {
                        k: audit.get(k)
                        for k in (
                            "TOTAL_QUERIES_ARED_MADE",
                            "TOTAL_RELEVANT_TILES",
                            "TOTAL_RELEVANT_TILES_QUERIED",
                            "CLASSES_DISCOVERED_X_Y",
                            "TP",
                            "FP",
                            "FN",
                            "QUERY_PRECISION_WORK",
                            "RELEVANT_RECALL_WORK",
                            "F1_WORK",
                            "RANDOM_BASELINE",
                            "RANDOM_RR_EQUALS_QUERY_RATE",
                            "QP_IMPROVEMENT_RATIO_VS_RANDOM",
                            "RR_IMPROVEMENT_RATIO_VS_RANDOM",
                            "RUN_PARAMS",
                        )
                        if k in audit
                    }
                    try:
                        audit_path = self.run_dir / "final_audit.txt"
                        with open(audit_path, "w", encoding="utf-8") as f:
                            f.write(str(final_metrics.get("summary", "")) + "\n\n")
                            for k, v in audit.items():
                                f.write(f"{k}: {v}\n")
                    except Exception as e:
                        print(f"[RunMetrics] Could not write final_audit.txt: {e}")
                else:
                    # No detailed audit (error path) — still write a minimal final_audit.txt
                    try:
                        audit_path = self.run_dir / "final_audit.txt"
                        with open(audit_path, "w", encoding="utf-8") as f:
                            f.write(str(final_metrics.get("summary", "")) + "\n\n")
                            for k, v in final_metrics.items():
                                if k in ("detailed_breakdown", "audit"):
                                    continue
                                f.write(f"{k}: {v}\n")
                    except Exception as e:
                        print(f"[RunMetrics] Could not write final_audit.txt: {e}")

            self.final_metrics = _json_safe(compact) if compact else None
            self.status = status
            self.ended_at = _utc_now_iso()

            # Standalone full final metrics file (includes audit) for tooling / paper tables
            try:
                full_path = self.run_dir / "final_metrics.json"
                full_doc = _json_safe(final_metrics) if final_metrics else {}
                # Keep detailed_breakdown in the standalone file
                tmp = full_path.with_suffix(".json.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "run_id": self.run_id,
                            "status": self.status,
                            "ended_at": self.ended_at,
                            "run_params": _json_safe(self.run_params),
                            "final_metrics": full_doc,
                            "last_checkpoint": _json_safe(self.last_checkpoint),
                        },
                        f,
                        indent=2,
                    )
                tmp.replace(full_path)
            except Exception as e:
                print(f"[RunMetrics] Could not write final_metrics.json: {e}")

            self._write_json()
            qp = (self.final_metrics or {}).get("query_precision")
            rr = (self.final_metrics or {}).get("relevant_recall")
            f1 = (self.final_metrics or {}).get("f1_score")
            err = (self.final_metrics or {}).get("error")
            print(
                f"[RunMetrics] Finalized run → {self.run_dir} (status={status}) "
                f"QP={qp} RR={rr} F1={f1}"
                + (f" ERROR={err}" if err else "")
            )
            return self.last_checkpoint

    def one_line_status(self) -> str:
        c = self.last_checkpoint
        if not c:
            return "No metrics checkpoint yet."
        if c.get("metrics_available"):
            return (
                f"Running @ {c.get('tiles_processed')} tiles: "
                f"QP={c.get('query_precision')}  RR={c.get('relevant_recall')}  "
                f"F1={c.get('f1_score')}  "
                f"QR={c.get('query_rate')}  RelRate={c.get('relevant_rate')}  "
                f"secQR={c.get('section_query_rate')}  secRel={c.get('section_relevant_rate')}  "
                f"queries={c.get('ared_queries')}  frames={c.get('frames_read')}  "
                f"classes={c.get('classes_discovered_x_of_y')}"
            )
        return (
            f"Running @ {c.get('tiles_processed')} tiles: "
            f"queries={c.get('ared_queries')} frames={c.get('frames_read')}  "
            f"QR={c.get('query_rate')} secQR={c.get('section_query_rate')}  "
            f"(QP/RR/RelRate pending — need more DB labels)"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _section_bounds(self, tiles_now: int, queries_now: int) -> Dict[str, Any]:
        """Tiles / queries in the window since the previous checkpoint (or from stream start)."""
        prev = self.checkpoints[-1] if self.checkpoints else None
        tiles_prev = int(prev.get("tiles_processed") or 0) if prev else 0
        queries_prev = int(prev.get("ared_queries") or 0) if prev else 0
        section_tiles = max(0, tiles_now - tiles_prev)
        section_queries = max(0, queries_now - queries_prev)
        section_qr = (
            round(section_queries / section_tiles, 4) if section_tiles > 0 else 0.0
        )
        return {
            "section_tiles": section_tiles,
            "section_ared_queries": section_queries,
            "section_query_rate": section_qr,
            "_tiles_prev": tiles_prev,
        }

    def _section_relevant_rate(
        self,
        controller: "DroneAREDController",
        tiles_prev: int,
        tiles_now: int,
    ) -> Optional[float]:
        """Relevant rate over only the tiles in (tiles_prev, tiles_now] — paper Relevant Rate for that section."""
        if tiles_now <= tiles_prev:
            return None
        processed = list(getattr(controller, "processed_identities", None) or [])
        if not processed:
            return None
        # processed list is append-order stream; index aligns with tiles_processed growth
        lo = max(0, min(tiles_prev, len(processed)))
        hi = max(lo, min(tiles_now, len(processed)))
        section_keys = processed[lo:hi]
        if not section_keys:
            return None

        videos = sorted({p[0] for p in section_keys if p and p[0]})
        anns: List[Dict[str, Any]] = []
        for v in videos:
            anns.extend(self._load_annotations_for_video(controller, v))
        if not anns:
            return None

        # Filter annotations to this section of the stream
        keyset = set(section_keys)
        section_anns = [a for a in anns if make_tile_key(a) in keyset]
        if not section_anns:
            # No labels in this window yet — relevant rate unknown for labeled fraction
            return None

        relevant_classes = {
            str(a.get("label", "")) for a in section_anns if a.get("relevant")
        }
        # Paper relevant rate ≈ fraction of *streamed* points that are relevant-class.
        # Denominator = section stream size (tiles in window), not only labeled count.
        n_rel = sum(
            1
            for a in section_anns
            if str(a.get("label", "")) in relevant_classes
            and is_persistable_label(str(a.get("label", "")))
        )
        # Only count relevant among tiles we can identify; unlabeled stream tiles count as non-relevant for rate
        denom = max(1, hi - lo)
        return round(n_rel / denom, 4)

    def _build_snapshot(self, controller: "DroneAREDController", reason: str) -> Dict[str, Any]:
        stats = controller.stats or {}
        tiles = int(stats.get("tiles_processed", 0) or 0)
        frames = int(stats.get("frames_read", 0) or 0)
        queries = int(stats.get("ared_queries", 0) or 0)
        user_q = int(stats.get("user_queries", 0) or 0)
        cache_h = int(stats.get("cache_hits", 0) or 0)
        elapsed = round(time.time() - self._t0, 3)

        section = self._section_bounds(tiles, queries)
        tiles_prev = int(section.pop("_tiles_prev", 0))

        base: Dict[str, Any] = {
            "checkpoint_index": len(self.checkpoints) + 1,
            "reason": reason,
            "tiles_processed": tiles,
            "frames_read": frames,
            "ared_queries": queries,
            "user_queries": user_q,
            "cache_hits": cache_h,
            # Cumulative (stream so far)
            "query_rate": round(queries / max(1, tiles), 4) if tiles else 0.0,
            "relevant_rate": None,
            # Section (tiles since previous checkpoint)
            "section_tiles": section["section_tiles"],
            "section_ared_queries": section["section_ared_queries"],
            "section_query_rate": section["section_query_rate"],
            "section_relevant_rate": None,
            "query_precision": None,
            "relevant_recall": None,
            "f1_score": None,
            "n_should_query": None,
            "tp": None,
            "fp": None,
            "fn": None,
            "total_relevant_tiles": None,
            "total_relevant_tiles_queried": None,
            "classes_discovered_x_of_y": None,
            "n_classes_queried": None,
            "n_unique_classes": None,
            "ared_clusters": stats.get("ared_clusters"),
            "ared_known_labels": stats.get("ared_known_labels"),
            "elapsed_sec": elapsed,
            "current_video": stats.get("current_video", ""),
            "metrics_available": False,
            "note": None,
        }

        # Section relevant rate (needs DB labels for tiles in this window)
        try:
            base["section_relevant_rate"] = self._section_relevant_rate(
                controller, tiles_prev, tiles
            )
        except Exception as e:
            vprint(f"[RunMetrics] section_relevant_rate failed: {e}")

        # Label-only: counters only
        if getattr(controller, "label_only_mode", False):
            base["note"] = "label_only_mode (no A/RED query decisions)"
            return base

        try:
            result = self._compute_full_metrics(controller)
            if result and "error" not in result:
                summ = summarize_for_checkpoint(result)
                base.update({
                    "query_precision": summ.get("query_precision"),
                    "relevant_recall": summ.get("relevant_recall"),
                    "f1_score": summ.get("f1_score"),
                    "n_should_query": summ.get("n_should_query"),
                    "tp": summ.get("tp"),
                    "fp": summ.get("fp"),
                    "fn": summ.get("fn"),
                    "query_rate": summ.get("query_rate") if summ.get("query_rate") is not None else base["query_rate"],
                    "relevant_rate": summ.get("relevant_rate"),
                    "total_relevant_tiles": summ.get("total_relevant_tiles"),
                    "total_relevant_tiles_queried": summ.get("total_relevant_tiles_queried"),
                    "classes_discovered_x_of_y": summ.get("classes_discovered_x_of_y"),
                    "n_classes_queried": summ.get("n_classes_queried"),
                    "n_unique_classes": summ.get("n_unique_classes"),
                    "baseline_random_qp": summ.get("baseline_random_qp"),
                    "baseline_random_rr": summ.get("baseline_random_rr"),
                    "qp_improvement_ratio_vs_random": summ.get("qp_improvement_ratio_vs_random"),
                    "rr_improvement_ratio_vs_random": summ.get("rr_improvement_ratio_vs_random"),
                    "relevant_recall_strict": summ.get("relevant_recall_strict"),
                    "metrics_available": True,
                })
            else:
                err = (result or {}).get("error", "no labels yet")
                base["note"] = str(err)
        except Exception as e:
            base["note"] = f"metrics_error: {e}"
            print(f"[RunMetrics] Checkpoint metrics failed: {e}")

        return base

    def _compute_full_metrics(self, controller: "DroneAREDController") -> Optional[Dict[str, Any]]:
        """Evaluate over all processed tiles this run (multi-video aware)."""
        processed = list(getattr(controller, "processed_identities", None) or [])
        queried = list(getattr(controller, "queried_identities", None) or [])
        if not processed:
            return {"error": "no processed tiles"}

        # Annotations for every video that appears in processed keys
        videos = sorted({p[0] for p in processed if p and p[0]})
        anns: List[Dict[str, Any]] = []
        for v in videos:
            anns.extend(self._load_annotations_for_video(controller, v))

        if not anns:
            return {"error": "no annotations in DB for processed tiles yet"}

        ared_qc = {}
        if getattr(controller, "ared_adapter", None):
            try:
                ared_qc = controller.ared_adapter.get_query_counts() or {}
            except Exception:
                ared_qc = {}

        stream_total = len(processed)
        ared_query_count = int((controller.stats or {}).get("ared_queries", len(queried)))
        run_params = dict(self.run_params)

        result = ared_metrics.evaluate_from_annotations_and_queries(
            anns,
            queried,
            total_points=stream_total,
            ared_query_count_override=ared_query_count,
            processed_keys=processed,
            run_params=run_params,
            ared_query_counts=ared_qc,
        )
        result["n_processed_in_run"] = stream_total
        result["ared_queries_made"] = ared_query_count
        return result

    def _load_annotations_for_video(self, controller: "DroneAREDController", video_path: str) -> List[Dict[str, Any]]:
        sx = sy = None
        try:
            if controller.tiler:
                sx = getattr(controller.tiler, "stride_x", None)
                sy = getattr(controller.tiler, "stride_y", None)
            if (sx is None or sy is None) and controller.config:
                t = controller.config.tiling
                sx = getattr(t, "stride_x", None)
                sy = getattr(t, "stride_y", None)
        except Exception:
            pass

        anns: List[Dict[str, Any]] = []
        try:
            if getattr(controller, "annotation_manager", None):
                anns = controller.annotation_manager.get_annotations(video=video_path, use_scope=True) or []
                if not anns:
                    anns = controller.annotation_manager.get_annotations(video=video_path, use_scope=False) or []
            elif getattr(controller, "tile_db", None) is not None:
                anns = controller.tile_db.get_annotations_for_video(video_path, stride_x=sx, stride_y=sy) or []
                if not anns:
                    # basename fallback
                    from pathlib import Path as P
                    base = P(video_path).name
                    for v in controller.tile_db.list_videos():
                        if P(v).name == base:
                            anns = controller.tile_db.get_annotations_for_video(v, stride_x=sx, stride_y=sy) or []
                            break
        except Exception as e:
            print(f"[RunMetrics] annotation load failed for {video_path}: {e}")
            anns = []
        return anns

    def _init_csv(self) -> None:
        try:
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CHECKPOINT_CSV_FIELDS, extrasaction="ignore")
                w.writeheader()
        except Exception as e:
            print(f"[RunMetrics] CSV init failed: {e}")

    def _append_csv(self, snap: Dict[str, Any]) -> None:
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CHECKPOINT_CSV_FIELDS, extrasaction="ignore")
                w.writerow({k: snap.get(k) for k in CHECKPOINT_CSV_FIELDS})
                f.flush()
        except Exception as e:
            print(f"[RunMetrics] CSV append failed: {e}")

    def _document(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "run_dir": str(self.run_dir),
            "run_params": _json_safe(self.run_params),
            "checkpoints": _json_safe(self.checkpoints),
            "final_metrics": self.final_metrics,
            "notes": "",
        }

    def _write_json(self) -> None:
        if not self.enabled:
            return
        try:
            tmp = self._json_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._document(), f, indent=2)
            tmp.replace(self._json_path)
        except Exception as e:
            print(f"[RunMetrics] JSON write failed: {e}")
