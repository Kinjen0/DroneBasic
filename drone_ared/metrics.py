"""
Performance Metrics for A/RED on Drone Footage.

Implements the exact metrics used in the A/RED papers:
- Query Precision (QP)
- Relevant Recall (RR)
- Random baseline at matched query budget

Definitions (directly from SPIE_IVSP_2026 and IJSC_2026-1):

A/RED is evaluated as a *binary classifier on the query / no-query decision*.

Positives (things that SHOULD cause a query):
- The first sample of any previously unseen class (new class discovery)
- Any sample from a class that the user has designated as "relevant"

Query Precision = TP / (TP + FP)
Relevant Recall = TP / (TP + FN)   (only considering relevant-class points)

"Precision" and "Recall" in the papers always refer to these query-decision metrics.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import defaultdict
import random


# -----------------------------------------------------------------------------
# Stable tile identity
# -----------------------------------------------------------------------------

def make_tile_key(ann: Dict[str, Any]) -> Tuple:
    """
    Create a stable hashable key for a tile from annotation or meta dict.
    Matches the identity used in TileAnnotationDB and Tile objects.
    """
    v = ann.get("video_path") or ann.get("video") or ""
    f = int(ann.get("abs_frame", ann.get("frame", -1)))
    r = int(ann.get("tile_row", ann.get("row", -1)))
    c = int(ann.get("tile_col", ann.get("col", -1)))
    w = int(ann.get("tile_width", ann.get("w", 0)))
    h = int(ann.get("tile_height", ann.get("h", 0)))
    return (v, f, r, c, w, h)


# -----------------------------------------------------------------------------
# Ground-truth "should query" computation
# -----------------------------------------------------------------------------

def compute_should_query_from_annotations(
    annotations: List[Dict[str, Any]],
    order_by: str = "stream"
) -> Dict[Tuple, bool]:
    """
    Given a list of annotations (as returned by TileAnnotationDB.get_annotations_for_video),
    return a dict: tile_key -> should_query (bool)

    A tile should be queried if:
      - It is the first time we have seen its *final* label in the stream order, OR
      - Its label was marked relevant.

    `order_by` controls how we determine "first seen":
      - "stream" : use the order they appear in the list (assumes caller sorted by frame/position)
      - "abs_frame": sort by abs_frame then row/col before processing
    """
    if not annotations:
        return {}

    anns = list(annotations)

    if order_by == "abs_frame":
        anns.sort(key=lambda a: (a.get("abs_frame", 0), a.get("tile_row", 0), a.get("tile_col", 0)))

    seen_classes: Set[str] = set()
    should_query: Dict[Tuple, bool] = {}

    for a in anns:
        key = make_tile_key(a)
        label = str(a.get("label", ""))
        is_relevant = bool(a.get("relevant", False))

        is_first = label not in seen_classes
        if is_first:
            seen_classes.add(label)

        should = is_first or is_relevant
        should_query[key] = should

    return should_query


# -----------------------------------------------------------------------------
# Core A/RED query-decision metrics
# -----------------------------------------------------------------------------

def compute_query_metrics(
    actual_queried: List[Tuple],
    should_query: Dict[Tuple, bool],
    total_points: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Query Precision and Relevant Recall.

    actual_queried: list of tile_keys that A/RED actually decided to query.
    should_query: dict from compute_should_query_from_annotations (or equivalent)
    total_points: total number of tiles considered (for rates). If None, uses len(should_query)

    Returns a dict with:
      - query_precision
      - relevant_recall
      - tp, fp, fn, tn
      - n_queries (actual)
      - n_should_query
      - relevant_rate (fraction of points that are relevant-class)
    """
    if total_points is None:
        total_points = len(should_query) or 1

    actual_set = set(actual_queried)

    tp = fp = fn = 0
    relevant_tp = relevant_fn = 0   # for recall over relevant points only

    for key, should in should_query.items():
        was_queried = key in actual_set

        if should and was_queried:
            tp += 1
        elif should and not was_queried:
            fn += 1
        elif not should and was_queried:
            fp += 1
        # tn not tracked explicitly

        # For Relevant Recall we only care about points where the *final* label is relevant.
        # We approximate by checking if the point was marked should_query *because of relevance*.
        # A better way is to also pass per-point "is_relevant" .
        # For now we use a simple heuristic: if it contributed to should_query and is relevant in spirit.
        # We improve this below with richer input.

    # Simpler and more accurate: we will also compute relevant-only numbers if caller provides richer data.
    # For this basic version we compute RR only over points that "should" be queried *and* are from relevant classes.
    # The caller can pass a separate relevant_should set if desired.

    n_queries = len(actual_set)
    n_should = sum(1 for v in should_query.values() if v)

    qp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Rough relevant rate (fraction of points that are positives for the query task)
    relevant_rate = n_should / max(1, total_points)

    return {
        "query_precision": round(qp, 4),
        "relevant_recall": round(rr, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_actual_queries": n_queries,
        "n_should_query": n_should,
        "total_points": total_points,
        "query_rate": round(n_queries / max(1, total_points), 4),
        "relevant_rate": round(relevant_rate, 4),
    }


def compute_relevant_recall_only(
    actual_queried: List[Tuple],
    should_query_relevant: Dict[Tuple, bool],   # only the points that are relevant-class
) -> float:
    """Relevant Recall computed strictly over relevant-class points."""
    if not should_query_relevant:
        return 0.0
    actual_set = set(actual_queried)
    tp = sum(1 for k in should_query_relevant if k in actual_set)
    total_relevant = len(should_query_relevant)
    return tp / total_relevant if total_relevant > 0 else 0.0


# -----------------------------------------------------------------------------
# Random baseline (as defined in the papers)
# -----------------------------------------------------------------------------

def random_baseline_at_budget(
    n_queries: int,
    n_total: int,
    relevant_rate: float,
) -> Dict[str, float]:
    """
    What Query Precision would a random querier achieve at the same number of queries?

    From the papers:
        QueryPrecision_RDM ≈ Relevant Rate   (when querying at the natural rate of relevant points)

    More precisely, for a random strategy that makes exactly `n_queries` queries:
        Expected TP ≈ n_queries * relevant_rate
        Query Precision ≈ relevant_rate   (for large n)

    We also return the "lucky" upper bound and expected values.
    """
    if n_total <= 0:
        n_total = 1

    n_relevant = int(round(relevant_rate * n_total))
    # Expected TP if we randomly pick n_queries points
    # Hypergeometric, but for reporting we use the simple approximation
    exp_tp = (n_queries / n_total) * n_relevant
    qp_random = exp_tp / n_queries if n_queries > 0 else 0.0

    # The papers often just state "≈ Relevant Rate"
    qp_random_approx = relevant_rate

    return {
        "random_query_precision": round(qp_random, 4),
        "random_query_precision_approx": round(qp_random_approx, 4),
        "random_expected_tp": round(exp_tp, 1),
        "n_queries": n_queries,
        "n_total": n_total,
        "relevant_rate": round(relevant_rate, 4),
    }


# -----------------------------------------------------------------------------
# Convenience: full evaluation from DB annotations + logged queries
# -----------------------------------------------------------------------------

def evaluate_from_annotations_and_queries(
    annotations: List[Dict[str, Any]],
    actual_queried_keys: List[Tuple],
    total_points: Optional[int] = None,
) -> Dict[str, Any]:
    """
    High-level helper.

    annotations: list from TileAnnotationDB.get_annotations_for_video(...)
    actual_queried_keys: list of tile keys that were actually queried by A/RED in this run
    """
    should = compute_should_query_from_annotations(annotations)

    # Build a "relevant only" should set for cleaner RR
    relevant_only_should = {}
    for a in annotations:
        key = make_tile_key(a)
        if should.get(key, False) and a.get("relevant"):
            relevant_only_should[key] = True

    metrics = compute_query_metrics(actual_queried_keys, should, total_points)
    metrics["relevant_recall_strict"] = round(
        compute_relevant_recall_only(actual_queried_keys, relevant_only_should), 4
    )

    n_queries = metrics["n_actual_queries"]
    total = metrics["total_points"]
    rel_rate = metrics["relevant_rate"]

    baseline = random_baseline_at_budget(n_queries, total, rel_rate)
    metrics.update({f"baseline_{k}": v for k, v in baseline.items()})

    # Summary string for GUI
    metrics["summary"] = (
        f"QP={metrics['query_precision']:.3f}  "
        f"RR={metrics['relevant_recall']:.3f}  "
        f"Queries={n_queries}/{total} ({metrics['query_rate']*100:.1f}%)  "
        f"vs Random≈{baseline['random_query_precision_approx']:.3f}"
    )
    return metrics


# -----------------------------------------------------------------------------
# Helper for live runs: collect queried identities
# -----------------------------------------------------------------------------

def tile_identity_from_meta(meta: Dict[str, Any], tile: Any = None) -> Optional[Tuple]:
    """Create a stable key from the meta dict passed around the pipeline."""
    if not meta:
        return None
    try:
        v = meta.get("video_path") or meta.get("video") or ""
        f = int(meta.get("abs_frame", meta.get("frame", -1)))
        r = int(meta.get("row", meta.get("tile_row", -1)))
        c = int(meta.get("col", meta.get("tile_col", -1)))
        w = int(meta.get("tile_width", 0) or (tile.width if tile and hasattr(tile, "width") else 0))
        h = int(meta.get("tile_height", 0) or (tile.height if tile and hasattr(tile, "height") else 0))
        if f < 0 or r < 0 or c < 0:
            return None
        return (v, f, r, c, w, h)
    except Exception:
        return None
