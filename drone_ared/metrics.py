"""
Performance Metrics for A/RED on Drone Footage.

Implements the exact metrics used in the A/RED papers:
- Query Precision (QP)
- Relevant Recall (RR)
- Random baseline at matched query budget

Definitions (directly from SPIE_IVSP_2026 Sec.5 and IJSC_2026-1):

A/RED is evaluated as a *binary classifier on the query / no-query decision*.

Positives (things that SHOULD cause a query) -- see paper:
  i) They are the first sample of a given class, or
 ii) they are samples from classes designated as relevant.

Query Precision (QP) = TP / (TP + FP)          -- over ALL positives above
Relevant Recall (RR) = TP / (TP + FN)          -- over ALL positives (first appearances of any class + samples from relevant classes)

"Precision" and "Recall" in the papers always refer to these query-decision metrics.
Random baseline: QP_RDM ≈ Relevant_Rate (fraction of points from relevant classes)
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import defaultdict
import random

from .label_sentinels import is_control_label, is_persistable_label


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

    Matches the paper definition exactly (SPIE Sec.5):
      Positives (should query) =
        i) the first sample of a given class, OR
       ii) samples from classes designated as relevant.

    "Designated as relevant" is at *class* level: if the user ever marked any
    instance of the class as relevant, then ALL samples of that class are positives.

    `order_by` controls how we determine "first seen":
      - "stream" : use the order they appear in the list (assumes caller sorted by frame/position)
      - "abs_frame": sort by abs_frame then row/col before processing
    """
    if not annotations:
        return {}

    anns = list(annotations)

    if order_by == "abs_frame":
        anns.sort(key=lambda a: (a.get("abs_frame", 0), a.get("tile_row", 0), a.get("tile_col", 0)))

    # Class-level designation: a class is "relevant" if any of its annotations were marked relevant.
    # This matches the paper language: "samples from classes designated as relevant."
    relevant_classes: Set[str] = {
        str(a.get("label", "")) for a in anns if a.get("relevant")
    }

    seen_classes: Set[str] = set()
    should_query: Dict[Tuple, bool] = {}

    for a in anns:
        key = make_tile_key(a)
        label = str(a.get("label", ""))

        is_first = label not in seen_classes
        if is_first:
            seen_classes.add(label)

        is_from_relevant = label in relevant_classes
        should = is_first or is_from_relevant
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
      - relevant_recall   # uses full positives (firsts of any class + relevant-class samples)
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

        # Note: relevant_recall (primary) uses the full set of positives defined for the task
        # (first sample of any class OR samples from relevant classes). See evaluate_... for
        # the relevant-class-only reference stats.

    n_queries = len(actual_set)
    n_should = sum(1 for v in should_query.values() if v)

    qp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 score (harmonic mean of Query Precision and Relevant Recall)
    f1 = 2 * qp * rr / (qp + rr) if (qp + rr) > 0 else 0.0

    # Rough relevant rate (fraction of points that are positives for the query task)
    relevant_rate = n_should / max(1, total_points)

    return {
        "query_precision": round(qp, 4),
        "relevant_recall": round(rr, 4),
        "f1_score": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_actual_queries": n_queries,
        "n_should_query": n_should,
        "total_points": total_points,
        "query_rate": round(n_queries / max(1, total_points), 4),
        "relevant_rate": round(relevant_rate, 4),
        # Raw counts for full audit (see papers + this file for formulas)
        "tp_fp_fn_detail": f"TP={tp} (should+queried), FP={fp} (queried but !should), FN={fn} (should but !queried)",
    }


def compute_relevant_recall_only(
    actual_queried: List[Tuple],
    should_query_relevant: Dict[Tuple, bool],   # only the points that are relevant-class
) -> float:
    """Recall computed strictly over samples from relevant-designated classes (for reference only; primary RR includes first appearances)."""
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
    ared_query_count_override: Optional[int] = None,
    processed_keys: Optional[List[Tuple]] = None,
    run_params: Optional[Dict[str, Any]] = None,
    ared_query_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    High-level helper. Produces exhaustive data for the user.

    annotations: list from TileAnnotationDB.get_annotations_for_video(...)
    actual_queried_keys: list of tile keys where A/RED decided it needed a label.
        IMPORTANT: per user instruction, cache-satisfied queries ARE treated as
        real user queries (they are just answered automatically to save time).
        The decision to query counts fully.
    processed_keys: if provided, filter annotations and should-computation to *only*
        the tiles that were actually sent to A/RED in this run. This ensures we
        only count positives (firsts + relevant) for tiles/classes that were actually presented
        to A/RED.
    run_params: optional dict of key experiment settings for this run, e.g.:
        {"kappa": 1.0, "tile_size": (256,256), "frame_stride": 3,
         "annotation_db": "drone_tile_annotations.db", "dino_model": "...", ...}
        These are attached to the result and surfaced in the detailed audit.
    """
    if processed_keys:
        proc_set = set(processed_keys)
        annotations = [a for a in annotations if make_tile_key(a) in proc_set]

    should = compute_should_query_from_annotations(annotations)

    # Determine relevant classes at class level (consistent with updated should_query logic)
    relevant_classes: Set[str] = {
        str(a.get("label", "")) for a in annotations if a.get("relevant")
    }

    # Compute relevant-class-only stats for reference / detailed audit (not used for primary RR).
    # Primary RR now uses the full positives set (firsts of any class + rel samples).
    relevant_only_should = {}
    for a in annotations:
        key = make_tile_key(a)
        if str(a.get("label", "")) in relevant_classes:
            relevant_only_should[key] = True

    metrics = compute_query_metrics(actual_queried_keys, should, total_points)

    # RR now uses the full set of positives (including first appearances of ANY class)
    # per the requirement to match the paper definition of positives for both QP and RR.
    # We still compute the "relevant classes only" version for reference / detailed stats.
    actual_set = set(actual_queried_keys)
    n_relevant_pos = len(relevant_only_should)
    relevant_tp = sum(1 for k in relevant_only_should if k in actual_set)
    relevant_fn = n_relevant_pos - relevant_tp
    strict_rr = round(relevant_tp / n_relevant_pos, 4) if n_relevant_pos > 0 else 0.0

    metrics["relevant_recall_strict"] = strict_rr
    # Do NOT override relevant_recall -- keep the broad version from compute_query_metrics
    # which does RR = TP / (TP + FN) over ALL positives (firsts of any + relevant samples)
    # Expose the relevant-only numbers for the detailed breakdown / auditing
    metrics["relevant_tp"] = relevant_tp
    metrics["relevant_fn"] = relevant_fn
    metrics["n_relevant_positives"] = n_relevant_pos

    # Use override if provided (from live stats ared_queries, which counts cache decisions)
    # Per user: cache queries are user queries and count fully for "total queries A_RED made"
    original_nq = metrics["n_actual_queries"]
    n_queries = ared_query_count_override if ared_query_count_override is not None else original_nq
    if ared_query_count_override is not None:
        metrics["n_actual_queries"] = n_queries

    total = metrics["total_points"]
    # Use *class-relevant point rate* for the random baseline (paper's "Relevant_Rate").
    # This is the fraction of streamed points whose final class was designated relevant.
    # (Broader n_should includes first-of-irrelevant too; paper baseline approx uses relevant anomalies rate.)
    n_rel_class_points = sum(1 for a in annotations if str(a.get("label", "")) in relevant_classes)
    rel_rate = n_rel_class_points / max(1, total) if total > 0 else 0.0
    metrics["relevant_rate"] = round(rel_rate, 4)
    # Recompute query_rate using authoritative n_queries (important for display)
    metrics["query_rate"] = round(n_queries / max(1, total), 4)

    baseline = random_baseline_at_budget(n_queries, total, rel_rate)
    metrics.update({f"baseline_{k}": v for k, v in baseline.items()})

    # === EVERY SINGLE DATAPOINT - FULL WORK SHOWN ===
    # Sources:
    # 1. annotations: every row in DB for the video (human labels + relevant flags). 
    #    These are the "people tagged" results.
    # 2. actual_queried_keys or ared_query_count: the points A/RED's algorithm emitted
    #    a query for (the "Total number of queries A_RED made"). Cache hits count here.
    # 3. should computed from final labels using paper rule.
    # Formulas exactly as SPIE_IVSP_2026 eq.8/9 and IJSC paper.

    # Raw class counts (ignore control-plane sentinels so metrics stay meaningful)
    from collections import Counter
    label_counts = Counter(
        str(a.get("label", "")) for a in annotations
        if is_persistable_label(str(a.get("label", "")))
    )
    # relevant class counts now based on designation (any tile of class marked rel designates the class)
    relevant_counts = Counter(
        str(a.get("label", "")) for a in annotations
        if str(a.get("label", "")) in relevant_classes and is_persistable_label(str(a.get("label", "")))
    )

    # Relevant-class stats (class-level: a class is relevant if any of its instances were marked relevant).
    # These replace the former hard-coded "person" counts per user request.
    # "Total relevant queried" = number of tiles whose label belongs to a relevant-designated class AND that were queried by A/RED.
    relevant_queried = sum(1 for a in annotations if str(a.get("label", "")) in relevant_classes and make_tile_key(a) in actual_set)
    total_relevant_tiles = sum(label_counts[lab] for lab in relevant_counts)
    total_relevant_tiles_queried = relevant_queried

    # Legacy "person" numbers are now aliased to the relevant numbers so existing display code
    # shows "Total relevant queried" instead of person-specific counts.
    total_person_tiles = total_relevant_tiles
    total_person_tiles_queried = total_relevant_tiles_queried

    # Firsts computation (exact order used for "should")
    seen = set()
    first_of_class = {}
    for a in sorted(annotations, key=lambda x: (x.get("abs_frame", 0), x.get("tile_row", 0), x.get("tile_col", 0))):
        lab = str(a.get("label", ""))
        if lab and lab not in seen:
            seen.add(lab)
            first_of_class[lab] = a.get("abs_frame", -1)

    n_first = len(first_of_class)
    # n_rel_pos already in metrics; keep for any legacy refs in detailed
    n_rel_should = len(relevant_only_should)  # alias for the RR positives count

    # Recompute should_query breakdown (approximate; main logic is in compute_should_query_from_annotations)
    should_from_first = 0
    should_from_relevant_only = 0
    should_from_both = 0
    for a in annotations:
        key = make_tile_key(a)
        if not should.get(key, False): continue
        lab = str(a.get("label", ""))
        is_first_approx = lab in first_of_class and first_of_class.get(lab) == a.get("abs_frame", -1)
        is_rel_class = lab in relevant_classes
        if is_first_approx and is_rel_class:
            should_from_both += 1
        elif is_first_approx:
            should_from_first += 1
        elif is_rel_class:
            should_from_relevant_only += 1

    # Full TP/FP/FN already in metrics, but we document
    tp = metrics["tp"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    rel_tp = metrics.get("relevant_tp", 0)
    rel_fn = metrics.get("relevant_fn", 0)
    n_rel_pos = metrics.get("n_relevant_positives", 0)

    # Build giant transparent report
    n_sent = len(processed_keys) if processed_keys is not None else None

    # x/y classes discovered this run (queried by A/RED vs total unique classes in the run's processed tiles)
    qc = dict(ared_query_counts or {})
    queried_class_names = [c for c, cnt in qc.items() if cnt > 0]
    n_classes_queried = len(queried_class_names)
    n_unique_in_run = len(label_counts)
    classes_discovered_str = f"{n_classes_queried}/{n_unique_in_run}"

    metrics["n_classes_discovered_this_run"] = n_classes_queried
    metrics["n_unique_classes_in_run"] = n_unique_in_run
    metrics["classes_discovered_x_of_y"] = classes_discovered_str

    metrics["detailed_breakdown"] = {
        "TOTAL_TILES_ACTUALLY_SENT_TO_ARED_THIS_RUN": n_sent if n_sent is not None else "N/A (no processed_keys; falling back to all DB annotations)",
        "TOTAL_LABELED_TILES_IN_DB": len(annotations),
        # The following TOTAL_PERSON_* keys are retained for compatibility with any external consumers.
        # Their values are now the relevant-class totals (see user request to show "Total relevant queried").
        "TOTAL_PERSON_TILES": total_relevant_tiles,
        "TOTAL_PERSON_TILES_QUERIED": total_relevant_tiles_queried,
        "TOTAL_RELEVANT_TILES": total_relevant_tiles,
        "TOTAL_RELEVANT_TILES_QUERIED": total_relevant_tiles_queried,
        "TOTAL_QUERIES_ARED_MADE": n_queries,   # NOTE: cache queries counted fully as user queries
        "TOTAL_ACTUAL_QUERIED_KEYS_LOGGED": len(actual_queried_keys),
        "CACHE_TREATED_AS_USER_QUERY": True,  # per explicit user instruction
        "UNIQUE_CLASSES": len(label_counts),
        "CLASS_COUNTS": dict(label_counts),
        "RELEVANT_CLASS_COUNTS": dict(relevant_counts),
        "CLASSES_DISCOVERED_X_Y": classes_discovered_str,
        "CLASSES_QUERIED_BY_ARED_THIS_RUN": sorted(queried_class_names),
        "N_CLASSES_QUERIED_THIS_RUN": n_classes_queried,
        "N_UNIQUE_CLASSES_IN_RUN": n_unique_in_run,
        "FIRST_OCCURRENCE_BY_CLASS": first_of_class,
        "N_FIRST_OF_CLASS": n_first,
        "N_SHOULD_QUERY_TOTAL": metrics["n_should_query"],
        "SHOULD_BREAKDOWN": {
            "from_first_only_approx": should_from_first,
            "from_relevant_only_approx": should_from_relevant_only,
            "from_both_approx": should_from_both,
            "note": "Exact should computed per paper rule (positives = first of any class OR samples of relevant classes). Filtered to only tiles sent in this run if processed_keys provided."
        },
        "RELEVANT_CLASS_SAMPLES": n_rel_pos,
        "RELEVANT_TP": rel_tp,
        "RELEVANT_FN": rel_fn,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN_NOT_TRACKED": "We only care about query decisions (positives for the query task)",
        "QUERY_PRECISION_WORK": f"QP = {tp} / ({tp} + {fp}) = {metrics['query_precision']}   (TP/FP over ALL positives: first-of-any-class OR rel-class samples)",
        "RELEVANT_RECALL_WORK": f"RR = {tp} / ({tp} + {fn}) = {metrics['relevant_recall']}   (TP / (TP + FN) over ALL positives, INCLUDING first appearances of classes)",
        "F1_WORK": f"F1 = 2 * QP * RR / (QP + RR) = {metrics['f1_score']}   (harmonic mean of QP and RR)",
        "RANDOM_BASELINE": baseline,
        "TOTAL_STREAM_TILES_USED_FOR_RATES": total,
        "NOTE_ON_TOTAL": "total_points uses # tiles actually sent to A/RED this run (from processed). Only tiles actually sent are used for should/positives (firsts of any class + relevant class samples).",
        "PAPER_REFERENCES": "SPIE_IVSP_2026 Sec.5 eq.8-11 (positives = i first sample or ii relevant class); IJSC_2026-1 Alg.1 + evaluation; PerformanceMetricsPlan.md"
    }

    # Also keep the older audit for compatibility
    metrics["audit"] = metrics["detailed_breakdown"]  # alias for display

    # Summary string for GUI
    metrics["summary"] = (
        f"QP={metrics['query_precision']:.3f}  "
        f"RR={metrics['relevant_recall']:.3f}  "
        f"F1={metrics['f1_score']:.3f}  "
        f"Classes={classes_discovered_str}  "
        f"A/RED Queries={n_queries}  "
        f"Total relevant queried={total_relevant_tiles_queried}/{total_relevant_tiles}  "
        f"vs Random≈{baseline['random_query_precision_approx']:.3f}"
    )

    # Attach caller-provided run parameters (kappa, tile size, frame stride, DB path, model, etc.)
    # so that every metrics report is self-describing for reproducibility.
    if run_params:
        metrics["run_params"] = dict(run_params)
        # Also fold a compact subset into the detailed audit for easy reading
        if "detailed_breakdown" in metrics:
            metrics["detailed_breakdown"]["RUN_PARAMS"] = {
                k: v for k, v in run_params.items()
                if k in ("kappa", "tile_size", "frame_stride", "annotation_db", "db_path",
                         "dino_model", "model_name", "stride_x", "stride_y",
                         "l_buf_size", "k_comp_pts", "data_augmentation_enabled")
            }
    else:
        metrics["run_params"] = None

    return metrics


def summarize_for_checkpoint(result: Dict[str, Any]) -> Dict[str, Any]:
    """Compact subset of evaluate_from_annotations_and_queries for run logs / CSV.

    Safe to call with partial results; missing keys become None.
    """
    audit = result.get("detailed_breakdown") or result.get("audit") or {}
    return {
        "query_precision": result.get("query_precision"),
        "relevant_recall": result.get("relevant_recall"),
        "f1_score": result.get("f1_score"),
        "n_should_query": result.get("n_should_query"),
        "tp": result.get("tp"),
        "fp": result.get("fp"),
        "fn": result.get("fn"),
        "query_rate": result.get("query_rate"),
        "relevant_rate": result.get("relevant_rate"),
        "n_actual_queries": result.get("n_actual_queries"),
        "total_points": result.get("total_points"),
        "total_relevant_tiles": audit.get("TOTAL_RELEVANT_TILES"),
        "total_relevant_tiles_queried": audit.get("TOTAL_RELEVANT_TILES_QUERIED"),
        "classes_discovered_x_of_y": result.get("classes_discovered_x_of_y"),
        "n_classes_queried": result.get("n_classes_discovered_this_run"),
        "n_unique_classes": result.get("n_unique_classes_in_run"),
        "summary": result.get("summary"),
        "baseline_random_qp": result.get("baseline_random_query_precision_approx"),
    }


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
