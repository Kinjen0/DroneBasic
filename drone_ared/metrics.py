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
import os
import random

from .label_sentinels import is_control_label, is_persistable_label


# -----------------------------------------------------------------------------
# Stable tile identity
# -----------------------------------------------------------------------------

def normalize_video_key(video: Any) -> str:
    """
    Portable video identity for metrics keys: **filename only**.

    Matches ``TileAnnotationDB._normalize_video_path`` so live pipeline keys
    (often absolute paths) join correctly with DB annotations (basename).
    """
    if video is None:
        return ""
    try:
        s = str(video).strip().replace("\\", "/")
        name = os.path.basename(s)
        return name if name else s
    except Exception:
        return str(video) if video is not None else ""


def normalize_tile_key(key: Tuple) -> Tuple:
    """Normalize the video component of a tile-key tuple to basename."""
    if not key:
        return key
    try:
        parts = list(key)
        if parts:
            parts[0] = normalize_video_key(parts[0])
        return tuple(parts)
    except Exception:
        return key


def make_tile_key(ann: Dict[str, Any]) -> Tuple:
    """
    Create a stable hashable key for a tile from annotation or meta dict.
    Matches the identity used in TileAnnotationDB (filename-only video key).
    """
    v = normalize_video_key(ann.get("video_path") or ann.get("video") or "")
    f = int(ann.get("abs_frame", ann.get("frame", -1)))
    r = int(ann.get("tile_row", ann.get("row", -1)))
    c = int(ann.get("tile_col", ann.get("col", -1)))
    w = int(ann.get("tile_width", ann.get("w", 0)))
    h = int(ann.get("tile_height", ann.get("h", 0)))
    return (v, f, r, c, w, h)


# -----------------------------------------------------------------------------
# Ground-truth "should query" computation
# -----------------------------------------------------------------------------

# Modes for counting "first appearance of a class" as a query-decision positive.
FIRST_OCCURRENCE_PAPER = "paper"           # paper: first of any class always counts
FIRST_OCCURRENCE_SKIP_KNOWN = "skip_known"  # warm-start: first only if class unknown to A_RED
FIRST_OCCURRENCE_AUTO = "auto"             # skip_known if known_classes non-empty else paper


def _norm_class_label(label: Any) -> str:
    """Normalize class names for membership tests (strip + casefold)."""
    try:
        return str(label).strip().casefold()
    except Exception:
        return str(label) if label is not None else ""


def _persistable_known_set(known_classes: Optional[Set[str]]) -> Set[str]:
    """Normalized set of persistable known class names for warm-start matching."""
    out: Set[str] = set()
    if not known_classes:
        return out
    for c in known_classes:
        if c is None:
            continue
        s = str(c)
        if is_persistable_label(s):
            n = _norm_class_label(s)
            if n:
                out.add(n)
    return out


def resolve_first_occurrence_mode(
    mode: Optional[str],
    known_classes: Optional[Set[str]] = None,
) -> str:
    """Map config mode (+ optional known set) to an effective paper|skip_known mode."""
    m = (mode or FIRST_OCCURRENCE_AUTO).strip().lower()
    if m in (FIRST_OCCURRENCE_PAPER, "cold", "cold_start"):
        return FIRST_OCCURRENCE_PAPER
    if m in (FIRST_OCCURRENCE_SKIP_KNOWN, "warm", "warm_start"):
        return FIRST_OCCURRENCE_SKIP_KNOWN
    # auto
    if known_classes:
        return FIRST_OCCURRENCE_SKIP_KNOWN
    return FIRST_OCCURRENCE_PAPER


def compute_should_query_from_annotations(
    annotations: List[Dict[str, Any]],
    order_by: str = "stream",
    known_classes: Optional[Set[str]] = None,
    first_occurrence_mode: str = FIRST_OCCURRENCE_PAPER,
) -> Dict[Tuple, bool]:
    """
    Given a list of annotations (as returned by TileAnnotationDB.get_annotations_for_video),
    return a dict: tile_key -> should_query (bool)

    Paper definition (SPIE Sec.5), cold-start / ``first_occurrence_mode="paper"``:
      Positives (should query) =
        i) the first sample of a given class, OR
       ii) samples from classes designated as relevant.

    Warm-start / loaded A_RED model (``first_occurrence_mode="skip_known"``):
      Rule (i) is adjusted: the first sample of a class counts **only if that class
      was not already known to A_RED** at run start (``known_classes``).
      A_RED is designed not to re-query classes it already has in memory; counting
      those first-in-video tiles as FN would unfairly punish a correct warm-start.
      Rule (ii) is unchanged — all samples of relevant-designated classes remain positives.

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

    # Normalized known set for robust matching against annotation labels
    known_norm = _persistable_known_set(known_classes)
    mode = resolve_first_occurrence_mode(first_occurrence_mode, known_norm)
    skip_known_firsts = (mode == FIRST_OCCURRENCE_SKIP_KNOWN)

    # relevant_classes may mix raw strings; match via normalized form too
    relevant_norm = {_norm_class_label(c) for c in relevant_classes if c}

    seen_classes: Set[str] = set()  # raw labels as they appear
    should_query: Dict[Tuple, bool] = {}

    for a in anns:
        key = make_tile_key(a)
        label = str(a.get("label", "")).strip()
        if not is_persistable_label(label):
            should_query[key] = False
            continue

        is_first = label not in seen_classes
        if is_first:
            seen_classes.add(label)

        lab_n = _norm_class_label(label)
        # First-of-class positive only when the class is new to A_RED (warm-start mode)
        # or always (paper / cold-start mode).
        first_counts = is_first and not (skip_known_firsts and lab_n in known_norm)

        is_from_relevant = lab_n in relevant_norm
        should = first_counts or is_from_relevant
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

    # Normalize so full-path pipeline keys match basename DB / should keys.
    actual_set = {normalize_tile_key(k) for k in (actual_queried or [])}
    should_query = {normalize_tile_key(k): v for k, v in (should_query or {}).items()}

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
    known_classes_at_start: Optional[Set[str]] = None,
    first_occurrence_mode: str = FIRST_OCCURRENCE_AUTO,
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
    known_classes_at_start: labels already known to A_RED when the run began
        (e.g. after Load ARED Model). Used with first_occurrence_mode to avoid
        treating first-in-video tiles of already-known classes as should-query FN.
    first_occurrence_mode: "paper" | "skip_known" | "auto" (see MetricsLoggingConfig).
    run_params: optional dict of key experiment settings for this run, e.g.:
        {"kappa": 1.0, "tile_size": (256,256), "frame_stride": 3,
         "annotation_db": "drone_tile_annotations.db", "dino_model": "...", ...}
        These are attached to the result and surfaced in the detailed audit.
    """
    # Normalize all identities to filename-only video keys so absolute paths from the
    # live pipeline match basename keys stored in the annotation DB.
    actual_queried_keys = [normalize_tile_key(k) for k in (actual_queried_keys or [])]
    if processed_keys:
        proc_set = {normalize_tile_key(k) for k in processed_keys}
        annotations = [a for a in annotations if make_tile_key(a) in proc_set]

    # Keep original display names for reporting; matching uses normalized form.
    known_display: List[str] = []
    if known_classes_at_start:
        seen_k = set()
        for c in known_classes_at_start:
            if c is None:
                continue
            s = str(c).strip()
            if not is_persistable_label(s):
                continue
            n = _norm_class_label(s)
            if n and n not in seen_k:
                seen_k.add(n)
                known_display.append(s)
    known_norm = _persistable_known_set(set(known_display) if known_display else None)
    effective_first_mode = resolve_first_occurrence_mode(first_occurrence_mode, known_norm)

    should = compute_should_query_from_annotations(
        annotations,
        known_classes=set(known_display) if known_display else None,
        first_occurrence_mode=effective_first_mode,
    )

    # Determine relevant classes at class level (consistent with updated should_query logic)
    relevant_classes: Set[str] = {
        str(a.get("label", "")).strip() for a in annotations
        if a.get("relevant") and is_persistable_label(str(a.get("label", "")).strip())
    }
    relevant_norm = {_norm_class_label(c) for c in relevant_classes}

    # Compute relevant-class-only stats for reference / detailed audit (not used for primary RR).
    # Primary RR now uses the full positives set (firsts of any class + rel samples).
    relevant_only_should = {}
    for a in annotations:
        key = make_tile_key(a)
        if _norm_class_label(str(a.get("label", "")).strip()) in relevant_norm:
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

    # Paper random baselines (SPIE eq.10–11 / IJSC §6.4):
    #   QueryPrecision_RDM ≈ Relevant Rate
    #   RelevantRecall_RDM = Query Rate
    query_rate = metrics["query_rate"]
    metrics["baseline_random_relevant_recall"] = round(query_rate, 4)  # RR_RDM = QR
    metrics["baseline_random_relevant_recall_note"] = (
        "Paper: RelevantRecall_RDM = Query Rate (fraction of stream queried at random)."
    )
    # Improvement ratios vs random (SPIE Fig.3 style)
    qp = metrics["query_precision"]
    rr = metrics["relevant_recall"]
    qp_rdm = baseline.get("random_query_precision_approx") or rel_rate
    rr_rdm = query_rate
    metrics["qp_improvement_ratio_vs_random"] = (
        round(qp / qp_rdm, 3) if qp_rdm and qp_rdm > 0 else None
    )
    metrics["rr_improvement_ratio_vs_random"] = (
        round(rr / rr_rdm, 3) if rr_rdm and rr_rdm > 0 else None
    )

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
        lab = str(a.get("label", "")).strip()
        if lab and is_persistable_label(lab) and lab not in seen:
            seen.add(lab)
            first_of_class[lab] = a.get("abs_frame", -1)

    n_first = len(first_of_class)
    # Firsts that still count as should-query under the active mode
    firsts_counting = {
        lab: fr for lab, fr in first_of_class.items()
        if not (
            effective_first_mode == FIRST_OCCURRENCE_SKIP_KNOWN
            and _norm_class_label(lab) in known_norm
        )
    }
    firsts_skipped_as_known = sorted(
        lab for lab in first_of_class
        if (
            effective_first_mode == FIRST_OCCURRENCE_SKIP_KNOWN
            and _norm_class_label(lab) in known_norm
        )
    )
    firsts_counting_names = sorted(firsts_counting.keys())
    # n_rel_pos already in metrics; keep for any legacy refs in detailed
    n_rel_should = len(relevant_only_should)  # alias for the RR positives count

    # Recompute should_query breakdown (approximate; main logic is in compute_should_query_from_annotations)
    should_from_first = 0
    should_from_relevant_only = 0
    should_from_both = 0
    for a in annotations:
        key = make_tile_key(a)
        if not should.get(key, False):
            continue
        lab = str(a.get("label", "")).strip()
        is_first_approx = lab in firsts_counting and firsts_counting.get(lab) == a.get("abs_frame", -1)
        is_rel_class = _norm_class_label(lab) in relevant_norm
        if is_first_approx and is_rel_class:
            should_from_both += 1
        elif is_first_approx:
            should_from_first += 1
        elif is_rel_class:
            should_from_relevant_only += 1

    # ------------------------------------------------------------------
    # Per-class report for RR / should-query (what the user can inspect)
    # ------------------------------------------------------------------
    # For each labeled class in this run: how many tiles, how many should-query
    # positives, TP/FN among those, whether first counts, relevant, known at start.
    per_class: Dict[str, Dict[str, Any]] = {}
    all_class_names = sorted(
        {str(a.get("label", "")).strip() for a in annotations
         if is_persistable_label(str(a.get("label", "")).strip())},
        key=lambda s: s.casefold(),
    )
    # Earliest should-positive tile key per class (for first_of_class reason tagging)
    first_should_key_by_class: Dict[str, Tuple] = {}
    for a in sorted(
        annotations,
        key=lambda x: (x.get("abs_frame", 0), x.get("tile_row", 0), x.get("tile_col", 0)),
    ):
        lab = str(a.get("label", "")).strip()
        if not is_persistable_label(lab):
            continue
        key = make_tile_key(a)
        if should.get(key, False) and lab not in first_should_key_by_class:
            first_should_key_by_class[lab] = key

    for lab in all_class_names:
        lab_n = _norm_class_label(lab)
        class_anns = [a for a in annotations if str(a.get("label", "")).strip() == lab]
        n_tiles = len(class_anns)
        n_should_c = 0
        n_tp_c = 0
        n_fn_c = 0
        n_queried_any = 0
        reasons = Counter()
        is_relevant_class = lab_n in relevant_norm
        first_counts_as_should = lab in firsts_counting

        for a in class_anns:
            key = make_tile_key(a)
            was_q = key in actual_set
            if was_q:
                n_queried_any += 1
            if not should.get(key, False):
                continue
            n_should_c += 1
            if was_q:
                n_tp_c += 1
            else:
                n_fn_c += 1
            # Why is this tile a should-query positive?
            if first_counts_as_should and first_should_key_by_class.get(lab) == key:
                reasons["first_of_class"] += 1
            if is_relevant_class:
                reasons["relevant_class_sample"] += 1

        known_at_start = lab_n in known_norm
        first_skipped = lab in firsts_skipped_as_known
        rr_class = round(n_tp_c / n_should_c, 4) if n_should_c > 0 else None

        per_class[lab] = {
            "n_tiles_in_eval": n_tiles,
            "n_should_query_positives": n_should_c,
            "tp": n_tp_c,   # should + queried
            "fn": n_fn_c,   # should + not queried
            "n_queried_any": n_queried_any,  # any query on this class (incl. non-should)
            "class_recall_on_should": rr_class,
            "is_relevant_class": is_relevant_class,
            "known_to_ared_at_run_start": known_at_start,
            "first_occurrence_frame": first_of_class.get(lab),
            "first_counts_as_should_positive": first_counts_as_should,
            "first_skipped_already_known": first_skipped,
            "positive_reason_counts": dict(reasons),
            "contributes_to_rr": n_should_c > 0,
        }

    # Human-readable lines for GUI / final_audit
    rr_class_report_lines: List[str] = []
    rr_class_report_lines.append(
        f"First-occurrence mode: {effective_first_mode} "
        f"(requested={first_occurrence_mode})"
    )
    rr_class_report_lines.append(
        f"A_RED known at run start ({len(known_display)}): "
        + (", ".join(sorted(known_display, key=str.casefold)) if known_display else "(none — cold start)")
    )
    rr_class_report_lines.append(
        f"Firsts COUNTED as should-query ({len(firsts_counting_names)}): "
        + (", ".join(firsts_counting_names) if firsts_counting_names else "(none)")
    )
    if firsts_counting_names:
        rr_class_report_lines.append(
            "  ↳ These classes appear in the *evaluation video labels* but were "
            "NOT in the A_RED model at Start (not in loaded/merged buffer). "
            "A_RED is expected to treat them as new — counting them as firsts is correct."
        )
    rr_class_report_lines.append(
        f"Firsts SKIPPED (already known) ({len(firsts_skipped_as_known)}): "
        + (", ".join(firsts_skipped_as_known) if firsts_skipped_as_known else "(none)")
    )
    rr_class_report_lines.append(
        f"Relevant-designated classes ({len(relevant_classes)}): "
        + (", ".join(sorted(relevant_classes, key=str.casefold)) if relevant_classes else "(none)")
    )
    # Classes in the eval video that the model already knew (good merge coverage)
    video_labs = set(all_class_names)
    model_labs_disp = set(known_display)
    novel_in_video = sorted(
        [lab for lab in video_labs if _norm_class_label(lab) not in known_norm],
        key=str.casefold,
    )
    covered_in_video = sorted(
        [lab for lab in video_labs if _norm_class_label(lab) in known_norm],
        key=str.casefold,
    )
    rr_class_report_lines.append(
        f"Eval-video classes already in model ({len(covered_in_video)}): "
        + (", ".join(covered_in_video) if covered_in_video else "(none)")
    )
    rr_class_report_lines.append(
        f"Eval-video classes NOVEL to model ({len(novel_in_video)}): "
        + (", ".join(novel_in_video) if novel_in_video else "(none)")
    )
    rr_class_report_lines.append("--- Per-class contribution to RR (should-query positives) ---")
    for lab in all_class_names:
        info = per_class[lab]
        if not info["contributes_to_rr"] and not info["first_skipped_already_known"]:
            # Still show skipped-known briefly only; skip pure background non-positives? Show all labeled.
            pass
        flags = []
        if info["is_relevant_class"]:
            flags.append("RELEVANT")
        if info["first_counts_as_should_positive"]:
            flags.append("FIRST_COUNTS")
        if info["first_skipped_already_known"]:
            flags.append("FIRST_SKIPPED_KNOWN")
        if info["known_to_ared_at_run_start"]:
            flags.append("KNOWN_AT_START")
        flag_s = ",".join(flags) if flags else "-"
        rr_s = (
            f"{info['class_recall_on_should']:.3f}"
            if info["class_recall_on_should"] is not None
            else "n/a"
        )
        rr_class_report_lines.append(
            f"  {lab}: tiles={info['n_tiles_in_eval']}  "
            f"should={info['n_should_query_positives']}  "
            f"TP={info['tp']} FN={info['fn']}  "
            f"class_RR={rr_s}  "
            f"queried_any={info['n_queried_any']}  "
            f"[{flag_s}]"
        )

    metrics["first_occurrence_mode"] = effective_first_mode
    metrics["first_occurrence_mode_requested"] = first_occurrence_mode
    metrics["known_classes_at_start"] = sorted(known_display, key=str.casefold)
    metrics["firsts_skipped_known_classes"] = firsts_skipped_as_known
    metrics["firsts_counting_as_should"] = firsts_counting_names
    metrics["per_class_rr_report"] = per_class
    metrics["rr_class_report_lines"] = rr_class_report_lines

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
        "N_FIRST_OF_CLASS_COUNTING_AS_SHOULD": len(firsts_counting),
        # Explicit names — this is what "3 class firsts" refers to under skip_known
        "FIRSTS_COUNTING_AS_SHOULD": firsts_counting,  # {class: first_frame}
        "FIRSTS_COUNTING_AS_SHOULD_NAMES": firsts_counting_names,
        "FIRSTS_COUNTING_EXPLANATION": (
            "Classes whose first tile in this eval still counts as a should-query positive. "
            "Under skip_known/auto these are classes present in the evaluation labels but "
            "absent from the A_RED model (loaded or merged) at run start — A_RED has never "
            "seen them, so a first-occurrence query is expected. They are NOT a merge bug "
            "unless those class names were supposed to be inside the merged model buffer."
        ),
        "FIRSTS_SKIPPED_ALREADY_KNOWN_TO_ARED": firsts_skipped_as_known,
        "KNOWN_CLASSES_AT_RUN_START": sorted(known_display, key=str.casefold),
        "EVAL_VIDEO_CLASSES_ALREADY_IN_MODEL": covered_in_video,
        "EVAL_VIDEO_CLASSES_NOVEL_TO_MODEL": novel_in_video,
        "FIRST_OCCURRENCE_MODE": effective_first_mode,
        "FIRST_OCCURRENCE_MODE_REQUESTED": first_occurrence_mode,
        "N_SHOULD_QUERY_TOTAL": metrics["n_should_query"],
        "SHOULD_BREAKDOWN": {
            "from_first_only_approx": should_from_first,
            "from_relevant_only_approx": should_from_relevant_only,
            "from_both_approx": should_from_both,
            "firsts_counting_names": firsts_counting_names,
            "firsts_skipped_names": firsts_skipped_as_known,
            "note": (
                "Positives = (first of class that counts under first_occurrence_mode) "
                "OR samples of relevant-designated classes. "
                f"Mode={effective_first_mode}: "
                + (
                    "first-of-class skipped when class was already known to A_RED at run start."
                    if effective_first_mode == FIRST_OCCURRENCE_SKIP_KNOWN
                    else "paper rule — first of any class always counts."
                )
                + " Filtered to tiles sent this run if processed_keys provided."
            ),
        },
        "PER_CLASS_RR_REPORT": per_class,
        "RR_CLASS_REPORT_LINES": rr_class_report_lines,
        "RELEVANT_CLASS_SAMPLES": n_rel_pos,
        "RELEVANT_TP": rel_tp,
        "RELEVANT_FN": rel_fn,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN_NOT_TRACKED": "We only care about query decisions (positives for the query task)",
        "QUERY_PRECISION_WORK": f"QP = {tp} / ({tp} + {fp}) = {metrics['query_precision']}   (TP/FP over should-query positives)",
        "RELEVANT_RECALL_WORK": f"RR = {tp} / ({tp} + {fn}) = {metrics['relevant_recall']}   (TP/(TP+FN) over should-query positives; firsts adjusted by mode={effective_first_mode})",
        "F1_WORK": f"F1 = 2 * QP * RR / (QP + RR) = {metrics['f1_score']}   (harmonic mean of QP and RR)",
        "RANDOM_BASELINE": baseline,
        "RANDOM_RR_EQUALS_QUERY_RATE": metrics.get("baseline_random_relevant_recall"),
        "QP_IMPROVEMENT_RATIO_VS_RANDOM": metrics.get("qp_improvement_ratio_vs_random"),
        "RR_IMPROVEMENT_RATIO_VS_RANDOM": metrics.get("rr_improvement_ratio_vs_random"),
        "TOTAL_STREAM_TILES_USED_FOR_RATES": total,
        "NOTE_ON_TOTAL": "total_points uses # tiles actually sent to A/RED this run (from processed). Only tiles actually sent are used for should/positives.",
        "PAPER_REFERENCES": "SPIE_IVSP_2026 Sec.5 eq.8-11 (positives = i first sample or ii relevant class); IJSC_2026-1 Alg.1 + evaluation; PerformanceMetricsPlan.md. Warm-start adjustment: first_occurrence_mode=skip_known."
    }

    # Also keep the older audit for compatibility
    metrics["audit"] = metrics["detailed_breakdown"]  # alias for display

    # Summary string for GUI
    mode_tag = f"firsts={effective_first_mode}"
    if firsts_skipped_as_known:
        mode_tag += f"(skip {len(firsts_skipped_as_known)} known)"
    if firsts_counting_names:
        mode_tag += f" count[{', '.join(firsts_counting_names)}]"
    metrics["summary"] = (
        f"QP={metrics['query_precision']:.3f}  "
        f"RR={metrics['relevant_recall']:.3f}  "
        f"F1={metrics['f1_score']:.3f}  "
        f"Classes={classes_discovered_str}  "
        f"A/RED Queries={n_queries}  "
        f"Total relevant queried={total_relevant_tiles_queried}/{total_relevant_tiles}  "
        f"vs Random≈{baseline['random_query_precision_approx']:.3f}  "
        f"[{mode_tag}]"
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
        "baseline_random_rr": result.get("baseline_random_relevant_recall"),
        "qp_improvement_ratio_vs_random": result.get("qp_improvement_ratio_vs_random"),
        "rr_improvement_ratio_vs_random": result.get("rr_improvement_ratio_vs_random"),
        "relevant_recall_strict": result.get("relevant_recall_strict"),
    }


# -----------------------------------------------------------------------------
# Helper for live runs: collect queried identities
# -----------------------------------------------------------------------------

def tile_identity_from_meta(meta: Dict[str, Any], tile: Any = None) -> Optional[Tuple]:
    """Create a stable key from the meta dict passed around the pipeline.

    Video component is normalized to filename only (same as the annotation DB).
    """
    if not meta:
        return None
    try:
        v = normalize_video_key(meta.get("video_path") or meta.get("video") or "")
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
