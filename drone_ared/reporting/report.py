"""Tables and markdown reports for saved A/RED runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .loader import RunRecord


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    if isinstance(v, (list, tuple)):
        return "x".join(str(x) for x in v)
    return str(v)


def runs_summary_table(runs: Sequence[RunRecord]) -> str:
    """Markdown table: one row per run with key params + final QP/RR/F1."""
    headers = [
        "run_id",
        "video",
        "A_RED model",
        "status",
        "κ",
        "tile",
        "stride",
        "fs",
        "buf",
        "tiles",
        "queries",
        "QP",
        "RR",
        "F1",
        "classes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in runs:
        ts = r.tile_size
        tile_s = f"{ts[0]}x{ts[1]}" if ts else "—"
        sx, sy = r.param("stride_x"), r.param("stride_y")
        stride_s = f"{sx}x{sy}" if sx is not None else "—"
        vid = r.video_filename() or "—"
        model = r.param("ared_model_summary") or r.ared_model_label()
        row = [
            r.run_id[:36],
            str(vid)[:28],
            str(model)[:36],
            r.status,
            _fmt(r.kappa, 3),
            tile_s,
            stride_s,
            _fmt(r.frame_stride, 0),
            _fmt(r.l_buf_size, 0),
            _fmt(r.final_value("tiles_processed", default=(r.checkpoints[-1].get("tiles_processed") if r.checkpoints else None)), 0),
            _fmt(r.final_value("ared_queries", "n_actual_queries"), 0),
            _fmt(r.final_value("query_precision")),
            _fmt(r.final_value("relevant_recall")),
            _fmt(r.final_value("f1_score")),
            _fmt(r.final_value("classes_discovered_x_of_y"), 0),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def final_metrics_table(run: RunRecord) -> str:
    """Markdown key/value table for one run's final metrics + params."""
    lines = ["### Parameters", ""]
    lines.append("| param | value |")
    lines.append("| --- | --- |")
    for key in (
        "kappa",
        "kappa_effective",
        "ared_model_saved_kappa",
        "tile_size",
        "stride_x",
        "stride_y",
        "frame_stride",
        "l_buf_size",
        "k_comp_pts",
        "qs_var",
        "dino_model",
        "annotation_db",
        "data_augmentation_enabled",
        "label_cache_enabled",
        "metrics_checkpoint_every",
        "video_filename",
        "video_filenames",
        "video_paths",
        "ared_model_used",
        "ared_model_source",
        "ared_model_name",
        "ared_model_path",
        "ared_model_strategy",
        "ared_model_name_a",
        "ared_model_name_b",
        "ared_model_summary",
        "ared_known_labels_at_run_start",
        "first_occurrence_mode",
    ):
        if key in run.run_params:
            lines.append(f"| {key} | `{_fmt(run.run_params[key])}` |")

    lines += ["", "### Final metrics", ""]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    keys = [
        ("query_precision", "Query Precision (QP)"),
        ("relevant_recall", "Relevant Recall (RR)"),
        ("f1_score", "F1"),
        ("n_should_query", "Should-query positives"),
        ("tp", "TP"),
        ("fp", "FP"),
        ("fn", "FN"),
        ("query_rate", "Query rate (cumulative)"),
        ("relevant_rate", "Relevant rate (cumulative)"),
        ("n_actual_queries", "Actual queries"),
        ("total_points", "Total points (stream)"),
        ("total_relevant_tiles", "Relevant tiles"),
        ("total_relevant_tiles_queried", "Relevant tiles queried"),
        ("classes_discovered_x_of_y", "Classes queried x/y"),
        ("baseline_random_query_precision_approx", "Random baseline QP ≈"),
        ("baseline_random_relevant_recall", "Random baseline RR (= QR)"),
        ("summary", "Summary"),
    ]
    fm = run.final_metrics or {}
    # Fall back to last checkpoint for streaming counters
    last = run.checkpoints[-1] if run.checkpoints else {}
    for key, label in keys:
        v = fm.get(key)
        if v is None:
            v = last.get(key)
        if v is not None:
            lines.append(f"| {label} | {_fmt(v) if key != 'summary' else str(v)[:200]} |")

    if run.checkpoints:
        lines += ["", f"### Checkpoints ({len(run.checkpoints)})", ""]
        lines.append(
            "| # | reason | tiles | queries | QP | RR | F1 | QR | RelRate | secQR | secRel | bQP | bRR | bF1 |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        for c in run.checkpoints:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(c.get("checkpoint_index"), 0),
                        str(c.get("reason") or ""),
                        _fmt(c.get("tiles_processed"), 0),
                        _fmt(c.get("ared_queries"), 0),
                        _fmt(c.get("query_precision")),
                        _fmt(c.get("relevant_recall")),
                        _fmt(c.get("f1_score")),
                        _fmt(c.get("query_rate")),
                        _fmt(c.get("relevant_rate")),
                        _fmt(c.get("section_query_rate")),
                        _fmt(c.get("section_relevant_rate")),
                        _fmt(c.get("batch_query_precision")),
                        _fmt(c.get("batch_relevant_recall")),
                        _fmt(c.get("batch_f1_score")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def batch_summary_stats(run: RunRecord) -> Dict[str, Any]:
    """Mean/median/count of batch-window quality metrics for one run."""
    def _vals(field: str) -> List[float]:
        out: List[float] = []
        for c in run.checkpoints:
            if not c.get("batch_metrics_available") and c.get(field) is None:
                continue
            v = c.get(field)
            if v is None:
                continue
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out

    qp = _vals("batch_query_precision")
    rr = _vals("batch_relevant_recall")
    f1 = _vals("batch_f1_score")
    return {
        "n_batches_with_metrics": max(len(qp), len(rr), len(f1)),
        "mean_batch_qp": _mean(qp),
        "median_batch_qp": _median(qp),
        "mean_batch_rr": _mean(rr),
        "median_batch_rr": _median(rr),
        "mean_batch_f1": _mean(f1),
        "median_batch_f1": _median(f1),
    }


def batch_checkpoints_table(run: RunRecord) -> str:
    """Markdown table of per-window batch metrics only."""
    lines = [
        "| # | reason | tiles end | window | batch Q | bQP | bRR | bF1 | bQR | bRel | secQR | secRel |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in run.checkpoints:
        start = c.get("batch_tile_start")
        end = c.get("batch_tile_end")
        if start is not None and end is not None:
            window = f"{_fmt(start, 0)}–{_fmt(end, 0)}"
        else:
            st = c.get("section_tiles")
            window = f"Δ{_fmt(st, 0)}" if st is not None else "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(c.get("checkpoint_index"), 0),
                    str(c.get("reason") or ""),
                    _fmt(c.get("tiles_processed"), 0),
                    window,
                    _fmt(c.get("section_ared_queries"), 0),
                    _fmt(c.get("batch_query_precision")),
                    _fmt(c.get("batch_relevant_recall")),
                    _fmt(c.get("batch_f1_score")),
                    _fmt(c.get("batch_query_rate")),
                    _fmt(c.get("batch_relevant_rate")),
                    _fmt(c.get("section_query_rate")),
                    _fmt(c.get("section_relevant_rate")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def batch_runs_summary_table(runs: Sequence[RunRecord]) -> str:
    """One row per run with mean/median batch QP/RR/F1."""
    headers = [
        "run_id",
        "video",
        "κ",
        "batches",
        "mean bQP",
        "med bQP",
        "mean bRR",
        "med bRR",
        "mean bF1",
        "final QP",
        "final RR",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in runs:
        st = batch_summary_stats(r)
        row = [
            r.run_id[:36],
            str(r.video_filename() or "—")[:28],
            _fmt(r.kappa, 3),
            _fmt(st["n_batches_with_metrics"], 0),
            _fmt(st["mean_batch_qp"]),
            _fmt(st["median_batch_qp"]),
            _fmt(st["mean_batch_rr"]),
            _fmt(st["median_batch_rr"]),
            _fmt(st["mean_batch_f1"]),
            _fmt(r.final_value("query_precision")),
            _fmt(r.final_value("relevant_recall")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown_report(
    runs: Sequence[RunRecord],
    out_path: Union[str, Path],
    figure_paths: Optional[Dict[str, Path]] = None,
    title: str = "A/RED Drone Run Report",
    mode: str = "cumulative",
) -> Path:
    """Write a paper-oriented markdown report covering one or more runs.

    ``mode``:
      - ``cumulative`` (default): classic full-stream report (unchanged focus).
      - ``batch``: batch-window focused report.
      - ``both``: cumulative summary plus batch tables/figures.
    """
    mode = (mode or "cumulative").strip().lower()
    if mode not in ("cumulative", "batch", "both"):
        mode = "cumulative"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "batch":
        return write_batch_markdown_report(
            runs, out_path, figure_paths=figure_paths, title=title
        )

    parts: List[str] = [
        f"# {title}",
        "",
        f"Generated from **{len(runs)}** run package(s).",
        "",
        "## Summary table",
        "",
        runs_summary_table(runs),
        "",
    ]

    if mode == "both":
        parts += [
            "## Batch-window summary (mean / median over checkpoints)",
            "",
            batch_runs_summary_table(runs),
            "",
        ]

    if figure_paths:
        parts.append("## Figures")
        parts.append("")
        for name, p in figure_paths.items():
            # Prefer relative path from report dir when possible
            try:
                rel = Path(p).resolve().relative_to(out_path.parent.resolve())
            except Exception:
                rel = Path(p)
            parts.append(f"### {name}")
            parts.append("")
            parts.append(f"![{name}]({rel.as_posix()})")
            parts.append("")

    for r in runs:
        parts.append(f"## Run `{r.run_id}`")
        parts.append("")
        parts.append(f"- **status**: {r.status}")
        parts.append(f"- **started**: {r.started_at or '—'}")
        parts.append(f"- **ended**: {r.ended_at or '—'}")
        parts.append(f"- **dir**: `{r.run_dir}`")
        parts.append(f"- **has batch metrics**: {r.has_batch_metrics}")
        parts.append("")
        parts.append(final_metrics_table(r))
        parts.append("")
        if mode == "both" and r.checkpoints:
            parts.append("### Batch checkpoints")
            parts.append("")
            parts.append(batch_checkpoints_table(r))
            parts.append("")
            st = batch_summary_stats(r)
            parts.append(
                f"Batch aggregates: n={st['n_batches_with_metrics']}  "
                f"mean QP={_fmt(st['mean_batch_qp'])}  "
                f"mean RR={_fmt(st['mean_batch_rr'])}  "
                f"mean F1={_fmt(st['mean_batch_f1'])}"
            )
            parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(
        "Metrics follow the A/RED paper definitions: "
        "Query Precision = TP/(TP+FP), Relevant Recall = TP/(TP+FN) over first appearances "
        "+ relevant-class tiles. See `drone_ared/metrics.py` and SPIE_IVSP_2026 / IJSC_2026-1."
    )
    if mode == "both":
        parts.append(
            "Batch columns (bQP/bRR/bF1) are scores for each checkpoint window only; "
            "first-occurrence positives use full-stream context through the batch end."
        )
    parts.append("")

    text = "\n".join(parts)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def write_batch_markdown_report(
    runs: Sequence[RunRecord],
    out_path: Union[str, Path],
    figure_paths: Optional[Dict[str, Path]] = None,
    title: str = "A/RED Drone Batch Metrics Report",
) -> Path:
    """Alternate report focused on per-window (batch) QP/RR/F1."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: List[str] = [
        f"# {title}",
        "",
        f"Generated from **{len(runs)}** run package(s).",
        "",
        "Each batch is the tile window between consecutive metrics checkpoints "
        "(usually `metrics_checkpoint_every` tiles). QP/RR/F1 use the same paper "
        "formulas as cumulative metrics, but TP/FP/FN are restricted to that window. "
        "First-of-class positives still respect stream order from the start of the run "
        "(a class first seen earlier is not re-counted as a first inside a later batch).",
        "",
        "## Batch summary table",
        "",
        batch_runs_summary_table(runs),
        "",
    ]

    if figure_paths:
        parts.append("## Figures")
        parts.append("")
        for name, p in figure_paths.items():
            try:
                rel = Path(p).resolve().relative_to(out_path.parent.resolve())
            except Exception:
                rel = Path(p)
            parts.append(f"### {name}")
            parts.append("")
            parts.append(f"![{name}]({rel.as_posix()})")
            parts.append("")

    for r in runs:
        parts.append(f"## Run `{r.run_id}`")
        parts.append("")
        parts.append(f"- **status**: {r.status}")
        parts.append(f"- **started**: {r.started_at or '—'}")
        parts.append(f"- **ended**: {r.ended_at or '—'}")
        parts.append(f"- **dir**: `{r.run_dir}`")
        parts.append(f"- **has batch metrics**: {r.has_batch_metrics}")
        ckpt_every = r.param("metrics_checkpoint_every")
        if ckpt_every is not None:
            parts.append(f"- **checkpoint every**: {ckpt_every} tiles")
        parts.append("")
        st = batch_summary_stats(r)
        parts.append("### Batch aggregates")
        parts.append("")
        parts.append("| stat | value |")
        parts.append("| --- | --- |")
        for key, label in (
            ("n_batches_with_metrics", "Batches with metrics"),
            ("mean_batch_qp", "Mean batch QP"),
            ("median_batch_qp", "Median batch QP"),
            ("mean_batch_rr", "Mean batch RR"),
            ("median_batch_rr", "Median batch RR"),
            ("mean_batch_f1", "Mean batch F1"),
            ("median_batch_f1", "Median batch F1"),
        ):
            parts.append(f"| {label} | {_fmt(st[key])} |")
        parts.append("")
        parts.append(
            f"Final cumulative (reference): QP={_fmt(r.final_value('query_precision'))}  "
            f"RR={_fmt(r.final_value('relevant_recall'))}  "
            f"F1={_fmt(r.final_value('f1_score'))}"
        )
        parts.append("")
        if r.checkpoints:
            parts.append("### Per-batch checkpoints")
            parts.append("")
            parts.append(batch_checkpoints_table(r))
            parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(
        "Batch metrics: same paper QP/RR definitions as cumulative, evaluated per "
        "checkpoint window. See `drone_ared/metrics.py` (`evaluate_batch_window`) and "
        "`batches.csv` inside each run package."
    )
    parts.append("")

    text = "\n".join(parts)
    out_path.write_text(text, encoding="utf-8")
    return out_path
