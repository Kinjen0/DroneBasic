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
            "| # | reason | tiles | queries | QP | RR | F1 | QR | RelRate | secQR | secRel |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
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
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def write_markdown_report(
    runs: Sequence[RunRecord],
    out_path: Union[str, Path],
    figure_paths: Optional[Dict[str, Path]] = None,
    title: str = "A/RED Drone Run Report",
) -> Path:
    """Write a paper-oriented markdown report covering one or more runs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
        parts.append("")
        parts.append(final_metrics_table(r))
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(
        "Metrics follow the A/RED paper definitions: "
        "Query Precision = TP/(TP+FP), Relevant Recall = TP/(TP+FN) over first appearances "
        "+ relevant-class tiles. See `drone_ared/metrics.py` and SPIE_IVSP_2026 / IJSC_2026-1."
    )
    parts.append("")

    text = "\n".join(parts)
    out_path.write_text(text, encoding="utf-8")
    return out_path
