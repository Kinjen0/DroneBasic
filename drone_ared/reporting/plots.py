"""Paper-style matplotlib charts for A/RED run packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .loader import RunRecord

# Lazy import so CLI can still do tables without a display backend failing early
def _plt():
    import matplotlib
    matplotlib.use("Agg")  # headless-safe default
    import matplotlib.pyplot as plt
    return plt


def plot_run_curves(
    run: RunRecord,
    out_path: Union[str, Path],
    metrics: Sequence[str] = (
        "query_precision",
        "relevant_recall",
        "f1_score",
        "section_query_rate",
        "section_relevant_rate",
    ),
    title: Optional[str] = None,
    dpi: int = 140,
) -> Path:
    """
    QP / RR / F1 plus section query/relevant rates vs tiles processed.

    Section rates are for the last checkpoint window (~N tiles only), not cumulative
    stream rates (those stay in CSV/logs but are omitted here for readability).
    """
    plt = _plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        "query_precision": "Query Precision (QP)",
        "relevant_recall": "Relevant Recall (RR)",
        "f1_score": "F1 score",
        "section_query_rate": "Query rate (section)",
        "section_relevant_rate": "Relevant rate (section)",
        "ared_queries": "Cumulative queries",
    }
    # Solid for quality metrics; dashed for section rates
    metric_styles = {
        "query_precision": {"linestyle": "-", "linewidth": 1.8},
        "relevant_recall": {"linestyle": "-", "linewidth": 1.8},
        "f1_score": {"linestyle": "-", "linewidth": 1.8},
        "section_query_rate": {"linestyle": "--", "linewidth": 1.6},
        "section_relevant_rate": {"linestyle": "--", "linewidth": 1.6},
    }

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.4, 1.2]})
    ax0, ax1 = axes

    any_line = False
    for m in metrics:
        series = run.checkpoint_series(m)
        if not series:
            continue
        xs = [t for t, _ in series]
        ys = [v for _, v in series]
        style = metric_styles.get(m, {"linestyle": "-", "linewidth": 1.5})
        ax0.plot(
            xs,
            ys,
            marker="o",
            markersize=3,
            label=metric_labels.get(m, m),
            **style,
        )
        any_line = True

    if not any_line:
        ax0.text(0.5, 0.5, "No metric checkpoints with QP/RR yet\n(need DB labels during run)",
                 ha="center", va="center", transform=ax0.transAxes, fontsize=11, color="#666")
    else:
        ax0.legend(loc="best", fontsize=8, ncol=2)
        ax0.set_ylim(-0.02, 1.05)

    ax0.set_ylabel("Score / rate")
    ax0.set_title(title or f"Running metrics — {run.short_label()}")
    ax0.grid(True, alpha=0.3)
    ax0.axhline(0, color="#ccc", linewidth=0.5)
    ax0.axhline(1, color="#ccc", linewidth=0.5)

    # Lower panel: cumulative queries + section query rate (right axis if available)
    q_series = run.checkpoint_series("ared_queries")
    if q_series:
        ax1.plot([t for t, _ in q_series], [v for _, v in q_series],
                 color="#d62728", marker="s", markersize=3, linewidth=1.5, label="A/RED queries (cumul.)")
    sec_qr = run.checkpoint_series("section_query_rate")
    if sec_qr:
        ax1b = ax1.twinx()
        ax1b.plot(
            [t for t, _ in sec_qr],
            [v for _, v in sec_qr],
            color="#ff7f0e",
            linestyle=":",
            marker="^",
            markersize=3,
            linewidth=1.4,
            label="Section query rate",
        )
        ax1b.set_ylabel("Section QR", color="#ff7f0e")
        ax1b.tick_params(axis="y", labelcolor="#ff7f0e")
        ax1b.set_ylim(-0.02, max(1.05, max((v for _, v in sec_qr), default=0.1) * 1.15))
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        f_series = run.checkpoint_series("frames_read")
        if f_series:
            ax1_t = ax1.twinx()
            ax1_t.plot([t for t, _ in f_series], [v for _, v in f_series],
                       color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.7, label="Frames")
            ax1_t.set_ylabel("Frames read", color="#1f77b4")
            ax1_t.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.legend(loc="upper left", fontsize=8)

    ax1.set_xlabel("Tiles processed")
    ax1.set_ylabel("Queries")
    ax1.grid(True, alpha=0.3)

    # Param footer (video + A_RED model provenance for reproducibility)
    rp = run.run_params or {}
    vid = run.video_filename() or rp.get("video_filename") or "?"
    model_s = rp.get("ared_model_summary") or run.ared_model_label()
    footer = (
        f"video={vid}  model={model_s}  "
        f"κ={rp.get('kappa', '?')}  tile={rp.get('tile_size', '?')}  "
        f"stride=({rp.get('stride_x', '?')},{rp.get('stride_y', '?')})  "
        f"frame_stride={rp.get('frame_stride', '?')}  "
        f"l_buf={rp.get('l_buf_size', '?')}  status={run.status}"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=7.5, color="#444")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_compare_metric(
    runs: Sequence[RunRecord],
    metric: str = "relevant_recall",
    out_path: Union[str, Path] = "compare_rr.png",
    title: Optional[str] = None,
    dpi: int = 140,
) -> Path:
    """Overlay the same metric across multiple runs (e.g. RR vs tiles for different κ)."""
    plt = _plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = {
        "query_precision": "Query Precision (QP)",
        "relevant_recall": "Relevant Recall (RR)",
        "f1_score": "F1 score",
        "query_rate": "Query rate (cumul.)",
        "relevant_rate": "Relevant rate (cumul.)",
        "section_query_rate": "Query rate (section)",
        "section_relevant_rate": "Relevant rate (section)",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for run in runs:
        series = run.checkpoint_series(metric)
        if not series:
            continue
        ax.plot(
            [t for t, _ in series],
            [v for _, v in series],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=run.short_label(),
        )
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, f"No '{metric}' data in selected runs",
                ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(loc="best", fontsize=8)
        if metric in ("query_precision", "relevant_recall", "f1_score"):
            ax.set_ylim(-0.02, 1.05)

    ax.set_xlabel("Tiles processed")
    ax.set_ylabel(labels.get(metric, metric))
    ax.set_title(title or f"{labels.get(metric, metric)} across runs")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_query_burden(
    runs: Sequence[RunRecord],
    out_path: Union[str, Path] = "query_burden.png",
    dpi: int = 140,
) -> Path:
    """Bar chart: final query count / query rate / F1 for each run (paper-style summary)."""
    plt = _plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels: List[str] = []
    queries: List[float] = []
    rates: List[float] = []
    f1s: List[float] = []

    for run in runs:
        labels.append(run.short_label())
        q = run.final_value("ared_queries", "n_actual_queries", default=0) or 0
        r = run.final_value("query_rate", default=None)
        if r is None:
            tiles = run.final_value("tiles_processed", "total_points", default=0) or 0
            r = (float(q) / float(tiles)) if tiles else 0.0
        f1 = run.final_value("f1_score", default=0.0) or 0.0
        queries.append(float(q))
        rates.append(float(r))
        f1s.append(float(f1))

    if not labels:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No runs", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return out_path

    x = list(range(len(labels)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(x, queries, color="#d62728", alpha=0.85)
    axes[0].set_title("Total A/RED queries")
    axes[0].set_ylabel("Queries")

    axes[1].bar(x, rates, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Query rate")
    axes[1].set_ylabel("Queries / tile")

    axes[2].bar(x, f1s, color="#2ca02c", alpha=0.85)
    axes[2].set_title("Final F1")
    axes[2].set_ylabel("F1")
    axes[2].set_ylim(0, 1.05)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Query burden & final F1 by run", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_final_qp_rr_scatter(
    runs: Sequence[RunRecord],
    out_path: Union[str, Path] = "qp_rr_scatter.png",
    dpi: int = 140,
) -> Path:
    """Scatter of final QP vs RR, annotated by κ (classic paper comparison view)."""
    plt = _plt()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for run in runs:
        qp = run.final_value("query_precision")
        rr = run.final_value("relevant_recall")
        if qp is None or rr is None:
            continue
        ax.scatter([float(qp)], [float(rr)], s=60, alpha=0.85)
        ax.annotate(run.short_label(), (float(qp), float(rr)),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Query Precision (QP)")
    ax.set_ylabel("Relevant Recall (RR)")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Final QP vs RR")
    ax.grid(True, alpha=0.3)
    ax.plot([0, 1], [0, 1], linestyle=":", color="#aaa", linewidth=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
