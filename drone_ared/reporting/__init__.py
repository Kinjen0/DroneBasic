"""
Paper-style reporting over saved A/RED run packages (runs/<run_id>/).

Load run.json + checkpoints.csv, produce curves (QP/RR/F1 vs tiles),
comparison tables across κ / tile size / stride, and markdown reports.

Supports two tracks:
  - cumulative (default): stream-so-far quality at each checkpoint
  - batch: per-window quality for tiles since the previous checkpoint
"""

from .loader import (
    load_run,
    load_runs,
    discover_runs,
    RunRecord,
)
from .plots import (
    plot_run_curves,
    plot_batch_curves,
    plot_batch_vs_cumulative,
    plot_compare_metric,
    plot_query_burden,
)
from .report import (
    write_markdown_report,
    write_batch_markdown_report,
    runs_summary_table,
    batch_runs_summary_table,
    batch_checkpoints_table,
    batch_summary_stats,
    final_metrics_table,
)

__all__ = [
    "load_run",
    "load_runs",
    "discover_runs",
    "RunRecord",
    "plot_run_curves",
    "plot_batch_curves",
    "plot_batch_vs_cumulative",
    "plot_compare_metric",
    "plot_query_burden",
    "write_markdown_report",
    "write_batch_markdown_report",
    "runs_summary_table",
    "batch_runs_summary_table",
    "batch_checkpoints_table",
    "batch_summary_stats",
    "final_metrics_table",
]
