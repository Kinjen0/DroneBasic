#!/usr/bin/env python3
"""
Generate paper-style charts and reports from saved A/RED run packages.

Each pipeline Start (with metrics logging enabled) writes:
  runs/<run_id>/
    run.json
    checkpoints.csv      # cumulative + batch_* columns
    batches.csv          # batch-window extract
    final_audit.txt

Two metric tracks
-----------------
  cumulative (default)  QP/RR/F1 over all tiles from run start → checkpoint
  batch                 QP/RR/F1 for tiles since the previous checkpoint only

Examples
--------
  # Curves for the latest run under ./runs (cumulative)
  python run_report.py curves --runs-dir runs

  # Batch-window curves for a specific run
  python run_report.py batch-curves --run runs/20260714_...__kappa5__...

  # Compare RR across all runs (different κ, etc.)
  python run_report.py compare --runs-dir runs --metric relevant_recall

  # Compare batch RR across runs
  python run_report.py compare --runs-dir runs --metric batch_relevant_recall

  # Full markdown report + figures into reports/<timestamp>/
  python run_report.py report --runs-dir runs --out reports/latest

  # Batch-focused markdown report + batch figures
  python run_report.py batch-report --runs-dir runs --out reports/batch_latest

  # Print summary table only (no plots)
  python run_report.py table --runs-dir runs
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Allow running from repo root without install
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drone_ared.reporting import (
    load_run,
    load_runs,
    discover_runs,
    plot_run_curves,
    plot_batch_curves,
    plot_batch_vs_cumulative,
    plot_compare_metric,
    plot_query_burden,
    write_markdown_report,
    write_batch_markdown_report,
    runs_summary_table,
    batch_runs_summary_table,
)
from drone_ared.reporting.plots import plot_final_qp_rr_scatter


def _resolve_runs(args) -> list:
    paths: List[Path] = []
    if getattr(args, "run", None):
        for r in args.run:
            paths.append(Path(r))
    if paths:
        return load_runs(paths)
    runs_dir = Path(getattr(args, "runs_dir", "runs") or "runs")
    return load_runs(root=runs_dir)


def cmd_table(args) -> int:
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found. Start a pipeline run with metrics logging, or pass --run PATH.")
        return 1
    print(runs_summary_table(runs))
    return 0


def cmd_batch_table(args) -> int:
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found.")
        return 1
    print(batch_runs_summary_table(runs))
    return 0


def cmd_curves(args) -> int:
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found.")
        return 1
    out_dir = Path(args.out or "reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        out = out_dir / f"{run.run_id}_curves.png"
        path = plot_run_curves(run, out)
        print(f"Wrote {path}")
    return 0


def cmd_batch_curves(args) -> int:
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found.")
        return 1
    out_dir = Path(args.out or "reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        out = out_dir / f"{run.run_id}_batch_curves.png"
        path = plot_batch_curves(run, out)
        print(f"Wrote {path}")
        if args.also_vs_cumulative:
            for metric in ("query_precision", "relevant_recall", "f1_score"):
                p = plot_batch_vs_cumulative(
                    run,
                    metric=metric,
                    out_path=out_dir / f"{run.run_id}_batch_vs_cumul_{metric}.png",
                )
                print(f"Wrote {p}")
    return 0


def cmd_compare(args) -> int:
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found.")
        return 1
    out = Path(args.out or f"reports/compare_{args.metric}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    path = plot_compare_metric(runs, metric=args.metric, out_path=out)
    print(f"Wrote {path}")
    if args.also_burden:
        b = out.with_name(out.stem + "_burden.png")
        print(f"Wrote {plot_query_burden(runs, out_path=b)}")
    return 0


def cmd_report(args) -> int:
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found.")
        return 1
    out_dir = Path(args.out or f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    mode = getattr(args, "mode", None) or "cumulative"

    figure_paths = {}
    # Per-run curves
    for run in runs:
        p = plot_run_curves(run, fig_dir / f"{run.run_id}_curves.png")
        figure_paths[f"Curves — {run.short_label()}"] = p
        if mode in ("both",) and run.has_batch_metrics:
            bp = plot_batch_curves(run, fig_dir / f"{run.run_id}_batch_curves.png")
            figure_paths[f"Batch curves — {run.short_label()}"] = bp

    # Cross-run comparisons when >1 run
    if len(runs) > 1:
        for metric, title in (
            ("relevant_recall", "Relevant Recall comparison"),
            ("query_precision", "Query Precision comparison"),
            ("f1_score", "F1 comparison"),
            ("section_query_rate", "Section query rate comparison"),
            ("section_relevant_rate", "Section relevant rate comparison"),
        ):
            p = plot_compare_metric(
                runs, metric=metric, out_path=fig_dir / f"compare_{metric}.png", title=title
            )
            figure_paths[title] = p
        if mode == "both":
            for metric, title in (
                ("batch_relevant_recall", "Batch Relevant Recall comparison"),
                ("batch_query_precision", "Batch Query Precision comparison"),
                ("batch_f1_score", "Batch F1 comparison"),
            ):
                p = plot_compare_metric(
                    runs, metric=metric, out_path=fig_dir / f"compare_{metric}.png", title=title
                )
                figure_paths[title] = p
        figure_paths["Query burden"] = plot_query_burden(runs, fig_dir / "query_burden.png")
        figure_paths["QP vs RR"] = plot_final_qp_rr_scatter(runs, fig_dir / "qp_rr_scatter.png")
    else:
        figure_paths["Query burden"] = plot_query_burden(runs, fig_dir / "query_burden.png")

    md = write_markdown_report(
        runs,
        out_dir / "report.md",
        figure_paths=figure_paths,
        title=args.title or "A/RED Drone Run Report",
        mode=mode,
    )
    print(f"Report written to {md}")
    print(f"Figures in {fig_dir}")
    print()
    print(runs_summary_table(runs))
    return 0


def cmd_batch_report(args) -> int:
    runs = _resolve_runs(args)
    if not runs:
        print("No runs found.")
        return 1
    out_dir = Path(
        args.out or f"reports/batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    figure_paths = {}
    for run in runs:
        p = plot_batch_curves(run, fig_dir / f"{run.run_id}_batch_curves.png")
        figure_paths[f"Batch curves — {run.short_label()}"] = p
        if args.also_vs_cumulative:
            for metric in ("query_precision", "relevant_recall"):
                vp = plot_batch_vs_cumulative(
                    run,
                    metric=metric,
                    out_path=fig_dir / f"{run.run_id}_batch_vs_cumul_{metric}.png",
                )
                figure_paths[f"Batch vs cumul {metric} — {run.short_label()}"] = vp

    if len(runs) > 1:
        for metric, title in (
            ("batch_relevant_recall", "Batch Relevant Recall comparison"),
            ("batch_query_precision", "Batch Query Precision comparison"),
            ("batch_f1_score", "Batch F1 comparison"),
            ("section_query_rate", "Section query rate comparison"),
        ):
            p = plot_compare_metric(
                runs, metric=metric, out_path=fig_dir / f"compare_{metric}.png", title=title
            )
            figure_paths[title] = p

    md = write_batch_markdown_report(
        runs,
        out_dir / "batch_report.md",
        figure_paths=figure_paths,
        title=args.title or "A/RED Drone Batch Metrics Report",
    )
    print(f"Batch report written to {md}")
    print(f"Figures in {fig_dir}")
    print()
    print(batch_runs_summary_table(runs))
    return 0


def cmd_list(args) -> int:
    root = Path(args.runs_dir or "runs")
    dirs = discover_runs(root)
    if not dirs:
        print(f"No runs under {root.resolve()}")
        return 1
    for d in dirs:
        try:
            r = load_run(d)
            batch_tag = "batch=yes" if r.has_batch_metrics else "batch=no"
            print(
                f"{r.run_id:60s}  status={r.status:10s}  "
                f"ckpts={len(r.checkpoints):3d}  {batch_tag:9s}  {r.short_label()}"
            )
        except Exception as e:
            print(f"{d.name}: ERROR {e}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Paper-style charts & reports from drone A/RED run metrics packages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Shared parent so --runs-dir works before OR after the subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory containing run packages (default: runs)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # list
    sp = sub.add_parser("list", parents=[common], help="List discovered run packages")
    sp.set_defaults(func=cmd_list)

    # table
    sp = sub.add_parser("table", parents=[common], help="Print markdown summary table")
    sp.add_argument("--run", action="append", help="Specific run dir (repeatable)")
    sp.set_defaults(func=cmd_table)

    # batch-table
    sp = sub.add_parser(
        "batch-table",
        parents=[common],
        help="Print batch-window mean/median summary table",
    )
    sp.add_argument("--run", action="append", help="Specific run dir (repeatable)")
    sp.set_defaults(func=cmd_batch_table)

    # curves
    sp = sub.add_parser("curves", parents=[common], help="QP/RR/F1 vs tiles for each run")
    sp.add_argument("--run", action="append", help="Specific run dir (repeatable)")
    sp.add_argument("--out", default=None, help="Output directory for PNGs")
    sp.set_defaults(func=cmd_curves)

    # batch-curves
    sp = sub.add_parser(
        "batch-curves",
        parents=[common],
        help="Batch-window QP/RR/F1 vs tiles for each run",
    )
    sp.add_argument("--run", action="append", help="Specific run dir (repeatable)")
    sp.add_argument("--out", default=None, help="Output directory for PNGs")
    sp.add_argument(
        "--also-vs-cumulative",
        action="store_true",
        help="Also write cumulative vs batch overlay plots",
    )
    sp.set_defaults(func=cmd_batch_curves)

    # compare
    sp = sub.add_parser("compare", parents=[common], help="Overlay one metric across runs")
    sp.add_argument("--run", action="append", help="Specific run dir (repeatable)")
    sp.add_argument(
        "--metric",
        default="relevant_recall",
        choices=[
            "query_precision",
            "relevant_recall",
            "f1_score",
            "query_rate",
            "relevant_rate",
            "section_query_rate",
            "section_relevant_rate",
            "ared_queries",
            "batch_query_precision",
            "batch_relevant_recall",
            "batch_f1_score",
            "batch_query_rate",
            "batch_relevant_rate",
        ],
    )
    sp.add_argument("--out", default=None, help="Output PNG path")
    sp.add_argument("--also-burden", action="store_true", help="Also write query-burden bars")
    sp.set_defaults(func=cmd_compare)

    # report
    sp = sub.add_parser("report", parents=[common], help="Full markdown report + all figures")
    sp.add_argument("--run", action="append", help="Specific run dir (repeatable)")
    sp.add_argument("--out", default=None, help="Output directory")
    sp.add_argument("--title", default=None, help="Report title")
    sp.add_argument(
        "--mode",
        default="cumulative",
        choices=["cumulative", "batch", "both"],
        help="Report focus (default: cumulative; use both to include batch tables)",
    )
    sp.set_defaults(func=cmd_report)

    # batch-report
    sp = sub.add_parser(
        "batch-report",
        parents=[common],
        help="Batch-window markdown report + batch figures",
    )
    sp.add_argument("--run", action="append", help="Specific run dir (repeatable)")
    sp.add_argument("--out", default=None, help="Output directory")
    sp.add_argument("--title", default=None, help="Report title")
    sp.add_argument(
        "--also-vs-cumulative",
        action="store_true",
        help="Also include cumulative vs batch overlay figures",
    )
    sp.set_defaults(func=cmd_batch_report)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
