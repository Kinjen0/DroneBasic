"""Load and index saved run packages under runs/."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class RunRecord:
    """One completed (or in-progress) metrics run package."""

    run_id: str
    run_dir: Path
    status: str = "unknown"
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    run_params: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    final_metrics: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def param(self, key: str, default: Any = None) -> Any:
        return self.run_params.get(key, default)

    @property
    def kappa(self) -> Optional[float]:
        v = self.param("kappa")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def tile_size(self) -> Optional[tuple]:
        ts = self.param("tile_size")
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            return (int(ts[0]), int(ts[1]))
        return None

    @property
    def frame_stride(self) -> Optional[int]:
        v = self.param("frame_stride")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def l_buf_size(self) -> Optional[int]:
        v = self.param("l_buf_size")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def video_filename(self) -> Optional[str]:
        """Primary video basename for this run (portable; not full path)."""
        v = self.param("video_filename")
        if v:
            return str(v)
        names = self.param("video_filenames")
        if isinstance(names, (list, tuple)) and names:
            return str(names[0])
        paths = self.param("video_paths")
        if isinstance(paths, (list, tuple)) and paths:
            try:
                return Path(str(paths[0])).name
            except Exception:
                return str(paths[0])
        return None

    def ared_model_label(self) -> str:
        """Short tag for whether a preloaded/merged A_RED model was used."""
        if not self.param("ared_model_used"):
            # Fallback: known labels at start imply warm-start even if older runs lack flags
            known = self.param("ared_known_labels_at_run_start") or []
            if known:
                return "model=session"
            return "cold"
        src = str(self.param("ared_model_source") or "session")
        name = self.param("ared_model_name")
        if src == "merged":
            strat = self.param("ared_model_strategy") or "?"
            a = self.param("ared_model_name_a") or "?"
            b = self.param("ared_model_name_b") or "?"
            if name and str(name) not in (a, b):
                return f"merge:{strat}:{name}"
            return f"merge:{strat}({a}+{b})"
        if src == "loaded":
            return f"load:{name or self.param('ared_model_path') or '?'}"
        return f"model:{name or src}"

    def short_label(self) -> str:
        """Compact legend label for plots (includes video + model provenance)."""
        parts = []
        vid = self.video_filename()
        if vid:
            parts.append(vid)
        if self.kappa is not None:
            parts.append(f"κ={self.kappa:g}")
        ts = self.tile_size
        if ts:
            parts.append(f"tile={ts[0]}x{ts[1]}")
        sx = self.param("stride_x")
        sy = self.param("stride_y")
        if sx is not None and sy is not None:
            parts.append(f"s={sx}x{sy}")
        if self.frame_stride is not None:
            parts.append(f"fs={self.frame_stride}")
        if self.l_buf_size is not None:
            parts.append(f"buf={self.l_buf_size}")
        # Always show cold vs loaded/merged so multi-run plots are distinguishable
        parts.append(self.ared_model_label())
        if not parts:
            return self.run_id[:24]
        return " ".join(parts)

    def checkpoint_series(self, field_name: str) -> List[tuple]:
        """List of (tiles_processed, value) for a numeric checkpoint field."""
        out = []
        for c in self.checkpoints:
            tiles = c.get("tiles_processed")
            val = c.get(field_name)
            if tiles is None or val is None:
                continue
            try:
                out.append((int(tiles), float(val)))
            except (TypeError, ValueError):
                continue
        return out

    @property
    def has_batch_metrics(self) -> bool:
        """True if any checkpoint recorded batch-window QP/RR."""
        for c in self.checkpoints:
            if c.get("batch_metrics_available"):
                return True
            if c.get("batch_query_precision") is not None:
                return True
        return False

    def batch_series(self, metric: str = "query_precision") -> List[tuple]:
        """List of (tiles_processed, value) for a batch-window metric.

        ``metric`` may be a bare name (``query_precision``) or already prefixed
        (``batch_query_precision``).
        """
        field = metric if str(metric).startswith("batch_") else f"batch_{metric}"
        return self.checkpoint_series(field)

    def final_value(self, *keys: str, default: Any = None) -> Any:
        """Look up a final metric by trying keys in order (final_metrics then last checkpoint)."""
        fm = self.final_metrics or {}
        for k in keys:
            if k in fm and fm[k] is not None:
                return fm[k]
        if self.checkpoints:
            last = self.checkpoints[-1]
            for k in keys:
                if k in last and last[k] is not None:
                    return last[k]
        return default


def discover_runs(root: Union[str, Path] = "runs") -> List[Path]:
    """Find directories that contain a run.json (one level deep under root)."""
    root = Path(root)
    if not root.exists():
        return []
    found: List[Path] = []
    # Direct children
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "run.json").is_file():
            found.append(p)
    # Also accept root itself if it is a single run dir
    if (root / "run.json").is_file() and root not in found:
        found.insert(0, root)
    return found


def load_run(path: Union[str, Path]) -> RunRecord:
    """Load one run package from a directory or run.json path."""
    path = Path(path)
    if path.is_file() and path.name == "run.json":
        run_dir = path.parent
        json_path = path
    elif path.is_dir():
        run_dir = path
        json_path = path / "run.json"
    else:
        raise FileNotFoundError(f"Not a run directory or run.json: {path}")

    if not json_path.is_file():
        raise FileNotFoundError(f"Missing run.json in {run_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    checkpoints = list(data.get("checkpoints") or [])

    # Prefer CSV if present (may be slightly ahead during a live run)
    csv_path = run_dir / "checkpoints.csv"
    if csv_path.is_file():
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                csv_rows = []
                for row in reader:
                    csv_rows.append(_coerce_csv_row(row))
            if csv_rows:
                checkpoints = csv_rows
        except Exception as e:
            print(f"[reporting] CSV load warning for {csv_path}: {e}")

    return RunRecord(
        run_id=str(data.get("run_id") or run_dir.name),
        run_dir=run_dir.resolve(),
        status=str(data.get("status") or "unknown"),
        started_at=data.get("started_at"),
        ended_at=data.get("ended_at"),
        run_params=dict(data.get("run_params") or {}),
        checkpoints=checkpoints,
        final_metrics=data.get("final_metrics"),
        raw=data,
    )


def load_runs(
    paths: Optional[Sequence[Union[str, Path]]] = None,
    root: Union[str, Path] = "runs",
) -> List[RunRecord]:
    """Load many runs. If paths is None, discover everything under root."""
    if paths:
        return [load_run(p) for p in paths]
    dirs = discover_runs(root)
    out: List[RunRecord] = []
    for d in dirs:
        try:
            out.append(load_run(d))
        except Exception as e:
            print(f"[reporting] skip {d}: {e}")
    return out


def _coerce_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Convert CSV string cells to int/float/bool/None where obvious."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if v is None or v == "":
            out[k] = None
            continue
        s = str(v).strip()
        if s.lower() in ("true", "false"):
            out[k] = s.lower() == "true"
            continue
        try:
            if "." in s or "e" in s.lower():
                out[k] = float(s)
            else:
                out[k] = int(s)
            continue
        except ValueError:
            pass
        out[k] = s
    return out
