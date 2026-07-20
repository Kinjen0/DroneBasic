"""
Model merging for A_RED (drone adapter layer).

Three strategies, all built on top of ``AREDAdapter`` without touching
``A_REDimplementation/A_RED/``:

1. **Squish** — double (or enlarge) the buffer and replay *all* labeled points
   from both models (forced labels). "Stick the models together."

2. **Ingest** — rebuild model A, then stream model B's points through live
   A_RED. Only points that trigger a query receive B's label and enter the
   buffer. Smart forgetting applies when the buffer is full.

3. **Interleave** — start a fresh A_RED and alternate points from A and B.
   A_RED decides which points enter; donor labels answer real queries only.

Design notes
------------
- Strategy pattern (SOLID: open for new merge modes, closed for core A_RED).
- Clusters are never hand-edited; reconstruction always goes through
  ``process`` (replay or query-aware donor labels).
- Order matters (A-then-B ≠ B-then-A). Callers should document which is base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import pickle

import numpy as np

from .ared_adapter import AREDAdapter, AREDState
from .config import AREDConfig


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    """Outcome of a merge operation (live adapter + bookkeeping)."""

    adapter: AREDAdapter
    strategy_name: str
    points_from_a: int
    points_from_b: int
    points_accepted: int
    queries_during_merge: int
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def pretty(self) -> str:
        """Human-readable multi-line summary for GUI / CLI."""
        lines = [
            f"Strategy: {self.strategy_name}",
            f"Points from A: {self.points_from_a}",
            f"Points from B: {self.points_from_b}",
            f"Points in final buffer: {self.points_accepted}",
            f"Queries during merge: {self.queries_during_merge}",
            f"Clusters: {self.summary.get('num_clusters', '?')}",
            f"Known labels: {self.summary.get('known_labels', [])}",
            f"Buffer size: {self.summary.get('l_buf_size', '?')}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_ared_state(path: Union[str, Path]) -> AREDState:
    """Load a pickled ``AREDState`` from disk."""
    path = Path(path)
    with open(path, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, AREDState):
        # Tolerate plain objects that look like AREDState (older tooling).
        if hasattr(state, "labeled_points") and hasattr(state, "kappa"):
            return state  # type: ignore[return-value]
        raise TypeError(
            f"Expected AREDState in {path}, got {type(state).__name__}"
        )
    return state


def _copy_points(points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Defensive copy of labeled-point dicts (float32 embeddings)."""
    out: List[Dict[str, Any]] = []
    for p in points:
        emb = np.asarray(p["emb"], dtype=np.float32).reshape(-1).copy()
        out.append({
            "emb": emb,
            "label": str(p["label"]),
            "relevant": bool(p.get("relevant", False)),
        })
    return out


def points_in_order(state: AREDState) -> List[Dict[str, Any]]:
    """Return a defensive copy of ``state.labeled_points`` (oldest → newest)."""
    return _copy_points(getattr(state, "labeled_points", None) or [])


def _emb_dim_of(points: Sequence[Dict[str, Any]]) -> Optional[int]:
    if not points:
        return None
    try:
        return int(np.asarray(points[0]["emb"]).reshape(-1).shape[0])
    except Exception:
        return None


def validate_merge_compatible(
    base: AREDState,
    other: AREDState,
    *,
    allow_empty: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Check two states before merging.

    Returns
    -------
    errors : hard failures (caller should raise)
    warnings : soft issues (hyperparam mismatch, etc.)
    """
    errors: List[str] = []
    warnings: List[str] = []

    pts_a = getattr(base, "labeled_points", None) or []
    pts_b = getattr(other, "labeled_points", None) or []

    if not allow_empty:
        if len(pts_a) == 0:
            errors.append("Base model (A) has no labeled points.")
        if len(pts_b) == 0:
            errors.append("Other model (B) has no labeled points.")

    dim_a = _emb_dim_of(pts_a) if pts_a else getattr(base, "emb_dim", None)
    dim_b = _emb_dim_of(pts_b) if pts_b else getattr(other, "emb_dim", None)
    if dim_a is not None and dim_b is not None and int(dim_a) != int(dim_b):
        errors.append(
            f"Embedding dimension mismatch: A has dim={dim_a}, B has dim={dim_b}. "
            "Models must share the same feature extractor / tile setup."
        )

    # Soft hyperparam checks — we keep A's values; warn if B differs.
    for name, attr in (
        ("kappa", "kappa"),
        ("qs_var", "qs_var"),
        ("k_comp_pts", "k_comp_pts"),
    ):
        va = getattr(base, attr, None)
        vb = getattr(other, attr, None)
        if va is not None and vb is not None and va != vb:
            warnings.append(
                f"Hyperparam '{name}' differs (A={va}, B={vb}); using A's value."
            )

    return errors, warnings


def _config_from_base(
    base: AREDState,
    *,
    l_buf_size: Optional[int] = None,
    template: Optional[AREDConfig] = None,
) -> AREDConfig:
    """Build an ``AREDConfig`` from base-state hyperparams (+ optional overrides)."""
    tpl = template or AREDConfig()
    sf = getattr(base, "smart_forgetting_var", None)
    if sf is None:
        sf = getattr(tpl, "smart_forgetting_var", (3, 0.01))
    return AREDConfig(
        kappa=float(base.kappa),
        l_buf_size=int(l_buf_size if l_buf_size is not None else base.l_buf_size),
        k_comp_pts=int(base.k_comp_pts),
        qs_var=int(base.qs_var),
        data_aug_var=getattr(tpl, "data_aug_var", (0, (0, 0))),
        nghbhood_merge=getattr(tpl, "nghbhood_merge", True),
        singleton_merge=getattr(tpl, "singleton_merge", True),
        smart_forgetting_var=tuple(sf),
        verbose_flags=list(getattr(tpl, "verbose_flags", []) or []),
        data_augmentation_enabled=False,  # never aug during merge
        augmentation_rotations=list(getattr(tpl, "augmentation_rotations", [90, 180, 270])),
    )


def build_adapter_for_merge(
    base: AREDState,
    *,
    l_buf_size: Optional[int] = None,
    template: Optional[AREDConfig] = None,
) -> AREDAdapter:
    """Fresh adapter with base hyperparams; no GUI provider (donor/replay only)."""
    cfg = _config_from_base(base, l_buf_size=l_buf_size, template=template)
    return AREDAdapter(cfg)


def _buffer_count(adapter: AREDAdapter) -> int:
    try:
        return int(adapter.ared.l_buf.data_circular_buffer.count)
    except Exception:
        return len(adapter.export_labeled_points())


def _finish_result(
    adapter: AREDAdapter,
    *,
    strategy_name: str,
    points_from_a: int,
    points_from_b: int,
    queries_during_merge: int,
    warnings: List[str],
    extra_summary: Optional[Dict[str, Any]] = None,
) -> MergeResult:
    accepted = _buffer_count(adapter)
    summary: Dict[str, Any] = {
        "num_clusters": adapter.num_clusters,
        "known_labels": adapter.get_known_labels(),
        "l_buf_size": int(adapter.ared.l_buf.buffer_size),
        "num_points_processed": int(adapter.num_points_processed),
        "num_queries": int(adapter.num_queries),
    }
    if extra_summary:
        summary.update(extra_summary)
    return MergeResult(
        adapter=adapter,
        strategy_name=strategy_name,
        points_from_a=points_from_a,
        points_from_b=points_from_b,
        points_accepted=accepted,
        queries_during_merge=queries_during_merge,
        warnings=list(warnings),
        summary=summary,
    )


def _replay_point(adapter: AREDAdapter, point: Dict[str, Any]) -> Dict[str, Any]:
    """Force-label replay of one saved point (squish / base rebuild)."""
    return adapter.process(
        point["emb"],
        tile_image=None,
        meta={
            "replay": True,
            "label": point["label"],
            "relevant": point["relevant"],
        },
    )


def _donor_process(adapter: AREDAdapter, point: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stream one donor point: A_RED decides query/no-query; on query use donor label.
    """
    return adapter.process(
        point["emb"],
        tile_image=None,
        meta={
            "merge_donor": True,
            "label": point["label"],
            "relevant": point["relevant"],
        },
    )


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

class MergeStrategy(ABC):
    """Abstract merge strategy (Strategy pattern)."""

    name: str = "base"

    @abstractmethod
    def merge(
        self,
        base: AREDState,
        other: AREDState,
        *,
        template: Optional[AREDConfig] = None,
        **opts: Any,
    ) -> MergeResult:
        """Merge ``other`` into ``base`` (semantics depend on concrete strategy)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Squish — union of buffers with enlarged capacity
# ---------------------------------------------------------------------------

class SquishMergeStrategy(MergeStrategy):
    """
    Double the buffer (at least) and replay all points from A then B.

    Every labeled point is forced in via the replay path, so the result is a
    true union of the two buffers (subject only to capacity — which we size
    to fit).
    """

    name = "squish"

    def merge(
        self,
        base: AREDState,
        other: AREDState,
        *,
        template: Optional[AREDConfig] = None,
        buffer_multiplier: int = 2,
        **opts: Any,
    ) -> MergeResult:
        del opts
        errors, warnings = validate_merge_compatible(base, other)
        if errors:
            raise ValueError("Squish merge failed validation:\n- " + "\n- ".join(errors))

        pts_a = points_in_order(base)
        pts_b = points_in_order(other)
        n_a, n_b = len(pts_a), len(pts_b)

        # Double the larger configured buffer, but never smaller than the union.
        mult = max(1, int(buffer_multiplier))
        doubled = max(int(base.l_buf_size), int(other.l_buf_size)) * mult
        needed = n_a + n_b
        new_buf = max(doubled, needed, 1)

        adapter = build_adapter_for_merge(base, l_buf_size=new_buf, template=template)

        queries_before = adapter.num_queries
        for p in pts_a + pts_b:
            _replay_point(adapter, p)
        # Replay counts queries in adapter; for squish every point is "forced".
        queries_during = max(0, adapter.num_queries - queries_before)

        result = _finish_result(
            adapter,
            strategy_name=self.name,
            points_from_a=n_a,
            points_from_b=n_b,
            queries_during_merge=queries_during,
            warnings=warnings,
            extra_summary={
                "buffer_multiplier": mult,
                "order": "A_then_B",
                "mode": "forced_replay_union",
            },
        )
        # Stamp provenance on the in-memory state if caller saves later.
        try:
            st = adapter.to_state()
            st.merge_meta = {
                "strategy": self.name,
                "points_from_a": n_a,
                "points_from_b": n_b,
            }
            # merge_meta is not kept on the live adapter; caller uses result.summary
        except Exception:
            pass
        return result


# ---------------------------------------------------------------------------
# 2. Ingest — steam B into rebuilt A (query-aware)
# ---------------------------------------------------------------------------

class IngestMergeStrategy(MergeStrategy):
    """
    Rebuild model A, then stream each of B's points through live A_RED.

    - If A_RED queries → answer with B's saved label and add like a normal query.
    - If A_RED does not query → point is absorbed / skipped (not forced into buffer).
    - Smart forgetting (already configured on A) applies when the buffer is full.
    """

    name = "ingest"

    def merge(
        self,
        base: AREDState,
        other: AREDState,
        *,
        template: Optional[AREDConfig] = None,
        expand_buffer: bool = False,
        **opts: Any,
    ) -> MergeResult:
        del opts
        errors, warnings = validate_merge_compatible(base, other)
        if errors:
            raise ValueError("Ingest merge failed validation:\n- " + "\n- ".join(errors))

        pts_a = points_in_order(base)
        pts_b = points_in_order(other)
        n_a, n_b = len(pts_a), len(pts_b)

        buf_size = int(base.l_buf_size)
        if expand_buffer:
            buf_size = max(buf_size, n_a + n_b, buf_size * 2)
            warnings.append(f"expand_buffer=True → l_buf_size set to {buf_size}.")

        adapter = build_adapter_for_merge(base, l_buf_size=buf_size, template=template)

        # Phase 1: rebuild A with forced labels (identical to load_state).
        for p in pts_a:
            _replay_point(adapter, p)

        # Phase 2: stream B query-aware.
        queries_before = adapter.num_queries
        b_queried = 0
        b_not_queried = 0
        for p in pts_b:
            info = _donor_process(adapter, p)
            if info.get("queried"):
                b_queried += 1
            else:
                b_not_queried += 1

        queries_during = max(0, adapter.num_queries - queries_before)

        return _finish_result(
            adapter,
            strategy_name=self.name,
            points_from_a=n_a,
            points_from_b=n_b,
            queries_during_merge=queries_during,
            warnings=warnings,
            extra_summary={
                "mode": "rebuild_A_then_stream_B",
                "b_points_queried": b_queried,
                "b_points_not_queried": b_not_queried,
                "expand_buffer": bool(expand_buffer),
            },
        )


# ---------------------------------------------------------------------------
# 3. Interleave — alternate A/B into a fresh model (query-aware)
# ---------------------------------------------------------------------------

class InterleaveMergeStrategy(MergeStrategy):
    """
    Fresh A_RED; alternate points from A and B (then drain the longer list).

    Each point is query-aware (donor label only when A_RED queries). The first
    point always queries (library rule) and seeds the new model.
    """

    name = "interleave"

    def merge(
        self,
        base: AREDState,
        other: AREDState,
        *,
        template: Optional[AREDConfig] = None,
        start_with: str = "a",
        **opts: Any,
    ) -> MergeResult:
        del opts
        errors, warnings = validate_merge_compatible(base, other)
        if errors:
            raise ValueError("Interleave merge failed validation:\n- " + "\n- ".join(errors))

        pts_a = points_in_order(base)
        pts_b = points_in_order(other)
        n_a, n_b = len(pts_a), len(pts_b)

        start = (start_with or "a").strip().lower()
        if start not in ("a", "b"):
            warnings.append(f"Unknown start_with={start_with!r}; defaulting to 'a'.")
            start = "a"

        # Capacity: large enough that smart forgetting is not the main story,
        # but we do not force-double like squish (order/decision is the point).
        buf_size = max(int(base.l_buf_size), int(other.l_buf_size), n_a + n_b, 1)

        adapter = build_adapter_for_merge(base, l_buf_size=buf_size, template=template)

        if start == "a":
            primary, secondary = pts_a, pts_b
            order_label = "A0,B0,A1,B1,..."
        else:
            primary, secondary = pts_b, pts_a
            order_label = "B0,A0,B1,A1,..."

        queries_before = adapter.num_queries
        queried = 0
        not_queried = 0
        for p_primary, p_secondary in zip_longest(primary, secondary, fillvalue=None):
            for p in (p_primary, p_secondary):
                if p is None:
                    continue
                info = _donor_process(adapter, p)
                if info.get("queried"):
                    queried += 1
                else:
                    not_queried += 1

        queries_during = max(0, adapter.num_queries - queries_before)

        return _finish_result(
            adapter,
            strategy_name=self.name,
            points_from_a=n_a,
            points_from_b=n_b,
            queries_during_merge=queries_during,
            warnings=warnings,
            extra_summary={
                "mode": "interleaved_stream",
                "start_with": start,
                "order": order_label,
                "points_queried": queried,
                "points_not_queried": not_queried,
            },
        )


# ---------------------------------------------------------------------------
# Façade
# ---------------------------------------------------------------------------

# Registry of built-in strategies (easy to extend).
STRATEGY_REGISTRY: Dict[str, type] = {
    SquishMergeStrategy.name: SquishMergeStrategy,
    IngestMergeStrategy.name: IngestMergeStrategy,
    InterleaveMergeStrategy.name: InterleaveMergeStrategy,
    # Friendly aliases
    "steam": IngestMergeStrategy,
    "steam_into": IngestMergeStrategy,
    "stream_merge": InterleaveMergeStrategy,
    "stick": SquishMergeStrategy,
}


def get_strategy(name: str) -> MergeStrategy:
    """Instantiate a strategy by name (case-insensitive)."""
    key = (name or "").strip().lower()
    cls = STRATEGY_REGISTRY.get(key)
    if cls is None:
        known = ", ".join(sorted(set(STRATEGY_REGISTRY.keys())))
        raise ValueError(f"Unknown merge strategy {name!r}. Known: {known}")
    return cls()


class AREDModelMerger:
    """
    Façade for merging two A_RED model states.

    Example
    -------
    >>> merger = AREDModelMerger()
    >>> result = merger.merge_files("a.pkl", "b.pkl", strategy="squish")
    >>> result.adapter.save_state("merged.pkl")
    >>> print(result.pretty())
    """

    def __init__(self, template: Optional[AREDConfig] = None):
        self.template = template

    def merge(
        self,
        base: AREDState,
        other: AREDState,
        strategy: Union[str, MergeStrategy] = "squish",
        **opts: Any,
    ) -> MergeResult:
        """Merge ``other`` into ``base`` using the named (or concrete) strategy."""
        if isinstance(strategy, MergeStrategy):
            strat = strategy
        else:
            strat = get_strategy(str(strategy))
        return strat.merge(base, other, template=self.template, **opts)

    def merge_files(
        self,
        path_a: Union[str, Path],
        path_b: Union[str, Path],
        strategy: Union[str, MergeStrategy] = "squish",
        **opts: Any,
    ) -> MergeResult:
        """Load two pickles and merge (A = path_a is base)."""
        base = load_ared_state(path_a)
        other = load_ared_state(path_b)
        return self.merge(base, other, strategy=strategy, **opts)
