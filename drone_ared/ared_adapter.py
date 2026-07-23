"""
AREDAdapter - Bridge between our drone pipeline and the original A_RED / A_REDIN implementation.

CRITICAL DESIGN GOAL (per project constraint):
    Make **zero** (or at most trivial) modifications inside A_REDimplementation/A_RED/.

Investigation (2026-07-01): The original A_RED code (A_REDIN, main.py, visualizations, Oracle)
does **not** contain an interactive labeling GUI. It relies exclusively on a pre-populated
Oracle that holds ground-truth labels + relevance flags for an entire offline dataset
(used both to answer "queries" during streaming and for final accuracy metrics).
All "GUIs" in the original repo are matplotlib (TkAgg) plots for post-run analysis
(cluster evolution, t-SNE, query stats, etc.). No human-in-the-loop tile labeling existed.

We are therefore implementing the first practical interactive version that matches the
spirit of the A/RED papers (the algorithm decides *when* to query for a label+relevance;
a human or cache supplies the answer).

Strategy:
  - Use sys.path + sys.modules shim to import cleanly.
  - Monkey-patch / wrap at runtime to support:
      * Open-world / on-the-fly string class names (the original Oracle assumes all labels known upfront).
      * Interactive (GUI or cache) label provision instead of static ground-truth Oracle.
      * State save / load via replay (instead of trying to pickle the complex internal objects).

The original ARED expects:
  - Oracle with .answer_query(abs_idx) -> (label: str, relevance: bool)
  - Oracle with .num_classes, .int_str_label_bidict, .y[...] (used for conf_matrix and final metrics)
  - Data points are numpy vectors (embeddings).

In our case:
  - Labels are discovered live by a human SME via GUI.
  - We have no ground truth "y" array.
  - We still want ARED's clustering / query logic (ARED itself decides when a tile needs a label).

This file contains all adaptation logic so the library files stay untouched.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable, TYPE_CHECKING
import numpy as np
import pickle
import threading
import contextlib
from dataclasses import dataclass, field

from .label_sentinels import LabelCancelled, is_control_label, is_persistable_label
from .logutil import vprint

if TYPE_CHECKING:
    from .label_store import PersistentLabelStore
    from .config import AREDConfig
    from .augmentation import DINOAugmenter

# ------------------------------------------------------------------
# 1. ZERO-EDIT IMPORT SHIM
# ------------------------------------------------------------------
# The A_REDIN.py (and A_RED.py) contain:
#     from main import QS_VAR
# We satisfy it without touching any file in A_REDimplementation.

_ARED_DIR = Path(__file__).resolve().parents[1] / "A_REDimplementation" / "A_RED"

if str(_ARED_DIR) not in sys.path:
    sys.path.insert(0, str(_ARED_DIR))

# Provide a minimal "main" module that only needs to supply QS_VAR at import time.
# The actual QS_VAR value we care about is passed explicitly into ARED.__init__ anyway.
class _MainShim:
    QS_VAR = 1   # default that matches our AREDConfig; harmless

if "main" not in sys.modules:
    sys.modules["main"] = _MainShim()

# Now safe to import the original implementation
try:
    from A_REDIN import ARED as _OriginalARED
    from Oracle import Oracle as _OriginalOracle
except ImportError as e:
    raise ImportError(
        f"Failed to import original A_RED implementation from {_ARED_DIR}. "
        f"Make sure the folder structure is intact. Original error: {e}"
    ) from e


# ------------------------------------------------------------------
# 2. DYNAMIC LABEL SUPPORT (no changes to Oracle.py or A_REDIN.py)
# ------------------------------------------------------------------

class _GrowingLabelMap:
    """
    Wraps (or replaces) the bidict so that new string labels can be registered at runtime.

    The original Oracle builds a fixed bidict + fixed-size conf_matrix at init.
    We give ARED a large pre-sized conf_matrix and this object for the bidict.
    When ARED does `bidict[new_label]`, we auto-assign the next integer and grow as needed.
    """

    def __init__(self, initial_bidict: Any, conf_matrix_ref: Callable[[], np.ndarray], grow_conf: Callable[[int], None]):
        self._bidict = initial_bidict
        self._next_idx = len(initial_bidict)
        self._conf_matrix_ref = conf_matrix_ref
        self._grow_conf = grow_conf

    def __getitem__(self, label: str) -> int:
        if label not in self._bidict:
            idx = self._next_idx
            self._bidict[label] = idx
            # also make reverse lookup work (bidict supports it)
            try:
                self._bidict[idx] = label
            except Exception:
                pass  # some bidict versions are strict; we only need label->int for ARED
            self._next_idx += 1
            self._grow_conf(self._next_idx)
        return self._bidict[label]

    def __contains__(self, label: str) -> bool:
        return label in self._bidict

    def get(self, label: str, default: Optional[int] = None) -> Optional[int]:
        try:
            return self[label]
        except Exception:
            return default

    # Expose underlying for debug / serialization
    @property
    def raw(self):
        return self._bidict


class _SparseY:
    """
    Replacement for oracle.y so that non-queried points don't crash when ARED peeks:
        actual = self.oracle.y[some_abs_idx]
    We back-fill labels for every point we process (queried or assigned by cluster).
    """

    def __init__(self):
        self._data: Dict[int, List[Any]] = {}

    def __getitem__(self, idx: int):
        if idx not in self._data:
            # Safe fallback (should be overwritten by backfill in almost all cases)
            return ["__UNLABELED__", False]
        return self._data[idx]

    def __setitem__(self, idx: int, value: List[Any]):
        self._data[idx] = value

    def __len__(self):
        return len(self._data)

    def keys(self):
        return self._data.keys()


def _make_large_dummy_oracle(max_classes: int = 300) -> _OriginalOracle:
    """
    Create an Oracle with enough pre-registered dummy labels so the original
    conf_matrix allocation succeeds and we have room to grow.
    We never actually use the dummy data points.
    """
    dummy_labels = [f"__DUMMY_{i:04d}" for i in range(max_classes)]
    # Minimal X shape (will be ignored)
    X = np.zeros((max_classes, 1), dtype=np.float32)
    y = np.empty((max_classes, 2), dtype=object)
    y[:, 0] = dummy_labels
    y[:, 1] = False
    oracle = _OriginalOracle(X, y)
    return oracle


# ------------------------------------------------------------------
# 3. ARED ADAPTER
# ------------------------------------------------------------------

@dataclass
class AREDState:
    """Serializable snapshot for 'Save ARED Model' feature.

    Cluster structure is *not* pickled. On load we rebuild by replaying
    ``labeled_points`` through A/RED (same path as a warm-start).

    Optional fields below are backward-compatible: older pickles without them
    still load (defaults apply).
    """
    kappa: float
    qs_var: int
    k_comp_pts: int
    l_buf_size: int
    labeled_points: List[Dict[str, Any]] = field(default_factory=list)  # each: {emb, label, relevant}
    # Optional metadata (populated on newer saves; safe to ignore when missing)
    emb_dim: Optional[int] = None
    smart_forgetting_var: Optional[Tuple[int, float]] = None
    merge_meta: Optional[Dict[str, Any]] = None  # provenance after a model merge


class AREDAdapter:
    """
    High-level wrapper around the original ARED.

    Responsibilities:
    - Own the ARED instance + adapted Oracle.
    - Provide a clean `process(embedding, tile_image, metadata)` API.
    - Support interactive labeling via a provided callback (GUI or cache).
    - Support optional state save/load via replay.
    - Maintain counters useful for the GUI.

    Threading note:
    The `set_label_provider` callback is expected to be thread-safe (or to use
    the queue/event pattern implemented in the Controller). The adapter itself
    is not re-entrant; one processing thread at a time.
    """

    def __init__(self, config: "AREDConfig"):  # type: ignore  # forward ref ok because we import late
        from .config import AREDConfig  # local to avoid circular at top

        self.config = config
        self._dino_augmenter = None  # lazy, set when we have a feature extractor reference
        self._lock = threading.Lock()

        # --- Create large dummy oracle + ARED ---
        self.oracle = _make_large_dummy_oracle(400)
        self.ared: _OriginalARED = _OriginalARED(
            self.oracle,
            kappa=config.kappa,
            l_buf_size=config.l_buf_size,
            K_COMP_PTS=config.k_comp_pts,
            QS_VAR=config.qs_var,
            DATA_AUG_VAR=config.data_aug_var,
            NGHBHOOD_MERGE=config.nghbhood_merge,
            SINGLETON_MERGE=config.singleton_merge,
            SMART_FORGETTING_VAR=config.smart_forgetting_var,
            VERBOSE_FLAGS=config.verbose_flags,
        )

        # Patch for open world
        self._install_dynamic_label_support()

        # Public stats (updated by adapter, displayed in GUI)
        self.num_points_processed: int = 0
        self.num_queries: int = 0
        self.discovered_labels: set = set()
        # Per-class count of how many times A/RED decided to query for this label during *this run*.
        # This is what the GUI class lists should display for "queried by A/RED".
        self.query_counts: dict[str, int] = {}

        # Current label provider (set by Controller / tests)
        # Signature: (emb: np.ndarray, tile_img: Optional[Any], meta: dict) -> (label: str, relevant: bool)
        self._label_provider: Optional[Callable] = None

        # For replay on model load we temporarily use a non-interactive provider
        self._replay_mode: bool = False
        self._replay_labels: Dict[int, Tuple[str, bool]] = {}

    def _install_dynamic_label_support(self):
        """Replace bidict and y + pre-size conf_matrix after ARED construction."""
        # Make conf_matrix huge from the start
        max_c = 500
        self.ared.conf_matrix = np.zeros((max_c, max_c), dtype=int)

        def _grow_conf(new_size: int):
            cm = self.ared.conf_matrix
            if new_size > cm.shape[0]:
                new_cm = np.zeros((new_size + 64, new_size + 64), dtype=int)
                new_cm[:cm.shape[0], :cm.shape[1]] = cm
                self.ared.conf_matrix = new_cm

        growing_bidict = _GrowingLabelMap(
            self.oracle.int_str_label_bidict,
            lambda: self.ared.conf_matrix,
            _grow_conf
        )
        self.oracle.int_str_label_bidict = growing_bidict   # type: ignore
        self.oracle.y = _SparseY()  # type: ignore

        # Also expose num_classes as a property that grows
        orig_num = self.oracle.num_classes
        def _num_classes_getter(o=self.oracle):
            try:
                return len(o.int_str_label_bidict.raw)
            except Exception:
                return orig_num
        self.oracle.num_classes = property(_num_classes_getter)  # type: ignore

    def set_label_provider(self, provider: Callable[[np.ndarray, Optional[Any], Dict], Tuple[str, bool]]):
        """Register the function that will be called when ARED decides a query is needed."""
        self._label_provider = provider

    def set_label_store(self, store: Optional["PersistentLabelStore"]):
        """Optional: if set, the adapter will first try the store before calling the (GUI) provider."""
        self._label_store = store

    def set_feature_extractor(self, feature_extractor):
        """Optional: needed for DINO-based data augmentation (rotations)."""
        self._feature_extractor = feature_extractor
        if feature_extractor is not None:
            try:
                from .augmentation import DINOAugmenter
                self._dino_augmenter = DINOAugmenter(feature_extractor)
            except Exception:
                self._dino_augmenter = None

    # Internal reference for type checkers / later wiring
    _label_store: Optional["PersistentLabelStore"] = None
    _feature_extractor = None
    _dino_augmenter = None

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def process(self, embedding: np.ndarray, tile_image: Optional[Any] = None, meta: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Feed one tile embedding into ARED.

        If ARED internally decides this point requires a query (anomalous or near relevant),
        the registered label_provider will be invoked to obtain (label, relevant).

        Returns a small info dict for logging / GUI (cluster info, was_queried, etc.).
        """
        meta = meta or {}
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)

        vprint(f"[ARED] received tile for process (meta={meta})")
        with self._lock:
            abs_idx = self.ared.abs_index + 1   # what it will become inside

            # Decide whether we need a label now or ARED will decide inside process_*
            # We always provide a temporary answer_query that may call the provider.
            original_answer = getattr(self.oracle, "answer_query", None)

            was_queried = False
            obtained_label = None
            obtained_rel = None

            # Track invocations of answer_query during this single process() call.
            # ARED always does a "peek" answer_query (for stats) on every point.
            # Only on "real" query decisions (anomalous/near-relevant) does it call a 2nd time.
            # We must NOT force GUI/human on peeks, only on actual A/RED query decisions.
            hook_calls = [0]

            def _interactive_answer(_abs_index: int) -> Tuple[str, bool]:
                nonlocal was_queried, obtained_label, obtained_rel
                hook_calls[0] += 1
                call_num = hook_calls[0]

                # Special case for model load/replay: always honor the exact saved label
                # from meta, bypassing cache/provider. This ensures faithful reconstruction
                # of previous clusters.
                if meta and meta.get("replay"):
                    label = meta.get("label", f"replay_{_abs_index}")
                    rel = bool(meta.get("relevant", False))
                    obtained_label = label
                    obtained_rel = rel
                    self.oracle.y[_abs_index] = [label, rel]  # type: ignore
                    self.discovered_labels.add(label)
                    vprint(f"[ARED]   -> REPLAY using saved label '{label}' (relevant={rel})")
                    return label, rel

                is_peek = (call_num == 1 and self.num_points_processed > 0)

                # Model-merge donor path (ingest / interleave strategies):
                # A/RED still decides whether to query. On a *real* query we answer with
                # the donor model's saved (label, relevant). Peeks stay provisional so we
                # never force a label into the buffer without an A/RED query decision.
                if meta and meta.get("merge_donor") and not is_peek:
                    label = meta.get("label", f"donor_{_abs_index}")
                    rel = bool(meta.get("relevant", False))
                    if is_control_label(label):
                        raise LabelCancelled(f"merge_donor_control_label:{label}")
                    was_queried = True
                    obtained_label = label
                    obtained_rel = rel
                    self.oracle.y[_abs_index] = [label, rel]  # type: ignore
                    self.discovered_labels.add(label)
                    self.query_counts[label] = self.query_counts.get(label, 0) + 1
                    vprint(
                        f"[ARED]   -> MERGE_DONOR real query: label '{label}' "
                        f"(relevant={rel}) for abs_idx={_abs_index}"
                    )
                    return label, rel

                # Mark that A/RED decided this point needs a label query (for "user labels needed"
                # and for QP/RR). This must be counted EVEN IF the cache or exact DB satisfies it.
                # The cache is only for performance/testing convenience; the query decision itself
                # represents work that would be needed from a user in a cold-start or no-cache scenario.
                if not is_peek:
                    was_queried = True
                    vprint(f"[ARED] A_RED decided to QUERY (call#{call_num}) for abs_idx={_abs_index} (tile meta: {meta}). will count for labels-needed + metrics (cache may still satisfy)")

                # 1. Try persistent cache first -- always, for both peek and real queries
                if getattr(self, "_label_store", None) is not None:
                    cached = self._label_store.lookup(emb)
                    if cached is not None:
                        label, rel = cached
                        # Never accept control sentinels from a poisoned cache
                        if is_control_label(label):
                            vprint(f"[ARED]   -> Cache HIT is control sentinel '{label}' — ignoring (treating as miss).")
                        else:
                            obtained_label = label
                            obtained_rel = rel
                            self.oracle.y[_abs_index] = [label, rel]  # type: ignore
                            self.discovered_labels.add(label)
                            if not is_peek:
                                self.query_counts[label] = self.query_counts.get(label, 0) + 1
                            if is_peek:
                                vprint(f"[ARED]   -> Cache HIT on peek for abs_idx={_abs_index}. Auto (no GUI).")
                            else:
                                vprint(f"[ARED]   -> Cache HIT for this QUERY decision. Auto-labeled as '{label}' (relevant={rel}). (still counts as ARED query for labels-needed metric)")
                            return label, rel
                    else:
                        if is_peek:
                            vprint(f"[ARED]   -> Cache MISS on peek (will use provisional, no GUI).")
                        else:
                            vprint(f"[ARED]   -> Cache MISS for real query. Will request from provider (GUI).")

                if is_peek:
                    # Peek call (internal ARED accounting for non-queried points).
                    # Do NOT call human/GUI provider here -- that would query on every tile.
                    # Supply a provisional so ARED can continue; real cluster label may be back-filled later.
                    # Use a private provisional that is never persisted as a real class name.
                    label = meta.get("label") if meta.get("label") and not is_control_label(meta.get("label")) else f"__PEEK_{abs_idx % 100}"
                    rel = bool(meta.get("relevant", False))
                    obtained_label = label
                    obtained_rel = rel
                    self.oracle.y[_abs_index] = [label, rel]  # type: ignore
                    vprint(f"[ARED] Peek answer_query (call#{call_num}) for abs_idx={_abs_index} -- provisional (no oracle query to user).")
                    return label, rel

                # Real query path (non-peek) -- provider will be called (or exact DB in pipeline layer)
                vprint(f"[ARED] Real QUERY path reached provider for abs_idx={_abs_index}")

                # 2. Fall back to the registered provider (normally the GUI) -- only for real queries
                if self._label_provider is None:
                    # Fallback for headless / synthetic tests
                    label = meta.get("label", f"auto_{abs_idx % 5}")
                    rel = bool(meta.get("relevant", False))
                    if is_control_label(label):
                        raise LabelCancelled(f"fallback_control_label:{label}")
                    vprint(f"[ARED]   -> No provider, using fallback label '{label}' (relevant={rel})")
                else:
                    try:
                        label, rel = self._label_provider(emb, tile_image, meta)
                    except LabelCancelled:
                        # Propagate so process() can abort without learning a junk class
                        raise
                    vprint(f"[ARED]   -> Provider returned label '{label}' (relevant={rel})")

                # Hard refuse control sentinels even if provider returned them as strings
                if not is_persistable_label(label):
                    vprint(f"[ARED]   -> REFUSING control/empty label '{label}' (will not learn or store).")
                    raise LabelCancelled(f"control_label:{label}")

                obtained_label = label
                obtained_rel = rel

                # Record + also feed back into the store so identical future tiles are cached
                self.oracle.y[_abs_index] = [label, rel]  # type: ignore
                self.discovered_labels.add(label)
                # Count this A/RED query decision for the class (what the user wants to see in the class lists).
                self.query_counts[label] = self.query_counts.get(label, 0) + 1
                if getattr(self, "_label_store", None) is not None:
                    try:
                        self._label_store.add(emb, label, rel)
                        vprint(f"[ARED]   -> Added to label store for future cache hits.")
                    except Exception as e:
                        vprint(f"[ARED]   -> WARNING: Failed to add to label store: {e}")
                return label, rel

            # Temporarily install
            self.oracle.answer_query = _interactive_answer  # type: ignore

            cancelled_reason: Optional[str] = None
            # Track buffer occupancy so replay can force-insert if A_RED skipped
            # the query branch (non-anomalous / not near-relevant). Saved models only
            # contain previously labeled points; warm-start must restore all of them.
            buf_count_before = 0
            try:
                buf_count_before = int(self.ared.l_buf.data_circular_buffer.count)
            except Exception:
                buf_count_before = 0

            try:
                if self.num_points_processed == 0:
                    self.ared.process_first_point(emb)
                else:
                    self.ared.process_point(emb)
            except LabelCancelled as e:
                # Stop / timeout / skip during a real query: do NOT learn a fake class.
                # Best-effort: leave this point without a durable discovered label.
                cancelled_reason = getattr(e, "reason", None) or str(e)
                vprint(f"[ARED] LabelCancelled during process (reason={cancelled_reason}). "
                      f"Not learning control labels; aborting this point cleanly.")
                was_queried = False
                obtained_label = None
                obtained_rel = None
            finally:
                # Restore
                if original_answer is not None:
                    self.oracle.answer_query = original_answer
                else:
                    # Remove our monkey if it was the first
                    if hasattr(self.oracle, "answer_query"):
                        delattr(self.oracle, "answer_query")

            if cancelled_reason is not None:
                # Count the point as processed so the stream can continue / stop loop can exit,
                # but report queried=False and no label so callers do not treat it as a real answer.
                self.num_points_processed += 1
                return {
                    "abs_idx": abs_idx,
                    "queried": False,
                    "label": None,
                    "relevant": None,
                    "cancelled": True,
                    "cancel_reason": cancelled_reason,
                    "num_clusters": len(self.ared.subspace_partition.cluster_dict),
                    "num_known_labels": len(self.ared.subspace_partition.set_of_known_labels),
                }

            # Replay warm-start: if A_RED did not take the query branch, the labeled
            # point never entered l_buf. Force-insert via add_labeled_variant so
            # save/load and squish-merge restore the full labeled multiset.
            # (Does not change live streaming — only meta["replay"] paths.)
            if (
                meta
                and meta.get("replay")
                and obtained_label is not None
                and is_persistable_label(obtained_label)
            ):
                try:
                    buf_count_after = int(self.ared.l_buf.data_circular_buffer.count)
                except Exception:
                    buf_count_after = buf_count_before
                if buf_count_after <= buf_count_before:
                    try:
                        if hasattr(self.ared, "add_labeled_variant"):
                            self.ared.add_labeled_variant(
                                emb, obtained_label, bool(obtained_rel)
                            )
                            vprint(
                                f"[ARED]   -> REPLAY force-insert via add_labeled_variant "
                                f"label='{obtained_label}' (A_RED skipped query branch)"
                            )
                        else:
                            vprint(
                                "[ARED]   -> REPLAY warning: point not in buffer and "
                                "add_labeled_variant unavailable"
                            )
                    except Exception as e:
                        print(
                            f"[AREDAdapter] Replay force-insert failed (non-fatal): {e}"
                        )

            # Back-fill the label we (or ARED) decided for this abs_idx so later peeks are happy
            if obtained_label is not None and is_persistable_label(obtained_label):
                self.oracle.y[abs_idx] = [obtained_label, obtained_rel]  # type: ignore
            elif obtained_label is not None and is_control_label(obtained_label):
                # Never back-fill control sentinels into oracle.y as durable classes
                obtained_label = None
                obtained_rel = None
            else:
                # Non-queried point: ARED assigned it to an existing cluster.
                # Best effort: find the cluster of the most recent point in the buffer.
                try:
                    # The newest point is at internal index -1 in the circular buffer
                    cb = self.ared.l_buf.cluster_key_circular_buffer
                    if cb.count > 0:
                        cluster_key = cb.get(-1)
                        if cluster_key in self.ared.subspace_partition.cluster_dict:
                            cl = self.ared.subspace_partition.cluster_dict[cluster_key]
                            obtained_label = cl.label
                            obtained_rel = cl.relevance
                            if is_persistable_label(obtained_label):
                                self.oracle.y[abs_idx] = [obtained_label, obtained_rel]  # type: ignore
                                self.discovered_labels.add(obtained_label)
                            else:
                                obtained_label = None
                                obtained_rel = None
                except Exception:
                    pass  # best effort only

            self.num_points_processed += 1
            if was_queried:
                self.num_queries += 1
                # Note: per-class query_counts is incremented inside the real-query answer_query path
                # (for both cache hits and provider answers) so we count exactly once per A/RED decision.

            # ------------------------------------------------------------------
            # Data Augmentation (optional, DINO rotation variants)
            # After we have a real label for a queried point, generate rotated
            # versions of the tile image, extract fresh DINO embs, and insert
            # them as labeled variants (same label + relevance).
            # This is done here so it only affects points A/RED actually decided
            # to query.
            # ------------------------------------------------------------------
            if (was_queried and obtained_label and is_persistable_label(obtained_label)
                    and getattr(self.config, "data_augmentation_enabled", False)):
                try:
                    self._apply_data_augmentation(emb, tile_image, obtained_label, obtained_rel)
                except Exception as e:
                    print(f"[AREDAdapter] Data augmentation failed (non-fatal): {e}")

            # Return useful info (expandable)
            info = {
                "abs_idx": abs_idx,
                "queried": was_queried,
                "label": obtained_label,
                "relevant": obtained_rel,
                "num_clusters": len(self.ared.subspace_partition.cluster_dict),
                "num_known_labels": len(self.ared.subspace_partition.set_of_known_labels),
            }
            if not was_queried:
                vprint(f"[ARED] Point abs_idx={abs_idx} did NOT trigger a (real) query. ARED assigned internally (peek satisfied with provisional or cache). label='{obtained_label}'")
            else:
                vprint(f"[ARED] A_RED query to oracle COMPLETE for abs_idx={abs_idx}. Label='{obtained_label}' (relevant={obtained_rel})")
            vprint(f"[ARED] Finished processing point. Total processed: {self.num_points_processed}, queries so far: {self.num_queries}")
            return info

    # ------------------------------------------------------------------
    # Model save / load (replay based - robust & no pickle of ARED internals)
    # ------------------------------------------------------------------
    def export_labeled_points(self) -> List[Dict[str, Any]]:
        """
        Walk the live labeled buffer and return a list of
        ``{emb, label, relevant}`` dicts (oldest → newest).

        Used by save_state and by model-merge strategies. Does not mutate ARED.
        """
        labeled_points: List[Dict[str, Any]] = []
        try:
            lock = getattr(self, "_lock", None)
            ctx = lock if lock is not None else contextlib.nullcontext()
            with ctx:
                l_buf = self.ared.l_buf
                for i in range(l_buf.data_circular_buffer.count):
                    emb = l_buf.data_circular_buffer.get(i)
                    label = l_buf.label_circular_buffer.get(i)
                    rel = l_buf.relevance_circular_buffer.get(i)
                    if emb is not None:
                        labeled_points.append({
                            "emb": np.asarray(emb, dtype=np.float32).copy(),
                            "label": str(label),
                            "relevant": bool(rel),
                        })
        except Exception as e:
            print(f"[AREDAdapter] Warning during export_labeled_points buffer walk: {e}")
        return labeled_points

    def to_state(self) -> AREDState:
        """Build an in-memory ``AREDState`` snapshot of the current adapter."""
        labeled_points = self.export_labeled_points()
        emb_dim: Optional[int] = None
        if labeled_points:
            try:
                emb_dim = int(np.asarray(labeled_points[0]["emb"]).reshape(-1).shape[0])
            except Exception:
                emb_dim = None
        sf = getattr(self.config, "smart_forgetting_var", None)
        return AREDState(
            kappa=float(self.ared.kappa),
            qs_var=int(self.ared.QS_VAR),
            k_comp_pts=int(self.ared.K_COMP_PTS),
            l_buf_size=int(self.ared.l_buf.buffer_size),
            labeled_points=labeled_points,
            emb_dim=emb_dim,
            smart_forgetting_var=tuple(sf) if sf is not None else None,
            merge_meta=None,
        )

    def save_state(self, path: str | Path) -> None:
        """
        Capture enough state to recreate an equivalent ARED on similar data.
        We export the *labeled* points that are currently in the live buffer.
        """
        state = self.to_state()
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(
            f"[AREDAdapter] Saved ARED state with {len(state.labeled_points)} "
            f"labeled points -> {path}"
        )

    def apply_runtime_hyperparams(self, config: Optional["AREDConfig"] = None) -> Dict[str, Any]:
        """
        Push live-tunable hyperparameters from ``config`` onto the running ARED
        instance **without** wiping the labeled buffer / clusters.

        This is required after Load/Merge model: the pickle stores the kappa used
        when the model was trained, but the GUI kappa (and a few other knobs)
        must control *this* run's query decisions.

        Safe to change on a warm-started model:
          - kappa (paranoia / query rate)
          - QS_VAR (comparison distance style for future updates)
          - K_COMP_PTS (k for nearest labeled neighbors)
          - NGHBHOOD_MERGE / SINGLETON_MERGE flags
          - smart_forgetting_var, verbose_flags, data_augmentation_enabled

        Not resized here (would require rebuild): l_buf_size.

        Returns a small dict of before→after values for logging.
        """
        from .config import AREDConfig  # local import

        cfg = config if config is not None else self.config
        if cfg is None:
            return {}

        changes: Dict[str, Any] = {}
        ared = self.ared

        def _set(attr_ared: str, attr_cfg: str, cast=lambda x: x):
            if not hasattr(cfg, attr_cfg):
                return
            old = getattr(ared, attr_ared, None)
            new = cast(getattr(cfg, attr_cfg))
            if old != new:
                setattr(ared, attr_ared, new)
                changes[attr_ared] = {"from": old, "to": new}
            # Keep adapter.config in sync
            try:
                setattr(self.config, attr_cfg, new)
            except Exception:
                pass

        _set("kappa", "kappa", float)
        _set("QS_VAR", "qs_var", int)
        _set("K_COMP_PTS", "k_comp_pts", int)
        _set("NGHBHOOD_MERGE", "nghbhood_merge", bool)
        _set("SINGLETON_MERGE", "singleton_merge", bool)

        if hasattr(cfg, "smart_forgetting_var"):
            old = getattr(ared, "SMART_FORGETTING_VAR", None)
            new = tuple(cfg.smart_forgetting_var)
            if old != new:
                ared.SMART_FORGETTING_VAR = new
                changes["SMART_FORGETTING_VAR"] = {"from": old, "to": new}
            try:
                self.config.smart_forgetting_var = new
            except Exception:
                pass

        if hasattr(cfg, "verbose_flags"):
            flags = list(cfg.verbose_flags or [])
            ared.verbose_flags = flags
            try:
                self.config.verbose_flags = flags
            except Exception:
                pass

        for attr in ("data_augmentation_enabled", "augmentation_rotations"):
            if hasattr(cfg, attr):
                try:
                    setattr(self.config, attr, getattr(cfg, attr))
                except Exception:
                    pass

        # Remember what was in the pickle vs what we are running with
        if not hasattr(self, "_loaded_model_kappa"):
            self._loaded_model_kappa = None
        if changes.get("kappa"):
            # If we never recorded the pre-apply value as "loaded", stash it
            if self._loaded_model_kappa is None:
                self._loaded_model_kappa = changes["kappa"]["from"]

        if changes:
            print(f"[AREDAdapter] Applied runtime hyperparams (buffer preserved): {changes}")
        return changes

    def rebuild_from_state(
        self,
        state: AREDState,
        *,
        prefer_current_kappa: bool = False,
    ) -> None:
        """
        Create a fresh ARED and replay previously labeled points from ``state``.

        Preserves label store / provider / feature extractor attachments across
        the re-init. Query counts reset (warm-start is for clustering, not
        "this run" metrics).

        Hyperparams for the rebuild:
          - Buffer size / k_comp / qs from the **saved** model (structure fidelity).
          - kappa: from saved model by default; if ``prefer_current_kappa`` use the
            adapter's current config kappa (GUI). Regardless, Start will call
            ``apply_runtime_hyperparams`` so the live GUI kappa always wins for
            query decisions on the next run.
        """
        # Preserve attachments across re-init
        old_store = getattr(self, "_label_store", None)
        old_provider = getattr(self, "_label_provider", None)
        old_fe = getattr(self, "_feature_extractor", None)

        # Prefer saved smart-forgetting when present; else keep current config.
        sf = getattr(state, "smart_forgetting_var", None)
        if sf is None:
            sf = getattr(self.config, "smart_forgetting_var", (3, 0.01))

        # kappa for construction: optional GUI preference; Start still re-applies live.
        kappa_for_build = float(state.kappa)
        if prefer_current_kappa:
            try:
                kappa_for_build = float(self.config.kappa)
            except Exception:
                pass

        new_cfg = type(self.config)(  # type: ignore
            kappa=kappa_for_build,
            l_buf_size=state.l_buf_size,
            k_comp_pts=state.k_comp_pts,
            qs_var=state.qs_var,
            data_aug_var=getattr(self.config, "data_aug_var", (0, (0, 0))),
            nghbhood_merge=getattr(self.config, "nghbhood_merge", True),
            singleton_merge=getattr(self.config, "singleton_merge", True),
            smart_forgetting_var=sf,
            verbose_flags=getattr(self.config, "verbose_flags", [0]),
            data_augmentation_enabled=getattr(self.config, "data_augmentation_enabled", False),
            augmentation_rotations=getattr(self.config, "augmentation_rotations", [90, 180, 270]),
        )

        # Fresh instance (re-runs construction + open-world patches)
        self.__init__(new_cfg)

        # Record kappa that came from the pickle (for metrics provenance)
        self._loaded_model_kappa = float(state.kappa)

        if old_store:
            self.set_label_store(old_store)
        if old_provider:
            self.set_label_provider(old_provider)
        if old_fe:
            self.set_feature_extractor(old_fe)

        points = list(getattr(state, "labeled_points", None) or [])
        print(
            f"[AREDAdapter] Replaying {len(points)} points from state "
            f"(saved kappa={state.kappa}, build kappa={kappa_for_build})..."
        )

        for item in points:
            self.process(
                item["emb"],
                tile_image=None,
                meta={
                    "replay": True,
                    "label": item["label"],
                    "relevant": item["relevant"],
                },
            )

        print(
            f"[AREDAdapter] Rebuilt ARED from state. "
            f"Current known labels: {self.get_known_labels()}. "
            f"Note: GUI kappa is applied at Start via apply_runtime_hyperparams()."
        )

    def load_state(
        self,
        path: str | Path,
        label_lookup: Optional[Callable[[np.ndarray], Tuple[str, bool]]] = None,
        *,
        prefer_current_kappa: bool = False,
    ) -> None:
        """
        Load a pickled ``AREDState`` and rebuild via replay.

        Restores labeled buffer / clusters from the file. Decision hyperparameters
        such as kappa are overridden from the GUI when you press Start
        (``apply_runtime_hyperparams``).

        ``label_lookup`` is accepted for API compatibility with older callers;
        replay uses ``meta["replay"]`` and does not require the lookup.
        """
        del label_lookup  # unused; kept for backward-compatible signature
        with open(path, "rb") as f:
            state: AREDState = pickle.load(f)
        self.rebuild_from_state(state, prefer_current_kappa=prefer_current_kappa)
        print(
            f"[AREDAdapter] Loaded & replayed ARED state from {path} "
            f"(pickle kappa={getattr(state, 'kappa', '?')})."
        )

    # ------------------------------------------------------------------
    # Model merge (delegates to model_merge; keeps adapter as the façade)
    # ------------------------------------------------------------------
    def merge_with_state(
        self,
        other: "AREDState",
        strategy: str = "squish",
        **opts: Any,
    ):
        """
        Merge this adapter's current model (as base A) with another ``AREDState``.

        Returns a ``MergeResult`` (see ``drone_ared.model_merge``). Does not
        mutate this adapter unless the caller replaces it with ``result.adapter``.
        """
        from .model_merge import AREDModelMerger

        base = self.to_state()
        return AREDModelMerger().merge(base, other, strategy=strategy, **opts)

    def merge_with_file(
        self,
        path: str | Path,
        strategy: str = "squish",
        **opts: Any,
    ):
        """Merge this adapter (base A) with a saved model pickle (B)."""
        from .model_merge import load_ared_state

        other = load_ared_state(path)
        return self.merge_with_state(other, strategy=strategy, **opts)

    # ------------------------------------------------------------------
    # Convenience for GUI / inspection
    # ------------------------------------------------------------------
    def get_current_clusters_summary(self) -> List[Dict[str, Any]]:
        """Return a list of cluster info for display."""
        summary = []
        for key, cl in self.ared.subspace_partition.cluster_dict.items():
            summary.append({
                "cluster_id": key,
                "label": cl.label,
                "relevant": cl.relevance,
                "n_points": len(cl.l_pt_idxs),
                "comp_distance": cl.comp_distance,
            })
        return summary

    def get_known_labels(self) -> List[str]:
        return sorted(self.ared.subspace_partition.set_of_known_labels)

    def get_model_label_inventory(self) -> Dict[str, Any]:
        """
        Full inventory of labels the live model actually "knows".

        Union of:
          - subspace_partition.set_of_known_labels (A_RED's official set)
          - labels currently sitting in the labeled buffer
          - labels on live clusters

        Used for warm-start metrics (skip first-occurrence for these classes)
        and for merge diagnostics — buffer and set_of_known can briefly diverge
        after some replay paths, so the union is the safe source of truth.
        """
        known: set = set()
        try:
            known |= set(self.ared.subspace_partition.set_of_known_labels or set())
        except Exception:
            pass
        try:
            for cl in self.ared.subspace_partition.cluster_dict.values():
                if getattr(cl, "label", None) is not None:
                    known.add(str(cl.label))
        except Exception:
            pass
        buffer_counts: Dict[str, int] = {}
        try:
            for p in self.export_labeled_points():
                lab = str(p.get("label", "")).strip()
                if not lab or not is_persistable_label(lab):
                    continue
                known.add(lab)
                buffer_counts[lab] = buffer_counts.get(lab, 0) + 1
        except Exception:
            pass
        # Drop control sentinels if any slipped in
        known = {k for k in known if is_persistable_label(str(k))}
        return {
            "labels": sorted(known, key=lambda s: str(s).casefold()),
            "buffer_counts": dict(sorted(buffer_counts.items(), key=lambda kv: kv[0].casefold())),
            "n_buffer_points": sum(buffer_counts.values()),
            "n_labels": len(known),
        }

    def get_query_counts(self) -> Dict[str, int]:
        """Return per-class counts of A/RED queries (real decisions) this run.

        Incremented for every point where A/RED decided it needed a label
        (whether the answer came from cache, exact DB, or human GUI).
        This is the number the user wants to see in the class boxes:
        "how many of each class have been queried by A/RED throughout the run".
        """
        return dict(getattr(self, 'query_counts', {}))

    @property
    def num_clusters(self) -> int:
        try:
            return len(self.ared.subspace_partition.cluster_dict)
        except Exception:
            return 0

    @property
    def num_known_labels(self) -> int:
        try:
            return len(self.ared.subspace_partition.set_of_known_labels)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Data Augmentation support (DINO rotations on labeled queried points)
    # ------------------------------------------------------------------
    def _apply_data_augmentation(self, original_emb, tile_image, label: str, relevant: bool):
        """Generate rotated tile images, extract DINO embeddings, and insert
        them as labeled variants using the (possibly extended) ARED API.
        Only called for points that caused a real query.
        """
        if tile_image is None:
            return

        # tile_image may be a Tile dataclass or a raw PIL Image
        pil_img = getattr(tile_image, "image", tile_image) if tile_image is not None else None
        if pil_img is None:
            return

        if not getattr(self, "_dino_augmenter", None) and getattr(self, "_feature_extractor", None):
            try:
                from .augmentation import DINOAugmenter
                self._dino_augmenter = DINOAugmenter(self._feature_extractor)
            except Exception:
                return

        augmenter = getattr(self, "_dino_augmenter", None)
        if augmenter is None:
            return

        angles = getattr(self.config, "augmentation_rotations", [90, 180, 270])
        if not angles:
            return

        try:
            rot_embs = augmenter.get_rotated_embeddings(pil_img, angles)
        except Exception as e:
            print(f"[AREDAdapter] Rotation embedding generation failed: {e}")
            return

        if not rot_embs:
            return

        added = 0
        for remb in rot_embs:
            try:
                # Prefer the clean extension if present
                if hasattr(self.ared, "add_labeled_variant"):
                    self.ared.add_labeled_variant(remb, label, relevant)
                else:
                    # Fallback: use forced-label replay-style path
                    self.process(
                        remb,
                        tile_image=None,
                        meta={
                            "augmented": True,
                            "label": label,
                            "relevant": relevant,
                            "replay": True,
                        },
                    )
                added += 1
            except Exception as e:
                print(f"[AREDAdapter] Failed to insert augmented variant: {e}")

        if added > 0:
            print(f"[AREDAdapter] Inserted {added} rotation-augmented embeddings for label '{label}'")
