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

if TYPE_CHECKING:
    from .label_store import PersistentLabelStore
    from .config import AREDConfig

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
    """Serializable snapshot for 'Save ARED Model' feature."""
    kappa: float
    qs_var: int
    k_comp_pts: int
    l_buf_size: int
    labeled_points: List[Dict[str, Any]] = field(default_factory=list)  # each: {emb, label, relevant}
    # We store only what is needed to replay. Cluster keys are re-created on replay.


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

    # Internal reference for type checkers / later wiring
    _label_store: Optional["PersistentLabelStore"] = None

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

        print(f"[ARED] received tile for process (meta={meta})")
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
                    print(f"[ARED]   -> REPLAY using saved label '{label}' (relevant={rel})")
                    return label, rel

                is_peek = (call_num == 1 and self.num_points_processed > 0)

                # 1. Try persistent cache first -- always, for both peek and real queries
                if getattr(self, "_label_store", None) is not None:
                    cached = self._label_store.lookup(emb)
                    if cached is not None:
                        label, rel = cached
                        obtained_label = label
                        obtained_rel = rel
                        self.oracle.y[_abs_index] = [label, rel]  # type: ignore
                        self.discovered_labels.add(label)
                        if is_peek:
                            print(f"[ARED]   -> Cache HIT on peek for abs_idx={_abs_index}. Auto (no GUI).")
                        else:
                            print(f"[ARED]   -> Cache HIT for this query. Auto-labeled as '{label}' (relevant={rel}). No GUI.")
                        return label, rel
                    else:
                        if is_peek:
                            print(f"[ARED]   -> Cache MISS on peek (will use provisional, no GUI).")
                        else:
                            print(f"[ARED]   -> Cache MISS. Will request label from provider (GUI or fallback).")

                if is_peek:
                    # Peek call (internal ARED accounting for non-queried points).
                    # Do NOT call human/GUI provider here -- that would query on every tile.
                    # Supply a provisional so ARED can continue; real cluster label may be back-filled later.
                    label = meta.get("label", f"__PEEK_{abs_idx % 100}")
                    rel = bool(meta.get("relevant", False))
                    obtained_label = label
                    obtained_rel = rel
                    self.oracle.y[_abs_index] = [label, rel]  # type: ignore
                    print(f"[ARED] Peek answer_query (call#{call_num}) for abs_idx={_abs_index} -- provisional (no oracle query to user).")
                    return label, rel

                # This is a real query path (2nd hook call during process, or first_point).
                was_queried = True
                print(f"[ARED] A_RED decided to QUERY (sending request to oracle shim) for abs_idx={_abs_index} (tile meta: {meta}) (real decision, call#{call_num})")

                # 2. Fall back to the registered provider (normally the GUI) -- only for real queries
                if self._label_provider is None:
                    # Fallback for headless / synthetic tests
                    label = meta.get("label", f"auto_{abs_idx % 5}")
                    rel = bool(meta.get("relevant", False))
                    print(f"[ARED]   -> No provider, using fallback label '{label}' (relevant={rel})")
                else:
                    label, rel = self._label_provider(emb, tile_image, meta)
                    print(f"[ARED]   -> Provider returned label '{label}' (relevant={rel})")

                obtained_label = label
                obtained_rel = rel

                # Record + also feed back into the store so identical future tiles are cached
                self.oracle.y[_abs_index] = [label, rel]  # type: ignore
                self.discovered_labels.add(label)
                if getattr(self, "_label_store", None) is not None:
                    try:
                        self._label_store.add(emb, label, rel)
                        print(f"[ARED]   -> Added to label store for future cache hits.")
                    except Exception as e:
                        print(f"[ARED]   -> WARNING: Failed to add to label store: {e}")
                return label, rel

            # Temporarily install
            self.oracle.answer_query = _interactive_answer  # type: ignore

            try:
                if self.num_points_processed == 0:
                    self.ared.process_first_point(emb)
                else:
                    self.ared.process_point(emb)
            finally:
                # Restore
                if original_answer is not None:
                    self.oracle.answer_query = original_answer
                else:
                    # Remove our monkey if it was the first
                    if hasattr(self.oracle, "answer_query"):
                        delattr(self.oracle, "answer_query")

            # Back-fill the label we (or ARED) decided for this abs_idx so later peeks are happy
            if obtained_label is not None:
                self.oracle.y[abs_idx] = [obtained_label, obtained_rel]  # type: ignore
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
                            self.oracle.y[abs_idx] = [obtained_label, obtained_rel]  # type: ignore
                            self.discovered_labels.add(obtained_label)
                except Exception:
                    pass  # best effort only

            self.num_points_processed += 1
            if was_queried:
                self.num_queries += 1

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
                print(f"[ARED] Point abs_idx={abs_idx} did NOT trigger a (real) query. ARED assigned internally (peek satisfied with provisional or cache). label='{obtained_label}'")
            else:
                print(f"[ARED] A_RED query to oracle COMPLETE for abs_idx={abs_idx}. Label='{obtained_label}' (relevant={obtained_rel})")
            print(f"[ARED] Finished processing point. Total processed: {self.num_points_processed}, queries so far: {self.num_queries}")
            return info

    # ------------------------------------------------------------------
    # Model save / load (replay based - robust & no pickle of ARED internals)
    # ------------------------------------------------------------------
    def save_state(self, path: str | Path) -> None:
        """
        Capture enough state to recreate an equivalent ARED on similar data.
        We export the *labeled* points that are currently in the live buffer.
        """
        labeled_points = []
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
                            "emb": np.asarray(emb, dtype=np.float32),
                            "label": str(label),
                            "relevant": bool(rel),
                        })
        except Exception as e:
            print(f"[AREDAdapter] Warning during save_state buffer walk: {e}")

        state = AREDState(
            kappa=self.ared.kappa,
            qs_var=self.ared.QS_VAR,
            k_comp_pts=self.ared.K_COMP_PTS,
            l_buf_size=self.ared.l_buf.buffer_size,
            labeled_points=labeled_points,
        )
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[AREDAdapter] Saved ARED state with {len(labeled_points)} labeled points -> {path}")

    def load_state(self, path: str | Path, label_lookup: Optional[Callable[[np.ndarray], Tuple[str, bool]]] = None) -> None:
        """
        Create a fresh ARED and replay previously labeled points.
        If a PersistentLabelStore is attached (via set_label_store), it will be
        consulted automatically for every replayed point (and new points later).

        This implements the "A_RED model saving" requirement: warm-start the
        internal clusters without forcing the user to re-label everything.
        """
        with open(path, "rb") as f:
            state: AREDState = pickle.load(f)

        # Preserve attachments across re-init
        old_store = getattr(self, "_label_store", None)
        old_provider = getattr(self, "_label_provider", None)

        # Re-create adapter with (approximately) same hyperparams
        new_cfg = type(self.config)(  # type: ignore
            kappa=state.kappa,
            l_buf_size=state.l_buf_size,
            k_comp_pts=state.k_comp_pts,
            qs_var=state.qs_var,
            data_aug_var=getattr(self.config, "data_aug_var", (0, (0, 0))),
            nghbhood_merge=getattr(self.config, "nghbhood_merge", True),
            singleton_merge=getattr(self.config, "singleton_merge", True),
            smart_forgetting_var=getattr(self.config, "smart_forgetting_var", (3, 0.01)),
            verbose_flags=getattr(self.config, "verbose_flags", [0]),
        )

        # Fresh instance
        self.__init__(new_cfg)  # re-runs construction + patches

        if old_store:
            self.set_label_store(old_store)
        # Do not restore provider yet if we want to avoid it during replay;
        # the "replay" special case in _interactive_answer protects us anyway.
        # Restore after so that subsequent real processing uses it.
        if old_provider:
            self.set_label_provider(old_provider)

        # If we have a label store attached, the process() method will use it
        # automatically. We can still feed the old points so clusters are rebuilt.
        print(f"[AREDAdapter] Replaying {len(state.labeled_points)} points from saved state...")

        for item in state.labeled_points:
            self.process(
                item["emb"],
                tile_image=None,
                meta={"replay": True, "label": item["label"], "relevant": item["relevant"]}
            )

        print(f"[AREDAdapter] Loaded & replayed ARED state from {path}. "
              f"Current known labels: {self.get_known_labels()}")

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
