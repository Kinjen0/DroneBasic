"""
PersistentLabelStore

Stores previously labeled tiles (by their DINO embedding) so that on future runs
(or later in the same run) sufficiently similar tiles can be auto-labeled without
bothering the user / SME.

This directly fulfills the requirement:
    "create a method of 'Saving' the unique tiles labels so that they can be
     queried in future runs without a users input."

Implementation notes:
- Uses sklearn NearestNeighbors (or BallTree) for fast lookup.
- Persists via pickle (simple + sufficient; embeddings are moderate size).
- Distance is L2 to stay consistent with ARED's internal distance calculations.
- Threshold is configurable; conservative defaults are recommended.
- Rebuilds the index periodically (cheap for a few thousand examples).
- Future: easy to swap in FAISS, Annoy, or a proper vector DB.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors


class PersistentLabelStore:
    """
    Thread-safe-enough (for our usage) persistent cache of (embedding -> label, relevant).
    """

    def __init__(
        self,
        db_path: str | Path = "drone_ared_labels.pkl",
        auto_label_threshold: float = 0.15,
        rebuild_interval: int = 32,
        distance_metric: str = "l2",
    ):
        self.db_path = Path(db_path)
        self.auto_label_threshold = float(auto_label_threshold)
        self.rebuild_interval = max(1, int(rebuild_interval))
        self.distance_metric = distance_metric  # "l2" or "cosine" (cosine via normalized data)

        self._embeddings: List[np.ndarray] = []
        self._labels: List[str] = []
        self._relevances: List[bool] = []

        self._nn: Optional[NearestNeighbors] = None
        self._dirty_since_rebuild = 0

        self._load_if_exists()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def lookup(self, embedding: np.ndarray) -> Optional[Tuple[str, bool]]:
        """
        Return (label, relevant) if a sufficiently close stored example is found.
        Otherwise return None (caller should ask the user).
        """
        if not self._embeddings or self._nn is None:
            return None

        emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        # sklearn NearestNeighbors returns squared euclidean for 'euclidean' metric
        dists, idxs = self._nn.kneighbors(emb, n_neighbors=1, return_distance=True)
        dist = float(dists[0, 0])
        # For euclidean the actual distance is sqrt, but threshold is calibrated on raw.
        # We store the raw L2 distance from ARED usage, so compare directly.
        if dist <= self.auto_label_threshold:
            i = int(idxs[0, 0])
            return self._labels[i], self._relevances[i]
        return None

    def add(self, embedding: np.ndarray, label: str, relevant: bool) -> None:
        """Remember this decision."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self._embeddings.append(emb)
        self._labels.append(str(label))
        self._relevances.append(bool(relevant))
        self._dirty_since_rebuild += 1

        if self._dirty_since_rebuild >= self.rebuild_interval or self._nn is None:
            self._rebuild_index()

    def __len__(self) -> int:
        return len(self._embeddings)

    def save(self) -> None:
        """Persist current store to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "embeddings": [e.copy() for e in self._embeddings],
            "labels": list(self._labels),
            "relevances": list(self._relevances),
            "threshold": self.auto_label_threshold,
        }
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[LabelStore] Saved {len(self)} entries to {self.db_path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _rebuild_index(self) -> None:
        if not self._embeddings:
            self._nn = None
            return

        X = np.stack(self._embeddings).astype(np.float32)
        # Use 'euclidean' (L2). If user wants cosine they should normalize upstream.
        metric = "euclidean" if self.distance_metric == "l2" else "cosine"
        self._nn = NearestNeighbors(n_neighbors=1, metric=metric, algorithm="auto")
        self._nn.fit(X)
        self._dirty_since_rebuild = 0

    def _load_if_exists(self) -> None:
        if not self.db_path.exists():
            return
        try:
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            self._embeddings = [np.asarray(e, dtype=np.float32) for e in data.get("embeddings", [])]
            self._labels = list(data.get("labels", []))
            self._relevances = list(data.get("relevances", []))
            if "threshold" in data:
                self.auto_label_threshold = float(data["threshold"])
            self._rebuild_index()
            print(f"[LabelStore] Loaded {len(self)} cached labels from {self.db_path}")
        except Exception as e:
            print(f"[LabelStore] Failed to load {self.db_path}: {e}. Starting empty.")
            self._embeddings = []
            self._labels = []
            self._relevances = []

    # ------------------------------------------------------------------
    # Introspection / GUI helpers
    # ------------------------------------------------------------------
    def get_class_counts(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(self._labels))

    def get_all_labels(self) -> List[str]:
        return sorted(set(self._labels))

    def get_class_relevance(self, label: str) -> Optional[bool]:
        """Return the relevant flag associated with a previously stored example of this class, if any."""
        for i, lbl in enumerate(self._labels):
            if lbl == label:
                return self._relevances[i]
        return None
