"""
Unit tests for A_RED model merging (headless, no GUI / DINO / video).

Uses tiny synthetic embeddings and the adapter replay / donor paths only.
"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from drone_ared.ared_adapter import AREDAdapter, AREDState
from drone_ared.config import AREDConfig
from drone_ared.model_merge import (
    AREDModelMerger,
    IngestMergeStrategy,
    InterleaveMergeStrategy,
    SquishMergeStrategy,
    load_ared_state,
    validate_merge_compatible,
)


def _cfg(buf: int = 64, kappa: float = 1.0) -> AREDConfig:
    """Quiet, small-buffer config for fast tests."""
    return AREDConfig(
        kappa=kappa,
        l_buf_size=buf,
        k_comp_pts=3,
        qs_var=1,
        nghbhood_merge=True,
        singleton_merge=True,
        smart_forgetting_var=(3, 0.01),
        verbose_flags=[],
        data_augmentation_enabled=False,
    )


def _pt(emb, label: str, relevant: bool = False) -> dict:
    return {
        "emb": np.asarray(emb, dtype=np.float32).reshape(-1),
        "label": label,
        "relevant": relevant,
    }


def _make_state(
    points: list,
    *,
    kappa: float = 1.0,
    l_buf_size: int = 64,
    qs_var: int = 1,
    k_comp_pts: int = 3,
) -> AREDState:
    emb_dim = int(np.asarray(points[0]["emb"]).reshape(-1).shape[0]) if points else None
    return AREDState(
        kappa=kappa,
        qs_var=qs_var,
        k_comp_pts=k_comp_pts,
        l_buf_size=l_buf_size,
        labeled_points=list(points),
        emb_dim=emb_dim,
        smart_forgetting_var=(3, 0.01),
    )


def _cluster_a_points(n: int = 5, dim: int = 8, center: float = 0.0, label: str = "class_a"):
    """Tight cluster around a constant vector."""
    rng = np.random.default_rng(0)
    pts = []
    for _ in range(n):
        emb = np.full(dim, center, dtype=np.float32) + rng.normal(0, 0.01, size=dim).astype(np.float32)
        pts.append(_pt(emb, label, relevant=False))
    return pts


def _cluster_b_points(n: int = 5, dim: int = 8, center: float = 10.0, label: str = "class_b"):
    rng = np.random.default_rng(1)
    pts = []
    for _ in range(n):
        emb = np.full(dim, center, dtype=np.float32) + rng.normal(0, 0.01, size=dim).astype(np.float32)
        pts.append(_pt(emb, label, relevant=True))
    return pts


class TestExportRoundtrip(unittest.TestCase):
    def test_export_roundtrip(self):
        adapter = AREDAdapter(_cfg(buf=32))
        pts = _cluster_a_points(4) + _cluster_b_points(3)
        for p in pts:
            adapter.process(
                p["emb"],
                meta={"replay": True, "label": p["label"], "relevant": p["relevant"]},
            )

        state = adapter.to_state()
        self.assertEqual(len(state.labeled_points), len(pts))
        self.assertEqual(state.emb_dim, 8)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.pkl"
            adapter.save_state(path)

            adapter2 = AREDAdapter(_cfg(buf=32))
            adapter2.load_state(path)
            exported = adapter2.export_labeled_points()
            self.assertEqual(len(exported), len(pts))
            labels = {e["label"] for e in exported}
            self.assertIn("class_a", labels)
            self.assertIn("class_b", labels)


class TestSquish(unittest.TestCase):
    def setUp(self):
        self.state_a = _make_state(_cluster_a_points(5), l_buf_size=32)
        self.state_b = _make_state(_cluster_b_points(5), l_buf_size=32)

    def test_squish_contains_all_labels(self):
        result = SquishMergeStrategy().merge(self.state_a, self.state_b)
        labels = set(result.adapter.get_known_labels())
        self.assertIn("class_a", labels)
        self.assertIn("class_b", labels)
        # All points forced in → buffer should hold the union
        self.assertEqual(result.points_accepted, 10)
        self.assertEqual(result.points_from_a, 5)
        self.assertEqual(result.points_from_b, 5)

    def test_squish_buffer_doubled(self):
        result = SquishMergeStrategy().merge(self.state_a, self.state_b, buffer_multiplier=2)
        buf = result.summary["l_buf_size"]
        # max(32,32)*2 = 64, and union needs 10 → 64
        self.assertGreaterEqual(buf, 64)
        self.assertGreaterEqual(buf, result.points_from_a + result.points_from_b)

    def test_squish_via_merger_facade(self):
        result = AREDModelMerger().merge(self.state_a, self.state_b, strategy="squish")
        self.assertEqual(result.strategy_name, "squish")
        self.assertGreater(result.adapter.num_clusters, 0)


class TestIngest(unittest.TestCase):
    def test_ingest_base_preserved_then_grows(self):
        state_a = _make_state(_cluster_a_points(6), l_buf_size=64)
        # Far cluster → should trigger queries when streamed into A
        state_b = _make_state(_cluster_b_points(4, center=50.0), l_buf_size=64)

        result = IngestMergeStrategy().merge(state_a, state_b)
        self.assertEqual(result.strategy_name, "ingest")
        self.assertGreaterEqual(result.adapter.num_points_processed, 6 + 4)
        # Distant B points should cause at least one query
        self.assertGreaterEqual(result.queries_during_merge, 1)
        labels = set(result.adapter.get_known_labels())
        self.assertIn("class_a", labels)
        self.assertIn("class_b", labels)

    def test_ingest_near_duplicate_may_skip_query(self):
        """Near-identical B points to A's cluster should often not query."""
        pts_a = _cluster_a_points(8, center=0.0)
        # Same neighborhood as A
        pts_b = _cluster_a_points(3, center=0.0, label="class_a")
        state_a = _make_state(pts_a, l_buf_size=64)
        state_b = _make_state(pts_b, l_buf_size=64)

        result = IngestMergeStrategy().merge(state_a, state_b)
        # Not a hard guarantee of zero queries (κ / first-neighbor edge cases),
        # but accepted buffer should not explode beyond capacity and A remains.
        self.assertIn("class_a", result.adapter.get_known_labels())
        self.assertLessEqual(result.points_accepted, 64)
        # Bookkeeping fields present
        self.assertIn("b_points_queried", result.summary)
        self.assertIn("b_points_not_queried", result.summary)


class TestInterleave(unittest.TestCase):
    def test_interleave_builds_fresh_model(self):
        state_a = _make_state(_cluster_a_points(4), l_buf_size=64)
        state_b = _make_state(_cluster_b_points(4, center=20.0), l_buf_size=64)

        result = InterleaveMergeStrategy().merge(state_a, state_b, start_with="a")
        self.assertEqual(result.strategy_name, "interleave")
        self.assertGreater(result.adapter.num_points_processed, 0)
        # First point always queries → at least one query
        self.assertGreaterEqual(result.queries_during_merge, 1)
        labels = set(result.adapter.get_known_labels())
        # With far clusters, both labels should typically appear
        self.assertTrue(labels)  # non-empty

    def test_interleave_start_with_b(self):
        state_a = _make_state(_cluster_a_points(3), l_buf_size=32)
        state_b = _make_state(_cluster_b_points(3, center=15.0), l_buf_size=32)
        result = InterleaveMergeStrategy().merge(state_a, state_b, start_with="b")
        self.assertEqual(result.summary.get("start_with"), "b")


class TestValidation(unittest.TestCase):
    def test_dim_mismatch_raises(self):
        a = _make_state([_pt(np.zeros(4), "a")], l_buf_size=16)
        b = _make_state([_pt(np.zeros(8), "b")], l_buf_size=16)
        errors, _ = validate_merge_compatible(a, b)
        self.assertTrue(any("dimension" in e.lower() for e in errors))
        with self.assertRaises(ValueError):
            SquishMergeStrategy().merge(a, b)

    def test_empty_model_raises(self):
        a = _make_state(_cluster_a_points(2), l_buf_size=16)
        b = _make_state([], l_buf_size=16)
        with self.assertRaises(ValueError):
            SquishMergeStrategy().merge(a, b)

    def test_hyperparam_mismatch_warns(self):
        a = _make_state(_cluster_a_points(2), kappa=1.0, l_buf_size=16)
        b = _make_state(_cluster_b_points(2), kappa=0.5, l_buf_size=16)
        errors, warnings = validate_merge_compatible(a, b)
        self.assertEqual(errors, [])
        self.assertTrue(any("kappa" in w for w in warnings))


class TestOldPickle(unittest.TestCase):
    def test_old_pickle_loads(self):
        """Minimal AREDState without optional fields still loads and rebuilds."""
        # Build a classic-style state (only the original fields).
        pts = _cluster_a_points(3)
        # Simulate an older pickle by constructing and pickling only core fields
        # via a simple namespace-like object is fragile; instead pickle a real
        # AREDState and ensure missing optional attrs are tolerated on rebuild.
        state = AREDState(
            kappa=1.0,
            qs_var=1,
            k_comp_pts=3,
            l_buf_size=32,
            labeled_points=pts,
        )
        # Explicitly no emb_dim / smart_forgetting_var / merge_meta
        self.assertIsNone(state.emb_dim)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old.pkl"
            with open(path, "wb") as f:
                pickle.dump(state, f)
            loaded = load_ared_state(path)
            adapter = AREDAdapter(_cfg(buf=32))
            adapter.rebuild_from_state(loaded)
            self.assertGreaterEqual(len(adapter.export_labeled_points()), 1)
            self.assertIn("class_a", adapter.get_known_labels())


class TestAdapterMergeDelegates(unittest.TestCase):
    def test_merge_with_state(self):
        adapter = AREDAdapter(_cfg(buf=32))
        for p in _cluster_a_points(4):
            adapter.process(
                p["emb"],
                meta={"replay": True, "label": p["label"], "relevant": p["relevant"]},
            )
        other = _make_state(_cluster_b_points(3, center=12.0), l_buf_size=32)
        result = adapter.merge_with_state(other, strategy="squish")
        self.assertEqual(result.strategy_name, "squish")
        self.assertIn("class_b", result.adapter.get_known_labels())


if __name__ == "__main__":
    unittest.main()
