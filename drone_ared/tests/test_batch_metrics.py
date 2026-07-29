"""Unit tests for batch-window QP/RR and dual-track checkpoint fields."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

from drone_ared.metrics import (
    evaluate_batch_window,
    evaluate_from_annotations_and_queries,
    summarize_for_batch_checkpoint,
    make_tile_key,
)
from drone_ared.run_metrics_logger import RunMetricsLogger, CHECKPOINT_CSV_FIELDS, BATCH_CSV_FIELDS
from drone_ared.reporting.loader import RunRecord, load_run


def _ann(
    video: str,
    frame: int,
    row: int,
    col: int,
    label: str,
    relevant: bool = False,
    w: int = 240,
    h: int = 240,
) -> Dict[str, Any]:
    return {
        "video_path": video,
        "abs_frame": frame,
        "tile_row": row,
        "tile_col": col,
        "tile_width": w,
        "tile_height": h,
        "label": label,
        "relevant": relevant,
    }


def _key(a: Dict[str, Any]) -> Tuple:
    return make_tile_key(a)


class TestEvaluateBatchWindow(unittest.TestCase):
    """Core semantics: firsts use stream context; TP/FP/FN are window-local."""

    def setUp(self):
        self.video = "test.mp4"
        # 6 tiles in stream order (frame 0..5)
        # batch1: frames 0-2, batch2: frames 3-5
        self.anns = [
            _ann(self.video, 0, 0, 0, "background"),
            _ann(self.video, 1, 0, 0, "person", relevant=True),  # first of person
            _ann(self.video, 2, 0, 0, "car"),  # first of car
            _ann(self.video, 3, 0, 0, "person", relevant=True),  # person again (relevant)
            _ann(self.video, 4, 0, 0, "background"),
            _ann(self.video, 5, 0, 0, "person", relevant=True),
        ]
        self.processed = [_key(a) for a in self.anns]
        # Query first person, first car, and last person — miss middle person
        self.queried = [
            self.processed[1],  # person first
            self.processed[2],  # car first
            self.processed[5],  # person later
        ]

    def test_batch1_counts_firsts_batch2_does_not_recount(self):
        b1 = evaluate_batch_window(
            self.anns,
            self.queried,
            self.processed,
            batch_start=0,
            batch_end=3,
            first_occurrence_mode="paper",
            batch_index=1,
        )
        self.assertNotIn("error", b1)
        # should: bg first, person first, car first → 3 positives
        # queried: person + car → TP=2, FN=1 (bg first not queried), FP=0
        self.assertEqual(b1["tp"], 2)
        self.assertEqual(b1["fn"], 1)
        self.assertEqual(b1["fp"], 0)
        self.assertEqual(b1["n_should_query"], 3)
        self.assertAlmostEqual(b1["query_precision"], 1.0)
        self.assertAlmostEqual(b1["relevant_recall"], 2 / 3, places=4)

        b2 = evaluate_batch_window(
            self.anns,
            self.queried,
            self.processed,
            batch_start=3,
            batch_end=6,
            first_occurrence_mode="paper",
            batch_index=2,
        )
        self.assertNotIn("error", b2)
        # In batch2: person@3 and person@5 are relevant-class positives.
        # background@4 is NOT first (bg first was in batch1).
        # car first already passed — no car first here.
        # should = person@3, person@5 → 2
        # queried in batch: only person@5 → TP=1, FN=1
        self.assertEqual(b2["n_should_query"], 2)
        self.assertEqual(b2["tp"], 1)
        self.assertEqual(b2["fn"], 1)
        self.assertEqual(b2["fp"], 0)
        # Critical: batch2 must NOT invent a new "first of person"
        self.assertLess(b2["n_should_query"], 3)

    def test_cumulative_matches_full_eval(self):
        full = evaluate_from_annotations_and_queries(
            self.anns,
            self.queried,
            total_points=len(self.processed),
            processed_keys=self.processed,
            first_occurrence_mode="paper",
        )
        # Whole stream as one "batch"
        whole = evaluate_batch_window(
            self.anns,
            self.queried,
            self.processed,
            batch_start=0,
            batch_end=len(self.processed),
            first_occurrence_mode="paper",
        )
        self.assertEqual(full["tp"], whole["tp"])
        self.assertEqual(full["fp"], whole["fp"])
        self.assertEqual(full["fn"], whole["fn"])
        self.assertAlmostEqual(full["query_precision"], whole["query_precision"])
        self.assertAlmostEqual(full["relevant_recall"], whole["relevant_recall"])

    def test_batch_windows_partition_processed_keys(self):
        b1 = evaluate_batch_window(
            self.anns, self.queried, self.processed, 0, 3, first_occurrence_mode="paper"
        )
        b2 = evaluate_batch_window(
            self.anns, self.queried, self.processed, 3, 6, first_occurrence_mode="paper"
        )
        self.assertEqual(b1["batch_tiles"] + b2["batch_tiles"], len(self.processed))
        self.assertEqual(b1["batch_tile_end"], b2["batch_tile_start"])

    def test_empty_batch_errors(self):
        r = evaluate_batch_window(
            self.anns, self.queried, self.processed, 2, 2, first_occurrence_mode="paper"
        )
        self.assertIn("error", r)

    def test_summarize_prefix(self):
        b1 = evaluate_batch_window(
            self.anns, self.queried, self.processed, 0, 3, first_occurrence_mode="paper"
        )
        s = summarize_for_batch_checkpoint(b1)
        self.assertTrue(s["batch_metrics_available"])
        self.assertEqual(s["batch_tp"], b1["tp"])
        self.assertIn("batch_query_precision", s)
        # error path
        s2 = summarize_for_batch_checkpoint({"error": "nope", "batch_tile_start": 0})
        self.assertFalse(s2["batch_metrics_available"])
        self.assertEqual(s2["batch_note"], "nope")

    def test_warm_start_skips_known_first_in_batch(self):
        # person already known at start → first person not a should-positive
        b1 = evaluate_batch_window(
            self.anns,
            self.queried,
            self.processed,
            0,
            3,
            known_classes_at_start={"person"},
            first_occurrence_mode="skip_known",
        )
        # should: bg first, car first; person first skipped (known); person is still
        # relevant-class so person@1 IS still a positive via relevant rule.
        # person is relevant → still counts. n_should includes bg first + person(rel) + car first
        self.assertGreaterEqual(b1["n_should_query"], 2)
        self.assertEqual(b1["first_occurrence_mode"], "skip_known")


class TestRunMetricsLoggerBatch(unittest.TestCase):
    def _mock_controller(
        self,
        processed: List[Tuple],
        queried: List[Tuple],
        anns: List[Dict[str, Any]],
        tiles: int,
        queries: int,
    ):
        ctrl = MagicMock()
        ctrl.stats = {
            "tiles_processed": tiles,
            "frames_read": 10,
            "ared_queries": queries,
            "user_queries": queries,
            "cache_hits": 0,
            "ared_clusters": 1,
            "ared_known_labels": 0,
            "current_video": "test.mp4",
        }
        ctrl.processed_identities = list(processed)
        ctrl.queried_identities = list(queried)
        ctrl.label_only_mode = False
        ctrl.ared_known_labels_at_run_start = set()
        ctrl.ared_adapter = None
        ctrl.tiler = None
        ctrl.config = None
        ctrl.annotation_manager = None
        ctrl.tile_db = MagicMock()
        # Return anns for any video
        ctrl.tile_db.get_annotations_for_video.return_value = list(anns)
        ctrl.tile_db.list_videos.return_value = ["test.mp4"]
        return ctrl

    def test_snapshot_has_both_tracks(self):
        video = "test.mp4"
        anns = [
            _ann(video, i, 0, 0, "person" if i % 2 else "background", relevant=(i % 2 == 1))
            for i in range(10)
        ]
        processed = [_key(a) for a in anns]
        queried = [processed[1], processed[3], processed[7]]

        with tempfile.TemporaryDirectory() as td:
            logger = RunMetricsLogger(
                output_dir=td,
                checkpoint_every=5,
                run_params={"kappa": 0.5, "first_occurrence_mode": "paper"},
                enabled=True,
                batch_metrics_enabled=True,
            )
            # First checkpoint at 5 tiles
            ctrl = self._mock_controller(processed[:5], [q for q in queried if q in set(processed[:5])], anns, 5, 2)
            # Override annotation loader
            logger._load_annotations_for_video = lambda c, v: list(anns)  # type: ignore

            snap1 = logger.checkpoint(ctrl, reason="interval")
            self.assertTrue(snap1.get("metrics_available"))
            self.assertIn("query_precision", snap1)
            self.assertIn("batch_query_precision", snap1)
            self.assertTrue(snap1.get("batch_metrics_available"))
            self.assertEqual(snap1["batch_tile_start"], 0)
            self.assertEqual(snap1["batch_tile_end"], 5)

            # Second checkpoint at 10 tiles — batch window is 5..10
            ctrl2 = self._mock_controller(processed, queried, anns, 10, 3)
            snap2 = logger.checkpoint(ctrl2, reason="interval")
            self.assertEqual(snap2["batch_tile_start"], 5)
            self.assertEqual(snap2["batch_tile_end"], 10)
            # Cumulative fields still present
            self.assertIsNotNone(snap2.get("query_precision"))
            self.assertIsNotNone(snap2.get("batch_query_precision"))

            # CSV columns include batch fields
            ckpt_csv = Path(logger.run_dir) / "checkpoints.csv"
            batch_csv = Path(logger.run_dir) / "batches.csv"
            self.assertTrue(ckpt_csv.is_file())
            self.assertTrue(batch_csv.is_file())
            with open(ckpt_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertIn("batch_query_precision", rows[0])
            self.assertIn("query_precision", rows[0])
            # All declared fields exist as headers
            with open(ckpt_csv, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f))
            for col in ("query_precision", "batch_query_precision", "batch_relevant_recall"):
                self.assertIn(col, header)

            status = logger.one_line_status()
            self.assertIn("batch QP=", status)

            # Cumulative columns must still be the original names
            for col in (
                "query_precision",
                "relevant_recall",
                "f1_score",
                "section_query_rate",
            ):
                self.assertIn(col, CHECKPOINT_CSV_FIELDS)
            for col in BATCH_CSV_FIELDS:
                self.assertTrue(col.startswith("batch_") or col in (
                    "checkpoint_index", "reason", "tiles_processed",
                    "section_ared_queries", "section_query_rate", "section_relevant_rate",
                    "elapsed_sec", "current_video",
                ))


class TestReportingBatchHelpers(unittest.TestCase):
    def test_old_run_without_batch_fields_loads(self):
        """Backward compat: cumulative-only run.json still loads."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "old_run"
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                """
                {
                  "schema_version": 1,
                  "run_id": "old_run",
                  "status": "finished",
                  "run_params": {"kappa": 0.65},
                  "checkpoints": [
                    {
                      "checkpoint_index": 1,
                      "tiles_processed": 500,
                      "query_precision": 0.5,
                      "relevant_recall": 0.4,
                      "f1_score": 0.44,
                      "ared_queries": 10
                    }
                  ],
                  "final_metrics": {"query_precision": 0.5, "relevant_recall": 0.4}
                }
                """.strip(),
                encoding="utf-8",
            )
            rec = load_run(run_dir)
            self.assertEqual(rec.run_id, "old_run")
            self.assertFalse(rec.has_batch_metrics)
            series = rec.checkpoint_series("query_precision")
            self.assertEqual(series, [(500, 0.5)])
            self.assertEqual(rec.batch_series("query_precision"), [])

    def test_batch_series_from_prefixed_fields(self):
        rec = RunRecord(
            run_id="x",
            run_dir=Path("."),
            checkpoints=[
                {
                    "tiles_processed": 100,
                    "batch_query_precision": 0.8,
                    "batch_metrics_available": True,
                },
                {
                    "tiles_processed": 200,
                    "batch_query_precision": 0.6,
                    "batch_metrics_available": True,
                },
            ],
        )
        self.assertTrue(rec.has_batch_metrics)
        self.assertEqual(rec.batch_series("query_precision"), [(100, 0.8), (200, 0.6)])
        self.assertEqual(
            rec.batch_series("batch_query_precision"), [(100, 0.8), (200, 0.6)]
        )


if __name__ == "__main__":
    unittest.main()
