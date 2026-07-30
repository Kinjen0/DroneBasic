"""Smoke test for SeaDronesSeeRunner with a synthetic mini dataset."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from drone_ared.seadronesee.config import SeaDronesSeeConfig
from drone_ared.seadronesee.runner import SeaDronesSeeRunner


def _write_mini_dataset(root: Path) -> None:
    """64x64 image with a 20x20 boat box; one empty image."""
    (root / "annotations").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    # Image with object
    img1 = np.zeros((64, 64, 3), dtype=np.uint8)
    img1[:, :] = (30, 80, 160)  # water-ish
    img1[20:40, 20:40] = (200, 200, 50)
    Image.fromarray(img1).save(root / "images" / "train" / "1.jpg")
    # Empty water
    img2 = np.zeros((64, 64, 3), dtype=np.uint8)
    img2[:, :] = (30, 80, 160)
    Image.fromarray(img2).save(root / "images" / "train" / "2.jpg")

    coco = {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 0, "name": "ignored"},
            {"id": 2, "name": "boat"},
        ],
        "images": [
            {"id": 1, "file_name": "1.jpg", "width": 64, "height": 64},
            {"id": 2, "file_name": "2.jpg", "width": 64, "height": 64},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [20, 20, 20, 20], "area": 400},
        ],
    }
    (root / "annotations" / "instances_train.json").write_text(json.dumps(coco), encoding="utf-8")
    # empty val so dataset doesn't break if probed
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "instances_val.json").write_text(
        json.dumps({"categories": coco["categories"], "images": [], "annotations": []}),
        encoding="utf-8",
    )


class TestSDSRunnerSmoke(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="sds_smoke_"))
        self.ds_root = self.tmp / "export"
        _write_mini_dataset(self.ds_root)
        self.db_path = str(self.tmp / "sds.db")
        self.model_path = str(self.tmp / "model.pkl")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_raw_run_and_save_load(self):
        cfg = SeaDronesSeeConfig.default()
        cfg.dataset_root = str(self.ds_root)
        cfg.split = "train"
        cfg.feature_mode = "raw"
        cfg.tiling.tile_width = 32
        cfg.tiling.tile_height = 32
        cfg.tiling.stride_x = 32
        cfg.tiling.stride_y = 32
        cfg.tile_annotations_db = self.db_path
        cfg.metrics_logging.enabled = True
        cfg.metrics_logging.output_dir = str(self.tmp / "runs")
        cfg.metrics_logging.checkpoint_every = 50
        cfg.ared.kappa = 2.0  # a bit more queries

        runner = SeaDronesSeeRunner(cfg)
        runner.start()
        t0 = time.time()
        while runner.is_running() and time.time() - t0 < 60:
            time.sleep(0.05)
        self.assertIn(runner.stats.get("status"), ("finished", "stopped"))
        self.assertGreater(runner.stats["tiles_processed"], 0)
        # 2 images * 4 tiles each (64/32)^2 = 8
        self.assertEqual(runner.stats["tiles_processed"], 8)
        self.assertGreater(runner.stats["gt_positives"], 0)
        self.assertGreater(runner.stats["gt_negatives"], 0)
        self.assertEqual(len(runner.processed_identities), 8)

        # GT written to DB
        n = runner.tile_db.get_annotation_count()
        self.assertEqual(n, 8)

        # Save / load model
        runner.save_ared_state(self.model_path)
        self.assertTrue(Path(self.model_path).is_file())

        runner2 = SeaDronesSeeRunner(cfg)
        runner2.load_ared_state(self.model_path)
        self.assertIsNotNone(runner2.ared_adapter)
        known = runner2.ared_adapter.get_known_labels()
        self.assertTrue(len(known) >= 1)

        # Metrics run dir created
        rd = runner.stats.get("metrics_run_dir")
        self.assertTrue(rd and Path(rd).is_dir())


if __name__ == "__main__":
    unittest.main()
