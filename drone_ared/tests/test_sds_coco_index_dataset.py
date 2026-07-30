"""Tests for COCO index + dataset loader (uses real export when present)."""

from __future__ import annotations

import unittest
from pathlib import Path

from drone_ared.seadronesee.coco_index import CocoAnnotationIndex
from drone_ared.seadronesee.dataset import SeaDronesSeeDataset

ROOT = Path(__file__).resolve().parents[2] / "SeaDroneSeeProcessedDataExport"
HAS_EXPORT = ROOT.is_dir() and (ROOT / "annotations" / "instances_train.json").is_file()


@unittest.skipUnless(HAS_EXPORT, "SeaDroneSeeProcessedDataExport not present")
class TestRealExport(unittest.TestCase):
    def test_train_counts(self):
        ds = SeaDronesSeeDataset(ROOT)
        idx, imgs = ds.load_split("train")
        self.assertEqual(len(imgs), 70)
        self.assertEqual(idx.num_annotations, 158)

    def test_val_counts(self):
        ds = SeaDronesSeeDataset(ROOT)
        idx, imgs = ds.load_split("val")
        self.assertEqual(len(imgs), 18)
        self.assertEqual(idx.num_annotations, 33)

    def test_identity_is_filename(self):
        ds = SeaDronesSeeDataset(ROOT)
        _, imgs = ds.load_split("train")
        self.assertTrue(imgs[0].identity.endswith(".jpg"))
        self.assertNotIn("/", imgs[0].identity)

    def test_boxes_for_image(self):
        ds = SeaDronesSeeDataset(ROOT)
        idx, imgs = ds.load_split("train")
        # find an image that has boxes
        hit = None
        for im in imgs:
            b = idx.get_boxes(im.image_id)
            if b:
                hit = b
                break
        self.assertIsNotNone(hit)
        self.assertGreater(hit[0].w, 0)
        self.assertIn(hit[0].category_name, {
            "ignored", "swimmer", "boat", "jetski", "life_saving_appliances", "buoy"
        })

    def test_iter_max_images(self):
        ds = SeaDronesSeeDataset(ROOT)
        got = list(ds.iter_images("train", max_images=3))
        self.assertEqual(len(got), 3)


class TestCocoIndexSynthetic(unittest.TestCase):
    def test_from_dict(self):
        coco = {
            "categories": [{"id": 1, "name": "swimmer"}],
            "images": [{"id": 10, "file_name": "a.jpg", "width": 100, "height": 80}],
            "annotations": [
                {"id": 1, "image_id": 10, "category_id": 1, "bbox": [1, 2, 3, 4]},
            ],
        }
        idx = CocoAnnotationIndex(coco)
        self.assertEqual(idx.num_images, 1)
        self.assertEqual(idx.num_annotations, 1)
        b = idx.get_boxes(10)[0]
        self.assertEqual(b.category_name, "swimmer")
        self.assertEqual(b.xyxy, (1.0, 2.0, 4.0, 6.0))


if __name__ == "__main__":
    unittest.main()
