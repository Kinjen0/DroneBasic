"""Unit tests for BBoxTileLabeler (COCO → tile GT)."""

from __future__ import annotations

import unittest

from drone_ared.seadronesee.coco_index import CocoBox
from drone_ared.seadronesee.config import TileLabelConfig
from drone_ared.seadronesee.tile_labeler import BBoxTileLabeler


def _box(cid, name, x, y, w, h, ann_id=1):
    return CocoBox(
        ann_id=ann_id,
        image_id=1,
        category_id=cid,
        category_name=name,
        x=x,
        y=y,
        w=w,
        h=h,
    )


class TestBBoxTileLabeler(unittest.TestCase):
    def test_no_overlap_is_water(self):
        lab = BBoxTileLabeler()
        boxes = [_box(1, "swimmer", 100, 100, 20, 20)]
        r = lab.label_tile((0, 0, 32, 32), boxes)
        self.assertEqual(r.label, "water")
        self.assertFalse(r.relevant)
        self.assertEqual(r.n_overlapping_boxes, 0)

    def test_full_containment(self):
        lab = BBoxTileLabeler()
        boxes = [_box(1, "swimmer", 5, 5, 10, 10)]
        r = lab.label_tile((0, 0, 32, 32), boxes)
        self.assertEqual(r.label, "swimmer")
        self.assertTrue(r.relevant)
        self.assertGreater(r.overlap_area, 0)

    def test_partial_edge_overlap(self):
        lab = BBoxTileLabeler()
        # box straddles right edge of tile
        boxes = [_box(2, "boat", 30, 10, 20, 20)]
        r = lab.label_tile((0, 0, 32, 32), boxes)
        self.assertEqual(r.label, "boat")
        self.assertTrue(r.relevant)

    def test_largest_intersection_wins(self):
        lab = BBoxTileLabeler()
        boxes = [
            _box(1, "swimmer", 0, 0, 5, 5, ann_id=1),   # area 25
            _box(2, "boat", 0, 0, 20, 20, ann_id=2),    # area 400
        ]
        r = lab.label_tile((0, 0, 32, 32), boxes)
        self.assertEqual(r.label, "boat")
        self.assertEqual(r.n_overlapping_boxes, 2)

    def test_tie_break_lower_category_id(self):
        lab = BBoxTileLabeler()
        # equal overlap area 100
        boxes = [
            _box(5, "buoy", 0, 0, 10, 10, ann_id=1),
            _box(1, "swimmer", 0, 0, 10, 10, ann_id=2),
        ]
        r = lab.label_tile((0, 0, 32, 32), boxes)
        self.assertEqual(r.label, "swimmer")
        self.assertEqual(r.winning_category_id, 1)

    def test_ignored_only_is_water(self):
        lab = BBoxTileLabeler()
        boxes = [_box(0, "ignored", 0, 0, 30, 30)]
        r = lab.label_tile((0, 0, 32, 32), boxes)
        self.assertEqual(r.label, "water")
        self.assertFalse(r.relevant)

    def test_min_overlap_frac_rejects_glance(self):
        cfg = TileLabelConfig(min_overlap_frac_of_tile=0.5)
        lab = BBoxTileLabeler(cfg)
        # 2x2 overlap on 32x32 tile = 4/1024 << 0.5
        boxes = [_box(1, "swimmer", 31, 31, 10, 10)]
        r = lab.label_tile((0, 0, 32, 32), boxes)
        self.assertEqual(r.label, "water")

    def test_relevant_category_filter(self):
        cfg = TileLabelConfig(relevant_categories=["swimmer"])
        lab = BBoxTileLabeler(cfg)
        r = lab.label_tile((0, 0, 32, 32), [_box(2, "boat", 0, 0, 20, 20)])
        self.assertEqual(r.label, "boat")
        self.assertFalse(r.relevant)
        r2 = lab.label_tile((0, 0, 32, 32), [_box(1, "swimmer", 0, 0, 20, 20)])
        self.assertEqual(r2.label, "swimmer")
        self.assertTrue(r2.relevant)


if __name__ == "__main__":
    unittest.main()
