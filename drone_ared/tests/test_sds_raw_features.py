"""Tests for RawPixelFeatureExtractor."""

from __future__ import annotations

import unittest
import numpy as np
from PIL import Image

from drone_ared.seadronesee.feature_backends import (
    RawPixelFeatureExtractor,
    build_feature_extractor,
)
from drone_ared.config import TilingConfig
from drone_ared.seadronesee.config import RawFeatureConfig


class TestRawPixelFeatures(unittest.TestCase):
    def test_shape_rgb(self):
        fe = RawPixelFeatureExtractor(8, 8, grayscale=False, scale_to_unit=True, l2_normalize=False)
        img = Image.new("RGB", (8, 8), color=(255, 0, 0))
        out = fe.extract_images([img, img])
        self.assertEqual(out.shape, (2, 8 * 8 * 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=5)

    def test_grayscale_dim(self):
        fe = RawPixelFeatureExtractor(4, 4, grayscale=True, l2_normalize=False)
        img = Image.new("RGB", (4, 4), color=(10, 20, 30))
        out = fe.extract_images([img])
        self.assertEqual(out.shape, (1, 16))

    def test_l2_normalize(self):
        fe = RawPixelFeatureExtractor(4, 4, l2_normalize=True, scale_to_unit=True)
        img = Image.new("RGB", (4, 4), color=(128, 64, 32))
        out = fe.extract_images([img])[0]
        n = float(np.linalg.norm(out))
        self.assertAlmostEqual(n, 1.0, places=5)

    def test_deterministic(self):
        fe = RawPixelFeatureExtractor(4, 4)
        img = Image.new("RGB", (4, 4), color=(7, 8, 9))
        a = fe.extract_images([img])
        b = fe.extract_images([img])
        np.testing.assert_array_equal(a, b)

    def test_wrong_size_raises(self):
        fe = RawPixelFeatureExtractor(8, 8)
        img = Image.new("RGB", (4, 4), color=(0, 0, 0))
        with self.assertRaises(ValueError):
            fe.extract_images([img])

    def test_factory_raw(self):
        t = TilingConfig(tile_width=16, tile_height=16, stride_x=16, stride_y=16)
        fe = build_feature_extractor("raw", t, raw_cfg=RawFeatureConfig())
        self.assertEqual(fe.output_dim, 16 * 16 * 3)


if __name__ == "__main__":
    unittest.main()
