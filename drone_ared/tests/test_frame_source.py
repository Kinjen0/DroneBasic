"""
Unit tests for FrameSource (image sequences + factory).

Headless: no Tk, no A_RED, no real video decode required for the image path.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from drone_ared.frame_source import (
    ImageSequenceFrameSource,
    VideoFileFrameSource,
    list_image_files,
    natural_sort_key,
    open_frame_source,
)
from drone_ared.tile_database import resolve_video_file


def _write_rgb(path: Path, color=(10, 20, 30), size=(8, 6)):
    img = Image.new("RGB", size, color)
    # PNG is lossless so pixel colors round-trip exactly in tests.
    img.save(path)


class TestNaturalSort(unittest.TestCase):
    def test_numeric_order(self):
        names = ["img10.jpg", "img2.jpg", "img1.jpg"]
        ordered = sorted(names, key=natural_sort_key)
        self.assertEqual(ordered, ["img1.jpg", "img2.jpg", "img10.jpg"])


class TestImageSequenceFrameSource(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        # Deliberately out of lexical order: 1, 10, 2 (PNG for exact colors)
        _write_rgb(self.dir / "img_1.png", (1, 0, 0))
        _write_rgb(self.dir / "img_10.png", (10, 0, 0))
        _write_rgb(self.dir / "img_2.png", (2, 0, 0))
        # Non-image noise should be ignored
        (self.dir / "notes.txt").write_text("ignore me")

    def tearDown(self):
        self._tmp.cleanup()

    def test_list_and_natural_order(self):
        files = list_image_files(self.dir)
        self.assertEqual([p.name for p in files], ["img_1.png", "img_2.png", "img_10.png"])

    def test_frame_count_and_identity(self):
        src = ImageSequenceFrameSource(self.dir)
        self.assertEqual(src.frame_count, 3)
        self.assertEqual(Path(src.identity_path).name, self.dir.name)
        src.release()

    def test_sequential_read(self):
        src = ImageSequenceFrameSource(self.dir)
        ok1, f1 = src.read()
        ok2, f2 = src.read()
        ok3, f3 = src.read()
        ok4, f4 = src.read()
        self.assertTrue(ok1 and ok2 and ok3)
        self.assertFalse(ok4)
        self.assertIsNone(f4)
        # First pixel roughly matches written solid colors
        self.assertEqual(tuple(f1[0, 0].tolist()), (1, 0, 0))
        self.assertEqual(tuple(f2[0, 0].tolist()), (2, 0, 0))
        self.assertEqual(tuple(f3[0, 0].tolist()), (10, 0, 0))
        self.assertEqual(f1.dtype, np.uint8)
        self.assertEqual(f1.ndim, 3)
        src.release()

    def test_seek_and_read_frame(self):
        src = ImageSequenceFrameSource(self.dir)
        self.assertTrue(src.seek(2))
        ok, frame = src.read()
        self.assertTrue(ok)
        self.assertEqual(tuple(frame[0, 0].tolist()), (10, 0, 0))

        mid = src.read_frame(1)
        self.assertIsNotNone(mid)
        self.assertEqual(tuple(mid[0, 0].tolist()), (2, 0, 0))

        self.assertFalse(src.seek(-1))
        self.assertFalse(src.seek(99))
        self.assertIsNone(src.read_frame(99))
        src.release()

    def test_empty_directory_raises(self):
        empty = self.dir / "empty_sub"
        empty.mkdir()
        with self.assertRaises(OSError):
            ImageSequenceFrameSource(empty)

    def test_open_frame_source_directory(self):
        src = open_frame_source(self.dir)
        self.assertIsInstance(src, ImageSequenceFrameSource)
        self.assertEqual(src.frame_count, 3)
        src.release()

    def test_context_manager(self):
        with open_frame_source(self.dir) as src:
            ok, frame = src.read()
            self.assertTrue(ok)
            self.assertIsNotNone(frame)


class TestResolveMediaPath(unittest.TestCase):
    def test_resolve_directory_by_basename(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = root / "train_split"
            folder.mkdir()
            _write_rgb(folder / "a.png")
            hit = resolve_video_file(
                "train_split",
                search_paths=[str(folder)],
                search_dirs=[str(root)],
            )
            self.assertIsNotNone(hit)
            self.assertTrue(Path(hit).is_dir())
            self.assertEqual(Path(hit).name, "train_split")

    def test_resolve_existing_directory_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folder = root / "frames"
            folder.mkdir()
            hit = resolve_video_file(str(folder))
            self.assertEqual(Path(hit).resolve(), folder.resolve())


class TestOpenFrameSourceFactory(unittest.TestCase):
    def test_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            open_frame_source("/no/such/path/ever_12345")

    def test_video_file_type_when_cv2_available(self):
        """If a tiny invalid 'video' path is a file, factory still picks VideoFileFrameSource
        (open may fail — we only check routing when the file exists and is non-video).
        """
        # We cannot reliably create a real mp4 without ffmpeg; just assert that a
        # non-directory file goes to VideoFileFrameSource constructor path.
        # Use open_frame_source on a non-image file: it will try VideoFileFrameSource.
        with tempfile.TemporaryDirectory() as td:
            junk = Path(td) / "not_a_video.bin"
            junk.write_bytes(b"\x00\x01\x02")
            try:
                src = open_frame_source(junk)
            except (OSError, RuntimeError):
                # Expected: cv2 cannot open garbage, or cv2 missing
                return
            self.assertIsInstance(src, VideoFileFrameSource)
            src.release()


if __name__ == "__main__":
    unittest.main()
