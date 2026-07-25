"""
Unit tests for multi-frame tile-explorer selection helpers (headless, no Tk / torch).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from drone_ared.tile_selection import (
    TK_CONTROL_MASK,
    TK_SHIFT_MASK,
    linear_select_tile_indices,
    rect_select_tile_indices,
    tile_explorer_modifier_flags,
)


def _tile(row: int, col: int):
    return SimpleNamespace(tile_row=row, tile_col=col)


def _grid(rows: int, cols: int):
    """Row-major list of tiles covering [0..rows) x [0..cols)."""
    return [_tile(r, c) for r in range(rows) for c in range(cols)]


class TestModifierFlags(unittest.TestCase):
    def test_none(self):
        ctrl, shift = tile_explorer_modifier_flags(0)
        self.assertFalse(ctrl)
        self.assertFalse(shift)

    def test_shift_only(self):
        ctrl, shift = tile_explorer_modifier_flags(TK_SHIFT_MASK)
        self.assertFalse(ctrl)
        self.assertTrue(shift)

    def test_ctrl_only(self):
        ctrl, shift = tile_explorer_modifier_flags(TK_CONTROL_MASK)
        self.assertTrue(ctrl)
        self.assertFalse(shift)

    def test_ctrl_and_shift(self):
        ctrl, shift = tile_explorer_modifier_flags(TK_CONTROL_MASK | TK_SHIFT_MASK)
        self.assertTrue(ctrl)
        self.assertTrue(shift)

    def test_none_from_none_state(self):
        ctrl, shift = tile_explorer_modifier_flags(None)
        self.assertFalse(ctrl)
        self.assertFalse(shift)


class TestLinearSelect(unittest.TestCase):
    """Shift+click uses list order — critical when the range crosses frame rows."""

    def test_empty(self):
        self.assertEqual(linear_select_tile_indices(0, 0, 0), set())

    def test_out_of_range(self):
        self.assertEqual(linear_select_tile_indices(4, -1, 2), set())
        self.assertEqual(linear_select_tile_indices(4, 0, 99), set())

    def test_same_cell(self):
        self.assertEqual(linear_select_tile_indices(9, 4, 4), {4})

    def test_forward_range(self):
        self.assertEqual(linear_select_tile_indices(10, 2, 5), {2, 3, 4, 5})

    def test_backward_range(self):
        self.assertEqual(linear_select_tile_indices(10, 5, 2), {2, 3, 4, 5})

    def test_cross_row_keeps_full_previous_row(self):
        """4-col grid: idx 0 = (0,0), idx 4 = (1,0). Linear keeps all of row 0 + first of row 1.

        Spatial rect from (0,0)→(1,0) would only keep column 0 (the old bug).
        """
        cols = 4
        n = 4 * 3  # 3 rows
        anchor = 0
        end = 1 * cols + 0  # first tile of next frame row
        got = linear_select_tile_indices(n, anchor, end)
        self.assertEqual(got, set(range(0, end + 1)))
        self.assertEqual(len(got), 5)  # full first row (4) + first of second

        # Contrast: spatial box collapses to the thin column
        tiles = _grid(3, cols)
        spatial = rect_select_tile_indices(tiles, anchor, end)
        self.assertEqual(spatial, {0, 4})  # only col 0 on both rows

    def test_first_five_of_row_three_from_row_anchor(self):
        """Ctrl+click start of row 3, Shift+click 5th tile → only those five."""
        cols = 8
        row3_start = 3 * cols + 0
        row3_fifth = 3 * cols + 4
        n = 5 * cols
        self.assertEqual(
            linear_select_tile_indices(n, row3_start, row3_fifth),
            set(range(row3_start, row3_fifth + 1)),
        )


class TestRectSelect(unittest.TestCase):
    """Spatial helper still available; not the default Shift path."""

    def test_empty_tiles(self):
        self.assertEqual(rect_select_tile_indices([], 0, 0), set())

    def test_2x3_box(self):
        tiles = _grid(4, 4)
        anchor = 1 * 4 + 0
        end = 2 * 4 + 2
        got = rect_select_tile_indices(tiles, anchor, end)
        expected = {r * 4 + c for r in (1, 2) for c in (0, 1, 2)}
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
