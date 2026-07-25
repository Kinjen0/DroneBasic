"""
Pure helpers for multi-tile selection in the Multi-Frame tile explorer.

No Tk / torch / pipeline imports — safe for headless unit tests.
Used only by the GUI tile explorer; does not touch A_RED.
"""

from __future__ import annotations
from typing import Any, Sequence, Set, Tuple

# Tk/X11 event.state masks (also used on Windows/macOS Tk for Button-1 modifiers).
TK_SHIFT_MASK = 0x0001
TK_CONTROL_MASK = 0x0004


def tile_explorer_modifier_flags(event_state: Any) -> Tuple[bool, bool]:
    """Return ``(ctrl, shift)`` from a Tk ``event.state`` bitmask."""
    try:
        state = int(event_state or 0)
    except (TypeError, ValueError):
        state = 0
    return bool(state & TK_CONTROL_MASK), bool(state & TK_SHIFT_MASK)


def linear_select_tile_indices(n_tiles: int, anchor_idx: int, end_idx: int) -> Set[int]:
    """Inclusive index range in list / display order (row-major tile list).

    This matches how the explorer lays tiles out (left→right, top→bottom) and is what
    users expect when Shift+clicking "down through rows". Unlike a spatial (row,col)
    bounding box, selecting from the start of row 0 to the start of row 1 keeps every
    tile in between instead of collapsing to a single column.
    """
    n = int(n_tiles)
    if n <= 0:
        return set()
    if anchor_idx < 0 or end_idx < 0 or anchor_idx >= n or end_idx >= n:
        return set()
    lo, hi = (anchor_idx, end_idx) if anchor_idx <= end_idx else (end_idx, anchor_idx)
    return set(range(lo, hi + 1))


def rect_select_tile_indices(tiles: Sequence[Any], anchor_idx: int, end_idx: int) -> Set[int]:
    """Indices in the axis-aligned ``(tile_row, tile_col)`` box between two indices.

    Kept for callers that want a true spatial section of the *frame* grid. The default
    explorer Shift+click path uses :func:`linear_select_tile_indices` instead, because
    a thin box from (r0,c0)→(r1,0) drops most of the previous row.
    """
    n = len(tiles)
    if n == 0:
        return set()
    if anchor_idx < 0 or end_idx < 0 or anchor_idx >= n or end_idx >= n:
        return set()
    a = tiles[anchor_idx]
    b = tiles[end_idx]
    r0, r1 = sorted((int(a.tile_row), int(b.tile_row)))
    c0, c1 = sorted((int(a.tile_col), int(b.tile_col)))
    out: Set[int] = set()
    for i, t in enumerate(tiles):
        tr, tc = int(t.tile_row), int(t.tile_col)
        if r0 <= tr <= r1 and c0 <= tc <= c1:
            out.add(i)
    return out
