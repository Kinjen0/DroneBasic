"""
Control-plane label sentinels.

These strings were historically used to unblock the worker (stop / timeout / skip)
but must NEVER be treated as real human class labels:
  - never written to TileAnnotationDB
  - never written to the embedding label store
  - never learned by A/RED (discovered_labels / query_counts / clusters)

Intentional user/system class that IS allowed:
  - "__BACKGROUND__"  (user chose "Mark as Background")
"""

from __future__ import annotations
from typing import Optional


CONTROL_LABEL_SENTINELS = frozenset({
    "__STOPPED__",
    "__TIMEOUT__",
    "__DUPLICATE__",
    "__SKIPPED__",
    "__SKIP__",
    "__UNLABELED__",
    "__UNKNOWN__",
})


class LabelCancelled(Exception):
    """Raised when a label request was cancelled (stop/timeout/skip) rather than answered.

    Callers must not persist or feed this into A/RED as a real class.
    """

    def __init__(self, reason: str = "cancelled"):
        self.reason = reason
        super().__init__(f"LabelCancelled: {reason}")


def is_control_label(label: Optional[str]) -> bool:
    """True if label is a control-plane sentinel (not a real class name)."""
    if label is None:
        return True
    return str(label) in CONTROL_LABEL_SENTINELS


def is_persistable_label(label: Optional[str]) -> bool:
    """True if this label is safe to write to DB / label store / A/RED learning."""
    if not label:
        return False
    return not is_control_label(label)
