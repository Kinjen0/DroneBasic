"""
Terminal logging helpers.

`vprint` is for *repeating* progress noise (per-tile, per-frame, cache hits, etc.).
Always-on `print` stays for start/stop, errors, and one-shot events.

GUI checkbox "Terminal logging" toggles this via set_terminal_logging().
"""

from __future__ import annotations

# Default ON so existing debugging behavior is preserved until the user unchecks.
_TERMINAL_VERBOSE: bool = True


def set_terminal_logging(enabled: bool) -> None:
    global _TERMINAL_VERBOSE
    _TERMINAL_VERBOSE = bool(enabled)


def terminal_logging_enabled() -> bool:
    return bool(_TERMINAL_VERBOSE)


def vprint(*args, **kwargs) -> None:
    """Print only when terminal logging (repeating messages) is enabled."""
    if _TERMINAL_VERBOSE:
        print(*args, **kwargs)
