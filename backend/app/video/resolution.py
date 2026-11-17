"""Naming a picture size the way people do (720p, 1080p, 1440p, 4K) and deciding which sizes are honest to offer."""

from __future__ import annotations

# (name, label, smallest shorter side, smallest longer side). A size gets the highest class either side reaches, so
# 3840x1600 (ultrawide) is "4K" like 3840x2160, and 1080x1920 (portrait) is "1080p" like 1920x1080.
_CLASSES = (
    ("4k", "4K Ultra HD", 2160, 3840),
    ("1440p", "1440p QHD", 1440, 2560),
    ("1080p", "1080p Full HD", 1080, 1920),
    ("720p", "720p HD", 720, 1280),
    ("480p", "480p SD", 480, 854),
)
EXPORT_TARGETS = (2160, 1440, 1080, 720)  # shorter side, in pixels
_TARGET_NAMES = {2160: "4K Ultra HD", 1440: "1440p QHD", 1080: "1080p Full HD", 720: "720p HD"}


def resolution_class(width: int, height: int) -> tuple[str, str]:
    """(short name, label) for a picture size, e.g. (\"4k\", \"4K Ultra HD\")."""
    short, long = min(width, height), max(width, height)
    for name, label, min_short, min_long in _CLASSES:
        if short >= min_short * 0.97 or long >= min_long * 0.97:
            return name, label
    return f"{short}p", f"{short}p"


def target_label(target: int) -> str:
    return _TARGET_NAMES.get(target, f"{target}p")


def valid_targets(width: int, height: int) -> list[int]:
    """Sizes a video can honestly be exported at: strictly smaller than it is (an enlargement would only stretch the pixels
    that are there), named by their shorter side."""
    short = min(width, height)
    return [target for target in EXPORT_TARGETS if target < short]
