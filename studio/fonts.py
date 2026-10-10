"""Fonts for the rendered videos: the macOS faces when present, DejaVu in the Linux container, Pillow's default last."""

from __future__ import annotations

from PIL import ImageFont

CANDIDATES = {
    "display": ("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"),
    "mono": ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    "body": ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
}


def font(kind: str, size: int):
    for path in CANDIDATES[kind]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)
