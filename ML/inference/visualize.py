from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw


def render_ocr_overlay(image: Image.Image, boxes: Sequence[Sequence[Sequence[float]]]) -> Image.Image:
    """Draw OCR bounding polygons for demo/debug artifacts."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for box in boxes:
        if len(box) < 4:
            continue
        points = [(float(p[0]), float(p[1])) for p in box]
        draw.line(points + [points[0]], fill=(255, 64, 64), width=2)

    return canvas
