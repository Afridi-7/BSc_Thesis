"""Render an annotated overlay (boxes + labels) over an input image.

Kept dependency-light: PIL only. The detector already returns
``[{xyxy, class, confidence}, ...]`` so we don't re-run YOLO here.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, Iterable, Tuple

from PIL import Image, ImageDraw, ImageFont


# Class -> RGB. Mirrors config.yaml `detection.visualization.colors`.
_DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "WBC": (255, 107, 107),
    "RBC": (78, 205, 196),
    "Platelet": (149, 225, 211),
}
_FALLBACK_COLOR = (255, 215, 0)


def _color_for(cls: str) -> Tuple[int, int, int]:
    return _DEFAULT_COLORS.get(cls, _FALLBACK_COLOR)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort font loader; falls back to default bitmap font."""

    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except (OSError, IOError):
        return ImageFont.load_default()


def render_overlay(image: Image.Image, boxes: Iterable[dict]) -> bytes:
    """Draw boxes/labels over ``image`` and return PNG bytes."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    # Scale the line width with the image so it stays visible on big smears.
    line_width = max(2, min(canvas.size) // 400)
    font = _load_font(size=max(12, min(canvas.size) // 60))

    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        cls = box.get("class", "?")
        conf = float(box.get("confidence", 0.0))
        color = _color_for(cls)

        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)

        label = f"{cls} {conf:.2f}"
        # PIL >= 9.2 uses textbbox; older versions fall back to textsize.
        try:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:  # pragma: no cover
            text_w, text_h = draw.textsize(label, font=font)

        pad = 2
        bg_y0 = max(0, y1 - text_h - 2 * pad)
        draw.rectangle(
            [(x1, bg_y0), (x1 + text_w + 2 * pad, bg_y0 + text_h + 2 * pad)],
            fill=color,
        )
        draw.text((x1 + pad, bg_y0 + pad), label, fill=(0, 0, 0), font=font)

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_overlay_base64(image: Image.Image, boxes: Iterable[dict]) -> str:
    """Convenience: ``data:image/png;base64,...`` payload for the frontend."""

    png_bytes = render_overlay(image, boxes)
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
