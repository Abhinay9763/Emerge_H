from __future__ import annotations

import threading
from typing import Any

from PIL import Image

from .preprocess import preprocess_for_ocr
from .visualize import render_ocr_overlay

_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
_MODEL_VERSION = "paddleocr-te-v1"


def _get_ocr_model() -> Any:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is None:
            from paddleocr import PaddleOCR

            _MODEL = PaddleOCR(use_angle_cls=True, lang="te", show_log=False)

    return _MODEL


def _extract_lines(raw_result: Any) -> tuple[list[str], list[float], list[list[list[float]]]]:
    lines: list[str] = []
    confidences: list[float] = []
    boxes: list[list[list[float]]] = []

    if not raw_result:
        return lines, confidences, boxes

    result_blocks = raw_result[0] if isinstance(raw_result, list) else raw_result
    if not isinstance(result_blocks, list):
        return lines, confidences, boxes

    for item in result_blocks:
        if not isinstance(item, list) or len(item) < 2:
            continue

        box = item[0]
        rec = item[1]
        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            continue

        text = str(rec[0]).strip()
        score = float(rec[1])
        if not text:
            continue

        lines.append(text)
        confidences.append(score)
        boxes.append(box)

    return lines, confidences, boxes


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def predict_page(image: Image.Image) -> dict[str, Any]:
    """Run Telugu OCR and return contract-compatible payload for Backend."""
    ocr = _get_ocr_model()
    processed = preprocess_for_ocr(image)
    raw_result = ocr.ocr(processed, cls=True)

    lines, confidences, boxes = _extract_lines(raw_result)

    telugu = _normalize_text("\n".join(lines))
    confidence = sum(confidences) / len(confidences) if confidences else 0.0

    overlay_image = render_ocr_overlay(image, boxes)

    return {
        "telugu": telugu,
        "english": "translation_unavailable",
        "confidence": round(confidence, 4),
        "model_version": _MODEL_VERSION,
        "ocr_image": overlay_image,
    }
