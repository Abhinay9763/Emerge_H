from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """Apply lightweight normalization for manuscript OCR input."""
    gray = ImageOps.grayscale(image)
    autocontrast = ImageOps.autocontrast(gray)
    normalized = autocontrast.convert("RGB")
    return np.asarray(normalized)
