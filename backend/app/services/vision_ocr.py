import logging
import os
import re
import tempfile
from typing import Optional

import cv2
import numpy as np
import pytesseract

from app.config import settings
from app.graph.llm import ollama_chat

logger = logging.getLogger(__name__)

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

# Resolution standardization: long side scaled into this range, aspect ratio
# preserved. Below the floor, small screenshots get upscaled so text is tall
# enough to read; above the ceiling, huge screenshots get downscaled so the
# vision model isn't fed more detail than it can actually use.
MIN_LONG_SIDE = 768
MAX_LONG_SIDE = 1344
BORDER_SIZE = 20

DEFAULT_VISION_PROMPT = (
    "Analyze this UI screenshot. Produce a markdown map of every visible "
    "element: accessibility label, element type, input constraints, and "
    "layout order (top to bottom, left to right)."
)

# Repeating gibberish/broken-symbol detection: a run of 10+ identical
# characters (Qwen-VL's degenerate token-loop failure mode), or a cluster of
# block/shade/geometric Unicode characters or the U+FFFD replacement
# character (garbled decode artifacts) - either is a strong signal the model
# didn't actually read the image.
_REPEATED_CHAR_RE = re.compile(r"(.)\1{9,}")
_BROKEN_SYMBOL_RE = re.compile(r"[▀-◿�]{3,}")
_WORD_LIKE_RE = re.compile(r"[A-Za-z]{2,}")
_MIN_ALPHA_RATIO = 0.15


def preprocess_image(image_path: str) -> np.ndarray:
    """Resize/pad/binarize an image for maximum text legibility before OCR.

    Order matters: resolution standardization happens before padding so the
    768-1344px target reflects the actual screenshot content, not the border;
    contrast enhancement happens last since it operates on the final
    dimensions. Returns a single-channel (grayscale, binarized) array -
    usable directly by pytesseract, and written out to a temp PNG for the
    vision LLM call in extract_text_from_screenshot.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image at {image_path!r}")

    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side < MIN_LONG_SIDE:
        target = MIN_LONG_SIDE
    elif long_side > MAX_LONG_SIDE:
        target = MAX_LONG_SIDE
    else:
        target = long_side
    if target != long_side:
        scale = target / long_side
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        image = cv2.resize(image, new_size, interpolation=interpolation)

    image = cv2.copyMakeBorder(
        image,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A slight blur before Otsu's binarization smooths out the block-like
    # quantization/compression artifacts a screenshot can pick up, without
    # softening real text edges enough to hurt OCR.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


def _looks_like_gibberish(text: str) -> bool:
    if not text or not text.strip():
        return True
    alpha_count = sum(ch.isalpha() for ch in text)
    if alpha_count == 0:
        return True
    if _REPEATED_CHAR_RE.search(text):
        return True
    if _BROKEN_SYMBOL_RE.search(text):
        return True
    if not _WORD_LIKE_RE.search(text):
        return True
    if alpha_count / len(text) < _MIN_ALPHA_RATIO:
        return True
    return False


async def extract_text_from_screenshot(
    image_path: str,
    model: Optional[str] = None,
    prompt: str = DEFAULT_VISION_PROMPT,
) -> str:
    """Dual-layer fail-safe extraction: try the vision LLM first, and fall
    back to deterministic OCR (pytesseract) if it throws or its output looks
    like garbage. Both paths run against the same preprocessed image, so the
    fallback isn't handicapped by a worse-quality source than the primary
    path got."""
    resolved_model = model or settings.vision_model
    processed = preprocess_image(image_path)

    vision_text = ""
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(temp_path, processed)
        vision_text = await ollama_chat(resolved_model, prompt, images=[temp_path])
    except Exception:
        logger.warning(
            "extract_text_from_screenshot: vision model call failed for %s, "
            "falling back to OCR",
            image_path,
            exc_info=True,
        )
        vision_text = ""
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if isinstance(vision_text, str) and not _looks_like_gibberish(vision_text):
        return vision_text

    logger.warning(
        "extract_text_from_screenshot: vision output for %s failed the anomaly "
        "check, falling back to pytesseract",
        image_path,
    )
    return pytesseract.image_to_string(processed)
