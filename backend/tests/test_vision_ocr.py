import cv2
import numpy as np

from app.services import vision_ocr


def _write_test_image(path, width=2000, height=100):
    """A wide, short synthetic image - like a long single-line screenshot -
    to exercise the long-side resize logic against a real cv2 pipeline."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (width - 10, height - 10), (0, 0, 0), 2)
    cv2.imwrite(path, image)


def test_preprocess_image_resizes_long_side_into_range(tmp_path):
    path = str(tmp_path / "wide.png")
    _write_test_image(path, width=2000, height=100)

    result = vision_ocr.preprocess_image(path)

    # Padding is applied after resizing, so the long side (minus the 2x
    # border) must land inside [MIN_LONG_SIDE, MAX_LONG_SIDE].
    long_side_before_padding = max(result.shape) - 2 * vision_ocr.BORDER_SIZE
    assert vision_ocr.MIN_LONG_SIDE <= long_side_before_padding <= vision_ocr.MAX_LONG_SIDE


def test_preprocess_image_upscales_small_images(tmp_path):
    path = str(tmp_path / "tiny.png")
    _write_test_image(path, width=200, height=50)

    result = vision_ocr.preprocess_image(path)

    long_side_before_padding = max(result.shape) - 2 * vision_ocr.BORDER_SIZE
    assert long_side_before_padding == vision_ocr.MIN_LONG_SIDE


def test_preprocess_image_adds_border_and_binarizes(tmp_path):
    path = str(tmp_path / "square.png")
    _write_test_image(path, width=800, height=800)

    result = vision_ocr.preprocess_image(path)

    assert result.ndim == 2  # grayscale, single channel
    assert result.shape[0] == result.shape[1] == 800 + 2 * vision_ocr.BORDER_SIZE
    # Otsu's binarization must produce a strictly two-valued image.
    assert set(np.unique(result).tolist()) <= {0, 255}


def test_preprocess_image_raises_on_unreadable_path(tmp_path):
    missing = str(tmp_path / "does-not-exist.png")
    try:
        vision_ocr.preprocess_image(missing)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_looks_like_gibberish_detects_known_failure_modes():
    assert vision_ocr._looks_like_gibberish("") is True
    assert vision_ocr._looks_like_gibberish("   ") is True
    assert vision_ocr._looks_like_gibberish("12345 67890 !!! ???") is True  # zero alpha chars
    assert vision_ocr._looks_like_gibberish("▒▒▒▒▒▒▒▒▒▒") is True
    assert vision_ocr._looks_like_gibberish("aaaaaaaaaaaaaaaa") is True  # repeated char run
    assert vision_ocr._looks_like_gibberish("����") is True


def test_looks_like_gibberish_accepts_normal_text():
    assert (
        vision_ocr._looks_like_gibberish(
            "## Login Form\n- Email field: required, email format\n- Submit button"
        )
        is False
    )


async def test_extract_text_from_screenshot_uses_vision_output_when_clean(monkeypatch, tmp_path):
    path = str(tmp_path / "shot.png")
    _write_test_image(path)

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        return "## Clean vision output\n- Field: username"

    ocr_calls = {"count": 0}

    def fake_image_to_string(image):
        ocr_calls["count"] += 1
        return "should not be used"

    monkeypatch.setattr(vision_ocr, "ollama_chat", fake_ollama_chat)
    monkeypatch.setattr(vision_ocr.pytesseract, "image_to_string", fake_image_to_string)

    result = await vision_ocr.extract_text_from_screenshot(path, model="qwen2.5vl:7b")

    assert result == "## Clean vision output\n- Field: username"
    assert ocr_calls["count"] == 0


async def test_extract_text_from_screenshot_falls_back_when_vision_raises(monkeypatch, tmp_path):
    path = str(tmp_path / "shot.png")
    _write_test_image(path)

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        raise RuntimeError("Ollama connection refused")

    def fake_image_to_string(image):
        return "OCR fallback text"

    monkeypatch.setattr(vision_ocr, "ollama_chat", fake_ollama_chat)
    monkeypatch.setattr(vision_ocr.pytesseract, "image_to_string", fake_image_to_string)

    result = await vision_ocr.extract_text_from_screenshot(path, model="qwen2.5vl:7b")

    assert result == "OCR fallback text"


async def test_extract_text_from_screenshot_falls_back_when_vision_output_is_gibberish(
    monkeypatch, tmp_path
):
    path = str(tmp_path / "shot.png")
    _write_test_image(path)

    async def fake_ollama_chat(model, prompt, images=None, expect_json=False):
        return "▒░▒░▒░▒░▒░▒░"

    def fake_image_to_string(image):
        return "OCR fallback text"

    monkeypatch.setattr(vision_ocr, "ollama_chat", fake_ollama_chat)
    monkeypatch.setattr(vision_ocr.pytesseract, "image_to_string", fake_image_to_string)

    result = await vision_ocr.extract_text_from_screenshot(path, model="qwen2.5vl:7b")

    assert result == "OCR fallback text"
